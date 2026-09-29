// Lean compiler output
// Module: Aesop.RuleTac.Cases
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Basic public import Aesop.Script.SpecificTactics
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
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_openAbstractMVarsResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* lp_aesop_Aesop_isAppOfUpToDefeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_diffGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_tryCasesS___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_FVarSubst_get(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_decl_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_decl_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_patterns_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_patterns_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___boxed(lean_object**);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_RuleTac_cases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_RuleTac_cases___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_cases___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_cases___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "No matching hypothesis found."};
static const lean_object* lp_aesop_Aesop_RuleTac_cases___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleTac_cases___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_cases___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_cases___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_cases(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_cases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toExpr(lean_object* v_p_1_, lean_object* v_a_2_, lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Meta_openAbstractMVarsResult(v_p_1_, v_a_2_, v_a_3_, v_a_4_, v_a_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_17_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_17_ == 0)
{
v___x_10_ = v___x_7_;
v_isShared_11_ = v_isSharedCheck_17_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_a_8_);
lean_dec(v___x_7_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_17_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v_snd_12_; lean_object* v_snd_13_; lean_object* v___x_15_; 
v_snd_12_ = lean_ctor_get(v_a_8_, 1);
lean_inc(v_snd_12_);
lean_dec(v_a_8_);
v_snd_13_ = lean_ctor_get(v_snd_12_, 1);
lean_inc(v_snd_13_);
lean_dec(v_snd_12_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 0, v_snd_13_);
v___x_15_ = v___x_10_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v_snd_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
else
{
lean_object* v_a_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_25_; 
v_a_18_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_25_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_25_ == 0)
{
v___x_20_ = v___x_7_;
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_a_18_);
lean_dec(v___x_7_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_a_18_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toExpr___boxed(lean_object* v_p_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_aesop_Aesop_CasesPattern_toExpr(v_p_26_, v_a_27_, v_a_28_, v_a_29_, v_a_30_);
lean_dec(v_a_30_);
lean_dec_ref(v_a_29_);
lean_dec(v_a_28_);
lean_dec_ref(v_a_27_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorIdx(lean_object* v_x_33_){
_start:
{
if (lean_obj_tag(v_x_33_) == 0)
{
lean_object* v___x_34_; 
v___x_34_ = lean_unsigned_to_nat(0u);
return v___x_34_;
}
else
{
lean_object* v___x_35_; 
v___x_35_ = lean_unsigned_to_nat(1u);
return v___x_35_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorIdx___boxed(lean_object* v_x_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_aesop_Aesop_CasesTarget_x27_ctorIdx(v_x_36_);
lean_dec_ref(v_x_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(lean_object* v_t_38_, lean_object* v_k_39_){
_start:
{
if (lean_obj_tag(v_t_38_) == 0)
{
lean_object* v_decl_40_; lean_object* v___x_41_; 
v_decl_40_ = lean_ctor_get(v_t_38_, 0);
lean_inc(v_decl_40_);
lean_dec_ref_known(v_t_38_, 1);
v___x_41_ = lean_apply_1(v_k_39_, v_decl_40_);
return v___x_41_;
}
else
{
lean_object* v_ps_42_; lean_object* v___x_43_; 
v_ps_42_ = lean_ctor_get(v_t_38_, 0);
lean_inc_ref(v_ps_42_);
lean_dec_ref_known(v_t_38_, 1);
v___x_43_ = lean_apply_1(v_k_39_, v_ps_42_);
return v___x_43_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim(lean_object* v_motive_44_, lean_object* v_ctorIdx_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_k_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(v_t_46_, v_k_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_ctorElim___boxed(lean_object* v_motive_50_, lean_object* v_ctorIdx_51_, lean_object* v_t_52_, lean_object* v_h_53_, lean_object* v_k_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim(v_motive_50_, v_ctorIdx_51_, v_t_52_, v_h_53_, v_k_54_);
lean_dec(v_ctorIdx_51_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_decl_elim___redArg(lean_object* v_t_56_, lean_object* v_decl_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(v_t_56_, v_decl_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_decl_elim(lean_object* v_motive_59_, lean_object* v_t_60_, lean_object* v_h_61_, lean_object* v_decl_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(v_t_60_, v_decl_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_patterns_elim___redArg(lean_object* v_t_64_, lean_object* v_patterns_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(v_t_64_, v_patterns_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_x27_patterns_elim(lean_object* v_motive_67_, lean_object* v_t_68_, lean_object* v_h_69_, lean_object* v_patterns_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_aesop_Aesop_CasesTarget_x27_ctorElim___redArg(v_t_68_, v_patterns_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(lean_object* v_x_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = l_Lean_Meta_saveState___redArg(v___y_74_, v___y_76_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v_a_79_; lean_object* v_r_80_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_a_79_);
lean_dec_ref_known(v___x_78_, 1);
lean_inc(v___y_76_);
lean_inc_ref(v___y_75_);
lean_inc(v___y_74_);
lean_inc_ref(v___y_73_);
v_r_80_ = lean_apply_5(v_x_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_, lean_box(0));
if (lean_obj_tag(v_r_80_) == 0)
{
lean_object* v_a_81_; lean_object* v___x_82_; 
v_a_81_ = lean_ctor_get(v_r_80_, 0);
lean_inc(v_a_81_);
lean_dec_ref_known(v_r_80_, 1);
v___x_82_ = l_Lean_Meta_SavedState_restore___redArg(v_a_79_, v___y_74_, v___y_76_);
lean_dec(v_a_79_);
if (lean_obj_tag(v___x_82_) == 0)
{
lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_89_ == 0)
{
lean_object* v_unused_90_; 
v_unused_90_ = lean_ctor_get(v___x_82_, 0);
lean_dec(v_unused_90_);
v___x_84_ = v___x_82_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_dec(v___x_82_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 0, v_a_81_);
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_81_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
else
{
lean_object* v_a_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_98_; 
lean_dec(v_a_81_);
v_a_91_ = lean_ctor_get(v___x_82_, 0);
v_isSharedCheck_98_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_98_ == 0)
{
v___x_93_ = v___x_82_;
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_a_91_);
lean_dec(v___x_82_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___x_96_; 
if (v_isShared_94_ == 0)
{
v___x_96_ = v___x_93_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v_a_91_);
v___x_96_ = v_reuseFailAlloc_97_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
return v___x_96_;
}
}
}
}
else
{
lean_object* v_a_99_; lean_object* v___x_100_; 
v_a_99_ = lean_ctor_get(v_r_80_, 0);
lean_inc(v_a_99_);
lean_dec_ref_known(v_r_80_, 1);
v___x_100_ = l_Lean_Meta_SavedState_restore___redArg(v_a_79_, v___y_74_, v___y_76_);
lean_dec(v_a_79_);
if (lean_obj_tag(v___x_100_) == 0)
{
lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_107_; 
v_isSharedCheck_107_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_107_ == 0)
{
lean_object* v_unused_108_; 
v_unused_108_ = lean_ctor_get(v___x_100_, 0);
lean_dec(v_unused_108_);
v___x_102_ = v___x_100_;
v_isShared_103_ = v_isSharedCheck_107_;
goto v_resetjp_101_;
}
else
{
lean_dec(v___x_100_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_107_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___x_105_; 
if (v_isShared_103_ == 0)
{
lean_ctor_set_tag(v___x_102_, 1);
lean_ctor_set(v___x_102_, 0, v_a_99_);
v___x_105_ = v___x_102_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_a_99_);
v___x_105_ = v_reuseFailAlloc_106_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
return v___x_105_;
}
}
}
else
{
lean_object* v_a_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_116_; 
lean_dec(v_a_99_);
v_a_109_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_116_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_116_ == 0)
{
v___x_111_ = v___x_100_;
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_a_109_);
lean_dec(v___x_100_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___x_114_; 
if (v_isShared_112_ == 0)
{
v___x_114_ = v___x_111_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v_a_109_);
v___x_114_ = v_reuseFailAlloc_115_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
return v___x_114_;
}
}
}
}
}
else
{
lean_object* v_a_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_124_; 
lean_dec_ref(v_x_72_);
v_a_117_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_124_ == 0)
{
v___x_119_ = v___x_78_;
v_isShared_120_ = v_isSharedCheck_124_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_a_117_);
lean_dec(v___x_78_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_124_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_122_; 
if (v_isShared_120_ == 0)
{
v___x_122_ = v___x_119_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v_a_117_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg___boxed(lean_object* v_x_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(v_x_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1(lean_object* v_00_u03b1_132_, lean_object* v_x_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(v_x_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___boxed(lean_object* v_00_u03b1_140_, lean_object* v_x_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1(v_00_u03b1_140_, v_x_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0(lean_object* v_a_148_, size_t v_sz_149_, size_t v_i_150_, lean_object* v_bs_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
uint8_t v___x_157_; 
v___x_157_ = lean_usize_dec_lt(v_i_150_, v_sz_149_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; 
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v_bs_151_);
return v___x_158_;
}
else
{
lean_object* v___x_159_; 
v___x_159_ = l_Lean_Meta_SavedState_restore___redArg(v_a_148_, v___y_153_, v___y_155_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_v_160_; lean_object* v___x_161_; 
lean_dec_ref_known(v___x_159_, 1);
v_v_160_ = lean_array_uget_borrowed(v_bs_151_, v_i_150_);
lean_inc(v_v_160_);
v___x_161_ = lp_aesop_Aesop_CasesPattern_toExpr(v_v_160_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_object* v_a_162_; lean_object* v___x_163_; 
v_a_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_a_162_);
lean_dec_ref_known(v___x_161_, 1);
v___x_163_ = l_Lean_Meta_saveState___redArg(v___y_153_, v___y_155_);
if (lean_obj_tag(v___x_163_) == 0)
{
lean_object* v_a_164_; lean_object* v___x_165_; lean_object* v_bs_x27_166_; lean_object* v___x_167_; size_t v___x_168_; size_t v___x_169_; lean_object* v___x_170_; 
v_a_164_ = lean_ctor_get(v___x_163_, 0);
lean_inc(v_a_164_);
lean_dec_ref_known(v___x_163_, 1);
v___x_165_ = lean_unsigned_to_nat(0u);
v_bs_x27_166_ = lean_array_uset(v_bs_151_, v_i_150_, v___x_165_);
v___x_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_167_, 0, v_a_162_);
lean_ctor_set(v___x_167_, 1, v_a_164_);
v___x_168_ = ((size_t)1ULL);
v___x_169_ = lean_usize_add(v_i_150_, v___x_168_);
v___x_170_ = lean_array_uset(v_bs_x27_166_, v_i_150_, v___x_167_);
v_i_150_ = v___x_169_;
v_bs_151_ = v___x_170_;
goto _start;
}
else
{
lean_object* v_a_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
lean_dec(v_a_162_);
lean_dec_ref(v_bs_151_);
v_a_172_ = lean_ctor_get(v___x_163_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_163_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_163_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_a_172_);
lean_dec(v___x_163_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_a_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
else
{
lean_object* v_a_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_187_; 
lean_dec_ref(v_bs_151_);
v_a_180_ = lean_ctor_get(v___x_161_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_187_ == 0)
{
v___x_182_ = v___x_161_;
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_a_180_);
lean_dec(v___x_161_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_185_; 
if (v_isShared_183_ == 0)
{
v___x_185_ = v___x_182_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_a_180_);
v___x_185_ = v_reuseFailAlloc_186_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
return v___x_185_;
}
}
}
}
else
{
lean_object* v_a_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_195_; 
lean_dec_ref(v_bs_151_);
v_a_188_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_195_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_195_ == 0)
{
v___x_190_ = v___x_159_;
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_a_188_);
lean_dec(v___x_159_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
lean_object* v___x_193_; 
if (v_isShared_191_ == 0)
{
v___x_193_ = v___x_190_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v_a_188_);
v___x_193_ = v_reuseFailAlloc_194_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
return v___x_193_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0___boxed(lean_object* v_a_196_, lean_object* v_sz_197_, lean_object* v_i_198_, lean_object* v_bs_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
size_t v_sz_boxed_205_; size_t v_i_boxed_206_; lean_object* v_res_207_; 
v_sz_boxed_205_ = lean_unbox_usize(v_sz_197_);
lean_dec(v_sz_197_);
v_i_boxed_206_ = lean_unbox_usize(v_i_198_);
lean_dec(v_i_198_);
v_res_207_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0(v_a_196_, v_sz_boxed_205_, v_i_boxed_206_, v_bs_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
lean_dec_ref(v_a_196_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0(lean_object* v_patterns_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = l_Lean_Meta_saveState___redArg(v___y_210_, v___y_212_);
if (lean_obj_tag(v___x_214_) == 0)
{
lean_object* v_a_215_; size_t v_sz_216_; size_t v___x_217_; lean_object* v___x_218_; 
v_a_215_ = lean_ctor_get(v___x_214_, 0);
lean_inc(v_a_215_);
lean_dec_ref_known(v___x_214_, 1);
v_sz_216_ = lean_array_size(v_patterns_208_);
v___x_217_ = ((size_t)0ULL);
v___x_218_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__0(v_a_215_, v_sz_216_, v___x_217_, v_patterns_208_, v___y_209_, v___y_210_, v___y_211_, v___y_212_);
lean_dec(v_a_215_);
if (lean_obj_tag(v___x_218_) == 0)
{
lean_object* v_a_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_227_; 
v_a_219_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_227_ == 0)
{
v___x_221_ = v___x_218_;
v_isShared_222_ = v_isSharedCheck_227_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_a_219_);
lean_dec(v___x_218_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_227_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_223_; lean_object* v___x_225_; 
v___x_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_223_, 0, v_a_219_);
if (v_isShared_222_ == 0)
{
lean_ctor_set(v___x_221_, 0, v___x_223_);
v___x_225_ = v___x_221_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v___x_223_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
else
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
v_a_228_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_218_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_218_);
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
else
{
lean_object* v_a_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_243_; 
lean_dec_ref(v_patterns_208_);
v_a_236_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_243_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_243_ == 0)
{
v___x_238_ = v___x_214_;
v_isShared_239_ = v_isSharedCheck_243_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_a_236_);
lean_dec(v___x_214_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_243_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_241_; 
if (v_isShared_239_ == 0)
{
v___x_241_ = v___x_238_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v_a_236_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0___boxed(lean_object* v_patterns_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0(v_patterns_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_);
lean_dec(v___y_248_);
lean_dec_ref(v___y_247_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27(lean_object* v_x_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_){
_start:
{
if (lean_obj_tag(v_x_251_) == 0)
{
lean_object* v_decl_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_265_; 
v_decl_257_ = lean_ctor_get(v_x_251_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v_x_251_);
if (v_isSharedCheck_265_ == 0)
{
v___x_259_ = v_x_251_;
v_isShared_260_ = v_isSharedCheck_265_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_decl_257_);
lean_dec(v_x_251_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_265_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v_decl_257_);
v___x_262_ = v_reuseFailAlloc_264_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_263_; 
v___x_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
return v___x_263_;
}
}
}
else
{
lean_object* v_patterns_266_; lean_object* v___f_267_; lean_object* v___x_268_; 
v_patterns_266_ = lean_ctor_get(v_x_251_, 0);
lean_inc_ref(v_patterns_266_);
lean_dec_ref_known(v_x_251_, 1);
v___f_267_ = lean_alloc_closure((void*)(lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___lam__0___boxed), 6, 1);
lean_closure_set(v___f_267_, 0, v_patterns_266_);
v___x_268_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(v___f_267_, v_a_252_, v_a_253_, v_a_254_, v_a_255_);
return v___x_268_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesTarget_toCasesTarget_x27___boxed(lean_object* v_x_269_, lean_object* v_a_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_aesop_Aesop_CasesTarget_toCasesTarget_x27(v_x_269_, v_a_270_, v_a_271_, v_a_272_, v_a_273_);
lean_dec(v_a_273_);
lean_dec_ref(v_a_272_);
lean_dec(v_a_271_);
lean_dec_ref(v_a_270_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg(lean_object* v_mvarId_276_, lean_object* v_x_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_276_, v_x_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
v_a_284_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_283_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_283_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
else
{
lean_object* v_a_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_299_; 
v_a_292_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_299_ == 0)
{
v___x_294_ = v___x_283_;
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_a_292_);
lean_dec(v___x_283_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_297_; 
if (v_isShared_295_ == 0)
{
v___x_297_ = v___x_294_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_a_292_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg___boxed(lean_object* v_mvarId_300_, lean_object* v_x_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg(v_mvarId_300_, v_x_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3(lean_object* v_00_u03b1_308_, lean_object* v_mvarId_309_, lean_object* v_x_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg(v_mvarId_309_, v_x_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___boxed(lean_object* v_00_u03b1_317_, lean_object* v_mvarId_318_, lean_object* v_x_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3(v_00_u03b1_317_, v_mvarId_318_, v_x_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
return v_res_325_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1(lean_object* v_a_326_, lean_object* v_as_327_, size_t v_i_328_, size_t v_stop_329_){
_start:
{
uint8_t v___x_330_; 
v___x_330_ = lean_usize_dec_eq(v_i_328_, v_stop_329_);
if (v___x_330_ == 0)
{
lean_object* v___x_331_; uint8_t v___x_332_; 
v___x_331_ = lean_array_uget_borrowed(v_as_327_, v_i_328_);
v___x_332_ = l_Lean_instBEqFVarId_beq(v_a_326_, v___x_331_);
if (v___x_332_ == 0)
{
size_t v___x_333_; size_t v___x_334_; 
v___x_333_ = ((size_t)1ULL);
v___x_334_ = lean_usize_add(v_i_328_, v___x_333_);
v_i_328_ = v___x_334_;
goto _start;
}
else
{
return v___x_332_;
}
}
else
{
uint8_t v___x_336_; 
v___x_336_ = 0;
return v___x_336_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1___boxed(lean_object* v_a_337_, lean_object* v_as_338_, lean_object* v_i_339_, lean_object* v_stop_340_){
_start:
{
size_t v_i_boxed_341_; size_t v_stop_boxed_342_; uint8_t v_res_343_; lean_object* v_r_344_; 
v_i_boxed_341_ = lean_unbox_usize(v_i_339_);
lean_dec(v_i_339_);
v_stop_boxed_342_ = lean_unbox_usize(v_stop_340_);
lean_dec(v_stop_340_);
v_res_343_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1(v_a_337_, v_as_338_, v_i_boxed_341_, v_stop_boxed_342_);
lean_dec_ref(v_as_338_);
lean_dec(v_a_337_);
v_r_344_ = lean_box(v_res_343_);
return v_r_344_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1(lean_object* v_as_345_, lean_object* v_a_346_){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; uint8_t v___x_349_; 
v___x_347_ = lean_unsigned_to_nat(0u);
v___x_348_ = lean_array_get_size(v_as_345_);
v___x_349_ = lean_nat_dec_lt(v___x_347_, v___x_348_);
if (v___x_349_ == 0)
{
return v___x_349_;
}
else
{
if (v___x_349_ == 0)
{
return v___x_349_;
}
else
{
size_t v___x_350_; size_t v___x_351_; uint8_t v___x_352_; 
v___x_350_ = ((size_t)0ULL);
v___x_351_ = lean_usize_of_nat(v___x_348_);
v___x_352_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1_spec__1(v_a_346_, v_as_345_, v___x_350_, v___x_351_);
return v___x_352_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1___boxed(lean_object* v_as_353_, lean_object* v_a_354_){
_start:
{
uint8_t v_res_355_; lean_object* v_r_356_; 
v_res_355_ = lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1(v_as_353_, v_a_354_);
lean_dec(v_a_354_);
lean_dec_ref(v_as_353_);
v_r_356_ = lean_box(v_res_355_);
return v_r_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0(lean_object* v_ldecl_357_, lean_object* v_as_358_, size_t v_i_359_, size_t v_stop_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_){
_start:
{
uint8_t v___x_366_; 
v___x_366_ = lean_usize_dec_eq(v_i_359_, v_stop_360_);
if (v___x_366_ == 0)
{
lean_object* v___x_367_; lean_object* v_fst_368_; lean_object* v_snd_369_; lean_object* v___x_370_; 
v___x_367_ = lean_array_uget_borrowed(v_as_358_, v_i_359_);
v_fst_368_ = lean_ctor_get(v___x_367_, 0);
v_snd_369_ = lean_ctor_get(v___x_367_, 1);
v___x_370_ = l_Lean_Meta_SavedState_restore___redArg(v_snd_369_, v___y_362_, v___y_364_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v___x_371_; lean_object* v___x_372_; 
lean_dec_ref_known(v___x_370_, 1);
v___x_371_ = l_Lean_LocalDecl_type(v_ldecl_357_);
lean_inc(v_fst_368_);
v___x_372_ = l_Lean_Meta_isExprDefEq(v_fst_368_, v___x_371_, v___y_361_, v___y_362_, v___y_363_, v___y_364_);
if (lean_obj_tag(v___x_372_) == 0)
{
lean_object* v_a_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_384_; 
v_a_373_ = lean_ctor_get(v___x_372_, 0);
v_isSharedCheck_384_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_384_ == 0)
{
v___x_375_ = v___x_372_;
v_isShared_376_ = v_isSharedCheck_384_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_a_373_);
lean_dec(v___x_372_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_384_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
uint8_t v___x_377_; 
v___x_377_ = lean_unbox(v_a_373_);
if (v___x_377_ == 0)
{
size_t v___x_378_; size_t v___x_379_; 
lean_del_object(v___x_375_);
lean_dec(v_a_373_);
v___x_378_ = ((size_t)1ULL);
v___x_379_ = lean_usize_add(v_i_359_, v___x_378_);
v_i_359_ = v___x_379_;
goto _start;
}
else
{
lean_object* v___x_382_; 
if (v_isShared_376_ == 0)
{
v___x_382_ = v___x_375_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_383_; 
v_reuseFailAlloc_383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_383_, 0, v_a_373_);
v___x_382_ = v_reuseFailAlloc_383_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
return v___x_382_;
}
}
}
}
else
{
return v___x_372_;
}
}
else
{
lean_object* v_a_385_; lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_392_; 
v_a_385_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_392_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_392_ == 0)
{
v___x_387_ = v___x_370_;
v_isShared_388_ = v_isSharedCheck_392_;
goto v_resetjp_386_;
}
else
{
lean_inc(v_a_385_);
lean_dec(v___x_370_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_392_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
lean_object* v___x_390_; 
if (v_isShared_388_ == 0)
{
v___x_390_ = v___x_387_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_a_385_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
}
}
else
{
uint8_t v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_393_ = 0;
v___x_394_ = lean_box(v___x_393_);
v___x_395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
return v___x_395_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0___boxed(lean_object* v_ldecl_396_, lean_object* v_as_397_, lean_object* v_i_398_, lean_object* v_stop_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
size_t v_i_boxed_405_; size_t v_stop_boxed_406_; lean_object* v_res_407_; 
v_i_boxed_405_ = lean_unbox_usize(v_i_398_);
lean_dec(v_i_398_);
v_stop_boxed_406_ = lean_unbox_usize(v_stop_399_);
lean_dec(v_stop_399_);
v_res_407_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0(v_ldecl_396_, v_as_397_, v_i_boxed_405_, v_stop_boxed_406_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec_ref(v_as_397_);
lean_dec_ref(v_ldecl_396_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0(lean_object* v___x_408_, lean_object* v___x_409_, uint8_t v___y_410_, lean_object* v_val_411_, lean_object* v_ps_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
uint8_t v___x_418_; 
v___x_418_ = lean_nat_dec_lt(v___x_408_, v___x_409_);
if (v___x_418_ == 0)
{
lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_419_ = lean_box(v___y_410_);
v___x_420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
return v___x_420_;
}
else
{
if (v___x_418_ == 0)
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = lean_box(v___y_410_);
v___x_422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
return v___x_422_;
}
else
{
size_t v___x_423_; size_t v___x_424_; lean_object* v___x_425_; 
v___x_423_ = ((size_t)0ULL);
v___x_424_ = lean_usize_of_nat(v___x_409_);
v___x_425_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__0(v_val_411_, v_ps_412_, v___x_423_, v___x_424_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
return v___x_425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0___boxed(lean_object* v___x_426_, lean_object* v___x_427_, lean_object* v___y_428_, lean_object* v_val_429_, lean_object* v_ps_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_){
_start:
{
uint8_t v___y_5487__boxed_436_; lean_object* v_res_437_; 
v___y_5487__boxed_436_ = lean_unbox(v___y_428_);
v_res_437_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0(v___x_426_, v___x_427_, v___y_5487__boxed_436_, v_val_429_, v_ps_430_, v___y_431_, v___y_432_, v___y_433_, v___y_434_);
lean_dec(v___y_434_);
lean_dec_ref(v___y_433_);
lean_dec(v___y_432_);
lean_dec_ref(v___y_431_);
lean_dec_ref(v_ps_430_);
lean_dec_ref(v_val_429_);
lean_dec(v___x_427_);
lean_dec(v___x_426_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8(lean_object* v_a_441_, lean_object* v_excluded_442_, lean_object* v_as_443_, size_t v_sz_444_, size_t v_i_445_, lean_object* v_b_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_a_453_; uint8_t v___x_457_; 
v___x_457_ = lean_usize_dec_lt(v_i_445_, v_sz_444_);
if (v___x_457_ == 0)
{
lean_object* v___x_458_; 
lean_dec_ref(v_a_441_);
v___x_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_458_, 0, v_b_446_);
return v___x_458_;
}
else
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v_a_461_; 
lean_dec_ref(v_b_446_);
v___x_459_ = lean_box(0);
v___x_460_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v_a_461_ = lean_array_uget(v_as_443_, v_i_445_);
if (lean_obj_tag(v_a_461_) == 0)
{
v_a_453_ = v___x_460_;
goto v___jp_452_;
}
else
{
lean_object* v_val_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_515_; 
v_val_462_ = lean_ctor_get(v_a_461_, 0);
v_isSharedCheck_515_ = !lean_is_exclusive(v_a_461_);
if (v_isSharedCheck_515_ == 0)
{
v___x_464_ = v_a_461_;
v_isShared_465_ = v_isSharedCheck_515_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_val_462_);
lean_dec(v_a_461_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_515_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___y_467_; uint8_t v___y_492_; uint8_t v___x_512_; 
v___x_512_ = l_Lean_LocalDecl_isImplementationDetail(v_val_462_);
if (v___x_512_ == 0)
{
lean_object* v___x_513_; uint8_t v___x_514_; 
v___x_513_ = l_Lean_LocalDecl_fvarId(v_val_462_);
v___x_514_ = lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1(v_excluded_442_, v___x_513_);
lean_dec(v___x_513_);
v___y_492_ = v___x_514_;
goto v___jp_491_;
}
else
{
v___y_492_ = v___x_512_;
goto v___jp_491_;
}
v___jp_466_:
{
if (lean_obj_tag(v___y_467_) == 0)
{
lean_object* v_a_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_482_; 
v_a_468_ = lean_ctor_get(v___y_467_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___y_467_);
if (v_isSharedCheck_482_ == 0)
{
v___x_470_ = v___y_467_;
v_isShared_471_ = v_isSharedCheck_482_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_a_468_);
lean_dec(v___y_467_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_482_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
uint8_t v___x_472_; 
v___x_472_ = lean_unbox(v_a_468_);
lean_dec(v_a_468_);
if (v___x_472_ == 0)
{
lean_del_object(v___x_470_);
lean_del_object(v___x_464_);
lean_dec(v_val_462_);
v_a_453_ = v___x_460_;
goto v___jp_452_;
}
else
{
lean_object* v___x_473_; lean_object* v___x_475_; 
lean_dec_ref(v_a_441_);
v___x_473_ = l_Lean_LocalDecl_fvarId(v_val_462_);
lean_dec(v_val_462_);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 0, v___x_473_);
v___x_475_ = v___x_464_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v___x_473_);
v___x_475_ = v_reuseFailAlloc_481_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_479_; 
v___x_476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_476_, 0, v___x_475_);
v___x_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
lean_ctor_set(v___x_477_, 1, v___x_459_);
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 0, v___x_477_);
v___x_479_ = v___x_470_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v___x_477_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
else
{
lean_object* v_a_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_490_; 
lean_del_object(v___x_464_);
lean_dec(v_val_462_);
lean_dec_ref(v_a_441_);
v_a_483_ = lean_ctor_get(v___y_467_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___y_467_);
if (v_isSharedCheck_490_ == 0)
{
v___x_485_ = v___y_467_;
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_a_483_);
lean_dec(v___y_467_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_488_; 
if (v_isShared_486_ == 0)
{
v___x_488_ = v___x_485_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v_a_483_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
v___jp_491_:
{
if (v___y_492_ == 0)
{
if (lean_obj_tag(v_a_441_) == 0)
{
lean_object* v_decl_493_; lean_object* v___x_494_; 
v_decl_493_ = lean_ctor_get(v_a_441_, 0);
lean_inc(v_decl_493_);
v___x_494_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_493_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc(v_a_495_);
lean_dec_ref_known(v___x_494_, 1);
v___x_496_ = l_Lean_LocalDecl_type(v_val_462_);
v___x_497_ = lp_aesop_Aesop_isAppOfUpToDefeq(v_a_495_, v___x_496_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
v___y_467_ = v___x_497_;
goto v___jp_466_;
}
else
{
lean_object* v_a_498_; lean_object* v___x_500_; uint8_t v_isShared_501_; uint8_t v_isSharedCheck_505_; 
lean_dec_ref_known(v_a_441_, 1);
lean_del_object(v___x_464_);
lean_dec(v_val_462_);
v_a_498_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_505_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_505_ == 0)
{
v___x_500_ = v___x_494_;
v_isShared_501_ = v_isSharedCheck_505_;
goto v_resetjp_499_;
}
else
{
lean_inc(v_a_498_);
lean_dec(v___x_494_);
v___x_500_ = lean_box(0);
v_isShared_501_ = v_isSharedCheck_505_;
goto v_resetjp_499_;
}
v_resetjp_499_:
{
lean_object* v___x_503_; 
if (v_isShared_501_ == 0)
{
v___x_503_ = v___x_500_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v_a_498_);
v___x_503_ = v_reuseFailAlloc_504_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
return v___x_503_;
}
}
}
}
else
{
lean_object* v_ps_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___f_510_; lean_object* v___x_511_; 
v_ps_506_ = lean_ctor_get(v_a_441_, 0);
v___x_507_ = lean_unsigned_to_nat(0u);
v___x_508_ = lean_array_get_size(v_ps_506_);
v___x_509_ = lean_box(v___y_492_);
lean_inc_ref(v_ps_506_);
lean_inc(v_val_462_);
v___f_510_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0___boxed), 10, 5);
lean_closure_set(v___f_510_, 0, v___x_507_);
lean_closure_set(v___f_510_, 1, v___x_508_);
lean_closure_set(v___f_510_, 2, v___x_509_);
lean_closure_set(v___f_510_, 3, v_val_462_);
lean_closure_set(v___f_510_, 4, v_ps_506_);
v___x_511_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(v___f_510_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
v___y_467_ = v___x_511_;
goto v___jp_466_;
}
}
else
{
lean_del_object(v___x_464_);
lean_dec(v_val_462_);
v_a_453_ = v___x_460_;
goto v___jp_452_;
}
}
}
}
}
v___jp_452_:
{
size_t v___x_454_; size_t v___x_455_; 
v___x_454_ = ((size_t)1ULL);
v___x_455_ = lean_usize_add(v_i_445_, v___x_454_);
lean_inc_ref(v_a_453_);
v_i_445_ = v___x_455_;
v_b_446_ = v_a_453_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___boxed(lean_object* v_a_516_, lean_object* v_excluded_517_, lean_object* v_as_518_, lean_object* v_sz_519_, lean_object* v_i_520_, lean_object* v_b_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_){
_start:
{
size_t v_sz_boxed_527_; size_t v_i_boxed_528_; lean_object* v_res_529_; 
v_sz_boxed_527_ = lean_unbox_usize(v_sz_519_);
lean_dec(v_sz_519_);
v_i_boxed_528_ = lean_unbox_usize(v_i_520_);
lean_dec(v_i_520_);
v_res_529_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8(v_a_516_, v_excluded_517_, v_as_518_, v_sz_boxed_527_, v_i_boxed_528_, v_b_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec_ref(v_as_518_);
lean_dec_ref(v_excluded_517_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6(lean_object* v_a_530_, lean_object* v_excluded_531_, lean_object* v_as_532_, size_t v_sz_533_, size_t v_i_534_, lean_object* v_b_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_){
_start:
{
lean_object* v_a_542_; uint8_t v___x_546_; 
v___x_546_ = lean_usize_dec_lt(v_i_534_, v_sz_533_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; 
lean_dec_ref(v_a_530_);
v___x_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_547_, 0, v_b_535_);
return v___x_547_;
}
else
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v_a_550_; 
lean_dec_ref(v_b_535_);
v___x_548_ = lean_box(0);
v___x_549_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v_a_550_ = lean_array_uget(v_as_532_, v_i_534_);
if (lean_obj_tag(v_a_550_) == 0)
{
v_a_542_ = v___x_549_;
goto v___jp_541_;
}
else
{
lean_object* v_val_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_604_; 
v_val_551_ = lean_ctor_get(v_a_550_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v_a_550_);
if (v_isSharedCheck_604_ == 0)
{
v___x_553_ = v_a_550_;
v_isShared_554_ = v_isSharedCheck_604_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_val_551_);
lean_dec(v_a_550_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_604_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___y_556_; uint8_t v___y_581_; uint8_t v___x_601_; 
v___x_601_ = l_Lean_LocalDecl_isImplementationDetail(v_val_551_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; uint8_t v___x_603_; 
v___x_602_ = l_Lean_LocalDecl_fvarId(v_val_551_);
v___x_603_ = lp_aesop_Array_contains___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__1(v_excluded_531_, v___x_602_);
lean_dec(v___x_602_);
v___y_581_ = v___x_603_;
goto v___jp_580_;
}
else
{
v___y_581_ = v___x_601_;
goto v___jp_580_;
}
v___jp_555_:
{
if (lean_obj_tag(v___y_556_) == 0)
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_571_; 
v_a_557_ = lean_ctor_get(v___y_556_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___y_556_);
if (v_isSharedCheck_571_ == 0)
{
v___x_559_ = v___y_556_;
v_isShared_560_ = v_isSharedCheck_571_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___y_556_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_571_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
uint8_t v___x_561_; 
v___x_561_ = lean_unbox(v_a_557_);
lean_dec(v_a_557_);
if (v___x_561_ == 0)
{
lean_del_object(v___x_559_);
lean_del_object(v___x_553_);
lean_dec(v_val_551_);
v_a_542_ = v___x_549_;
goto v___jp_541_;
}
else
{
lean_object* v___x_562_; lean_object* v___x_564_; 
lean_dec_ref(v_a_530_);
v___x_562_ = l_Lean_LocalDecl_fvarId(v_val_551_);
lean_dec(v_val_551_);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 0, v___x_562_);
v___x_564_ = v___x_553_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_562_);
v___x_564_ = v_reuseFailAlloc_570_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_568_; 
v___x_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_565_, 0, v___x_564_);
v___x_566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
lean_ctor_set(v___x_566_, 1, v___x_548_);
if (v_isShared_560_ == 0)
{
lean_ctor_set(v___x_559_, 0, v___x_566_);
v___x_568_ = v___x_559_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v___x_566_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
}
else
{
lean_object* v_a_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_579_; 
lean_del_object(v___x_553_);
lean_dec(v_val_551_);
lean_dec_ref(v_a_530_);
v_a_572_ = lean_ctor_get(v___y_556_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v___y_556_);
if (v_isSharedCheck_579_ == 0)
{
v___x_574_ = v___y_556_;
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_a_572_);
lean_dec(v___y_556_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_577_; 
if (v_isShared_575_ == 0)
{
v___x_577_ = v___x_574_;
goto v_reusejp_576_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_a_572_);
v___x_577_ = v_reuseFailAlloc_578_;
goto v_reusejp_576_;
}
v_reusejp_576_:
{
return v___x_577_;
}
}
}
}
v___jp_580_:
{
if (v___y_581_ == 0)
{
if (lean_obj_tag(v_a_530_) == 0)
{
lean_object* v_decl_582_; lean_object* v___x_583_; 
v_decl_582_ = lean_ctor_get(v_a_530_, 0);
lean_inc(v_decl_582_);
v___x_583_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_582_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v_a_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref_known(v___x_583_, 1);
v___x_585_ = l_Lean_LocalDecl_type(v_val_551_);
v___x_586_ = lp_aesop_Aesop_isAppOfUpToDefeq(v_a_584_, v___x_585_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
v___y_556_ = v___x_586_;
goto v___jp_555_;
}
else
{
lean_object* v_a_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_594_; 
lean_dec_ref_known(v_a_530_, 1);
lean_del_object(v___x_553_);
lean_dec(v_val_551_);
v_a_587_ = lean_ctor_get(v___x_583_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_594_ == 0)
{
v___x_589_ = v___x_583_;
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_a_587_);
lean_dec(v___x_583_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_592_; 
if (v_isShared_590_ == 0)
{
v___x_592_ = v___x_589_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v_a_587_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
}
else
{
lean_object* v_ps_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___f_599_; lean_object* v___x_600_; 
v_ps_595_ = lean_ctor_get(v_a_530_, 0);
v___x_596_ = lean_unsigned_to_nat(0u);
v___x_597_ = lean_array_get_size(v_ps_595_);
v___x_598_ = lean_box(v___y_581_);
lean_inc_ref(v_ps_595_);
lean_inc(v_val_551_);
v___f_599_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___lam__0___boxed), 10, 5);
lean_closure_set(v___f_599_, 0, v___x_596_);
lean_closure_set(v___f_599_, 1, v___x_597_);
lean_closure_set(v___f_599_, 2, v___x_598_);
lean_closure_set(v___f_599_, 3, v_val_551_);
lean_closure_set(v___f_599_, 4, v_ps_595_);
v___x_600_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesTarget_toCasesTarget_x27_spec__1___redArg(v___f_599_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
v___y_556_ = v___x_600_;
goto v___jp_555_;
}
}
else
{
lean_del_object(v___x_553_);
lean_dec(v_val_551_);
v_a_542_ = v___x_549_;
goto v___jp_541_;
}
}
}
}
}
v___jp_541_:
{
size_t v___x_543_; size_t v___x_544_; lean_object* v___x_545_; 
v___x_543_ = ((size_t)1ULL);
v___x_544_ = lean_usize_add(v_i_534_, v___x_543_);
lean_inc_ref(v_a_542_);
v___x_545_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8(v_a_530_, v_excluded_531_, v_as_532_, v_sz_533_, v___x_544_, v_a_542_, v___y_536_, v___y_537_, v___y_538_, v___y_539_);
return v___x_545_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6___boxed(lean_object* v_a_605_, lean_object* v_excluded_606_, lean_object* v_as_607_, lean_object* v_sz_608_, lean_object* v_i_609_, lean_object* v_b_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
size_t v_sz_boxed_616_; size_t v_i_boxed_617_; lean_object* v_res_618_; 
v_sz_boxed_616_ = lean_unbox_usize(v_sz_608_);
lean_dec(v_sz_608_);
v_i_boxed_617_ = lean_unbox_usize(v_i_609_);
lean_dec(v_i_609_);
v_res_618_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6(v_a_605_, v_excluded_606_, v_as_607_, v_sz_boxed_616_, v_i_boxed_617_, v_b_610_, v___y_611_, v___y_612_, v___y_613_, v___y_614_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
lean_dec(v___y_612_);
lean_dec_ref(v___y_611_);
lean_dec_ref(v_as_607_);
lean_dec_ref(v_excluded_606_);
return v_res_618_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5(lean_object* v_a_619_, lean_object* v_excluded_620_, lean_object* v_x_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
if (lean_obj_tag(v_x_621_) == 0)
{
lean_object* v_cs_627_; lean_object* v___x_628_; lean_object* v___x_629_; size_t v_sz_630_; size_t v___x_631_; lean_object* v___x_632_; 
v_cs_627_ = lean_ctor_get(v_x_621_, 0);
v___x_628_ = lean_box(0);
v___x_629_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v_sz_630_ = lean_array_size(v_cs_627_);
v___x_631_ = ((size_t)0ULL);
v___x_632_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6(v_a_619_, v_excluded_620_, v_cs_627_, v_sz_630_, v___x_631_, v___x_629_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
if (lean_obj_tag(v___x_632_) == 0)
{
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_645_; 
v_a_633_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_645_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_645_ == 0)
{
v___x_635_ = v___x_632_;
v_isShared_636_ = v_isSharedCheck_645_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_632_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_645_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v_fst_637_; 
v_fst_637_ = lean_ctor_get(v_a_633_, 0);
lean_inc(v_fst_637_);
lean_dec(v_a_633_);
if (lean_obj_tag(v_fst_637_) == 0)
{
lean_object* v___x_639_; 
if (v_isShared_636_ == 0)
{
lean_ctor_set(v___x_635_, 0, v___x_628_);
v___x_639_ = v___x_635_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_640_; 
v_reuseFailAlloc_640_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_640_, 0, v___x_628_);
v___x_639_ = v_reuseFailAlloc_640_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
return v___x_639_;
}
}
else
{
lean_object* v_val_641_; lean_object* v___x_643_; 
v_val_641_ = lean_ctor_get(v_fst_637_, 0);
lean_inc(v_val_641_);
lean_dec_ref_known(v_fst_637_, 1);
if (v_isShared_636_ == 0)
{
lean_ctor_set(v___x_635_, 0, v_val_641_);
v___x_643_ = v___x_635_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v_val_641_);
v___x_643_ = v_reuseFailAlloc_644_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
return v___x_643_;
}
}
}
}
else
{
lean_object* v_a_646_; lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_653_; 
v_a_646_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_653_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_653_ == 0)
{
v___x_648_ = v___x_632_;
v_isShared_649_ = v_isSharedCheck_653_;
goto v_resetjp_647_;
}
else
{
lean_inc(v_a_646_);
lean_dec(v___x_632_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_653_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v___x_651_; 
if (v_isShared_649_ == 0)
{
v___x_651_ = v___x_648_;
goto v_reusejp_650_;
}
else
{
lean_object* v_reuseFailAlloc_652_; 
v_reuseFailAlloc_652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_652_, 0, v_a_646_);
v___x_651_ = v_reuseFailAlloc_652_;
goto v_reusejp_650_;
}
v_reusejp_650_:
{
return v___x_651_;
}
}
}
}
else
{
lean_object* v_vs_654_; lean_object* v___x_655_; lean_object* v___x_656_; size_t v_sz_657_; size_t v___x_658_; lean_object* v___x_659_; 
v_vs_654_ = lean_ctor_get(v_x_621_, 0);
v___x_655_ = lean_box(0);
v___x_656_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v_sz_657_ = lean_array_size(v_vs_654_);
v___x_658_ = ((size_t)0ULL);
v___x_659_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6(v_a_619_, v_excluded_620_, v_vs_654_, v_sz_657_, v___x_658_, v___x_656_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
if (lean_obj_tag(v___x_659_) == 0)
{
lean_object* v_a_660_; lean_object* v___x_662_; uint8_t v_isShared_663_; uint8_t v_isSharedCheck_672_; 
v_a_660_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_672_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_672_ == 0)
{
v___x_662_ = v___x_659_;
v_isShared_663_ = v_isSharedCheck_672_;
goto v_resetjp_661_;
}
else
{
lean_inc(v_a_660_);
lean_dec(v___x_659_);
v___x_662_ = lean_box(0);
v_isShared_663_ = v_isSharedCheck_672_;
goto v_resetjp_661_;
}
v_resetjp_661_:
{
lean_object* v_fst_664_; 
v_fst_664_ = lean_ctor_get(v_a_660_, 0);
lean_inc(v_fst_664_);
lean_dec(v_a_660_);
if (lean_obj_tag(v_fst_664_) == 0)
{
lean_object* v___x_666_; 
if (v_isShared_663_ == 0)
{
lean_ctor_set(v___x_662_, 0, v___x_655_);
v___x_666_ = v___x_662_;
goto v_reusejp_665_;
}
else
{
lean_object* v_reuseFailAlloc_667_; 
v_reuseFailAlloc_667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_667_, 0, v___x_655_);
v___x_666_ = v_reuseFailAlloc_667_;
goto v_reusejp_665_;
}
v_reusejp_665_:
{
return v___x_666_;
}
}
else
{
lean_object* v_val_668_; lean_object* v___x_670_; 
v_val_668_ = lean_ctor_get(v_fst_664_, 0);
lean_inc(v_val_668_);
lean_dec_ref_known(v_fst_664_, 1);
if (v_isShared_663_ == 0)
{
lean_ctor_set(v___x_662_, 0, v_val_668_);
v___x_670_ = v___x_662_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v_val_668_);
v___x_670_ = v_reuseFailAlloc_671_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
return v___x_670_;
}
}
}
}
else
{
lean_object* v_a_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_680_; 
v_a_673_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_680_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_680_ == 0)
{
v___x_675_ = v___x_659_;
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_a_673_);
lean_dec(v___x_659_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_678_; 
if (v_isShared_676_ == 0)
{
v___x_678_ = v___x_675_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_679_; 
v_reuseFailAlloc_679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_679_, 0, v_a_673_);
v___x_678_ = v_reuseFailAlloc_679_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
return v___x_678_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6(lean_object* v_a_681_, lean_object* v_excluded_682_, lean_object* v_as_683_, size_t v_sz_684_, size_t v_i_685_, lean_object* v_b_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_){
_start:
{
uint8_t v___x_692_; 
v___x_692_ = lean_usize_dec_lt(v_i_685_, v_sz_684_);
if (v___x_692_ == 0)
{
lean_object* v___x_693_; 
lean_dec_ref(v_a_681_);
v___x_693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_693_, 0, v_b_686_);
return v___x_693_;
}
else
{
lean_object* v_a_694_; lean_object* v___x_695_; 
lean_dec_ref(v_b_686_);
v_a_694_ = lean_array_uget_borrowed(v_as_683_, v_i_685_);
lean_inc_ref(v_a_681_);
v___x_695_ = lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5(v_a_681_, v_excluded_682_, v_a_694_, v___y_687_, v___y_688_, v___y_689_, v___y_690_);
if (lean_obj_tag(v___x_695_) == 0)
{
lean_object* v_a_696_; lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_710_; 
v_a_696_ = lean_ctor_get(v___x_695_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_710_ == 0)
{
v___x_698_ = v___x_695_;
v_isShared_699_ = v_isSharedCheck_710_;
goto v_resetjp_697_;
}
else
{
lean_inc(v_a_696_);
lean_dec(v___x_695_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_710_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
lean_object* v___x_700_; 
v___x_700_ = lean_box(0);
if (lean_obj_tag(v_a_696_) == 1)
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_704_; 
lean_dec_ref(v_a_681_);
v___x_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_701_, 0, v_a_696_);
v___x_702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
lean_ctor_set(v___x_702_, 1, v___x_700_);
if (v_isShared_699_ == 0)
{
lean_ctor_set(v___x_698_, 0, v___x_702_);
v___x_704_ = v___x_698_;
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
else
{
lean_object* v___x_706_; size_t v___x_707_; size_t v___x_708_; 
lean_del_object(v___x_698_);
lean_dec(v_a_696_);
v___x_706_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v___x_707_ = ((size_t)1ULL);
v___x_708_ = lean_usize_add(v_i_685_, v___x_707_);
v_i_685_ = v___x_708_;
v_b_686_ = v___x_706_;
goto _start;
}
}
}
else
{
lean_object* v_a_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_718_; 
lean_dec_ref(v_a_681_);
v_a_711_ = lean_ctor_get(v___x_695_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_718_ == 0)
{
v___x_713_ = v___x_695_;
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_a_711_);
lean_dec(v___x_695_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_716_; 
if (v_isShared_714_ == 0)
{
v___x_716_ = v___x_713_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_711_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6___boxed(lean_object* v_a_719_, lean_object* v_excluded_720_, lean_object* v_as_721_, lean_object* v_sz_722_, lean_object* v_i_723_, lean_object* v_b_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
size_t v_sz_boxed_730_; size_t v_i_boxed_731_; lean_object* v_res_732_; 
v_sz_boxed_730_ = lean_unbox_usize(v_sz_722_);
lean_dec(v_sz_722_);
v_i_boxed_731_ = lean_unbox_usize(v_i_723_);
lean_dec(v_i_723_);
v_res_732_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5_spec__6(v_a_719_, v_excluded_720_, v_as_721_, v_sz_boxed_730_, v_i_boxed_731_, v_b_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec_ref(v_as_721_);
lean_dec_ref(v_excluded_720_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5___boxed(lean_object* v_a_733_, lean_object* v_excluded_734_, lean_object* v_x_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5(v_a_733_, v_excluded_734_, v_x_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
lean_dec_ref(v_x_735_);
lean_dec_ref(v_excluded_734_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3(lean_object* v_a_742_, lean_object* v_excluded_743_, lean_object* v_t_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_){
_start:
{
lean_object* v_root_750_; lean_object* v_tail_751_; lean_object* v___x_752_; 
v_root_750_ = lean_ctor_get(v_t_744_, 0);
v_tail_751_ = lean_ctor_get(v_t_744_, 1);
lean_inc_ref(v_a_742_);
v___x_752_ = lp_aesop_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__5(v_a_742_, v_excluded_743_, v_root_750_, v___y_745_, v___y_746_, v___y_747_, v___y_748_);
if (lean_obj_tag(v___x_752_) == 0)
{
lean_object* v_a_753_; 
v_a_753_ = lean_ctor_get(v___x_752_, 0);
lean_inc(v_a_753_);
if (lean_obj_tag(v_a_753_) == 0)
{
lean_object* v___x_754_; size_t v_sz_755_; size_t v___x_756_; lean_object* v___x_757_; 
lean_dec_ref_known(v___x_752_, 1);
v___x_754_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6_spec__8___closed__0));
v_sz_755_ = lean_array_size(v_tail_751_);
v___x_756_ = ((size_t)0ULL);
v___x_757_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3_spec__6(v_a_742_, v_excluded_743_, v_tail_751_, v_sz_755_, v___x_756_, v___x_754_, v___y_745_, v___y_746_, v___y_747_, v___y_748_);
if (lean_obj_tag(v___x_757_) == 0)
{
lean_object* v_a_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_770_; 
v_a_758_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_770_ == 0)
{
v___x_760_ = v___x_757_;
v_isShared_761_ = v_isSharedCheck_770_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_a_758_);
lean_dec(v___x_757_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_770_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v_fst_762_; 
v_fst_762_ = lean_ctor_get(v_a_758_, 0);
lean_inc(v_fst_762_);
lean_dec(v_a_758_);
if (lean_obj_tag(v_fst_762_) == 0)
{
lean_object* v___x_764_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 0, v_a_753_);
v___x_764_ = v___x_760_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_a_753_);
v___x_764_ = v_reuseFailAlloc_765_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
return v___x_764_;
}
}
else
{
lean_object* v_val_766_; lean_object* v___x_768_; 
v_val_766_ = lean_ctor_get(v_fst_762_, 0);
lean_inc(v_val_766_);
lean_dec_ref_known(v_fst_762_, 1);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 0, v_val_766_);
v___x_768_ = v___x_760_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_val_766_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
}
}
else
{
lean_object* v_a_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_778_; 
v_a_771_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_778_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_778_ == 0)
{
v___x_773_ = v___x_757_;
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_a_771_);
lean_dec(v___x_757_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_776_; 
if (v_isShared_774_ == 0)
{
v___x_776_ = v___x_773_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v_a_771_);
v___x_776_ = v_reuseFailAlloc_777_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
return v___x_776_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_753_, 1);
lean_dec_ref(v_a_742_);
return v___x_752_;
}
}
else
{
lean_dec_ref(v_a_742_);
return v___x_752_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3___boxed(lean_object* v_a_779_, lean_object* v_excluded_780_, lean_object* v_t_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3(v_a_779_, v_excluded_780_, v_t_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
lean_dec(v___y_783_);
lean_dec_ref(v___y_782_);
lean_dec_ref(v_t_781_);
lean_dec_ref(v_excluded_780_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2(lean_object* v_a_788_, lean_object* v_excluded_789_, lean_object* v_lctx_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_){
_start:
{
lean_object* v_decls_796_; lean_object* v___x_797_; 
v_decls_796_ = lean_ctor_get(v_lctx_790_, 1);
v___x_797_ = lp_aesop_Lean_PersistentArray_findSomeM_x3f___at___00Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2_spec__3(v_a_788_, v_excluded_789_, v_decls_796_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2___boxed(lean_object* v_a_798_, lean_object* v_excluded_799_, lean_object* v_lctx_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2(v_a_798_, v_excluded_799_, v_lctx_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_);
lean_dec(v___y_804_);
lean_dec_ref(v___y_803_);
lean_dec(v___y_802_);
lean_dec_ref(v___y_801_);
lean_dec_ref(v_lctx_800_);
lean_dec_ref(v_excluded_799_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0(lean_object* v_target_807_, lean_object* v_excluded_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_){
_start:
{
lean_object* v___x_814_; 
v___x_814_ = lp_aesop_Aesop_CasesTarget_toCasesTarget_x27(v_target_807_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
if (lean_obj_tag(v___x_814_) == 0)
{
lean_object* v_a_815_; lean_object* v_lctx_816_; lean_object* v___x_817_; 
v_a_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_a_815_);
lean_dec_ref_known(v___x_814_, 1);
v_lctx_816_ = lean_ctor_get(v___y_809_, 2);
v___x_817_ = lp_aesop_Lean_LocalContext_findDeclM_x3f___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__2(v_a_815_, v_excluded_808_, v_lctx_816_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
return v___x_817_;
}
else
{
lean_object* v_a_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_825_; 
v_a_818_ = lean_ctor_get(v___x_814_, 0);
v_isSharedCheck_825_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_825_ == 0)
{
v___x_820_ = v___x_814_;
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_a_818_);
lean_dec(v___x_814_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_825_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_823_; 
if (v_isShared_821_ == 0)
{
v___x_823_ = v___x_820_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v_a_818_);
v___x_823_ = v_reuseFailAlloc_824_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
return v___x_823_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0___boxed(lean_object* v_target_826_, lean_object* v_excluded_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_){
_start:
{
lean_object* v_res_833_; 
v_res_833_ = lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0(v_target_826_, v_excluded_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec(v___y_829_);
lean_dec_ref(v___y_828_);
lean_dec_ref(v_excluded_827_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp(lean_object* v_target_834_, uint8_t v_md_835_, lean_object* v_excluded_836_, lean_object* v_goal_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_){
_start:
{
lean_object* v_keyedConfig_843_; uint8_t v_trackZetaDelta_844_; lean_object* v_zetaDeltaSet_845_; lean_object* v_lctx_846_; lean_object* v_localInstances_847_; lean_object* v_defEqCtx_x3f_848_; lean_object* v_synthPendingDepth_849_; lean_object* v_customCanUnfoldPredicate_x3f_850_; uint8_t v_univApprox_851_; uint8_t v_inTypeClassResolution_852_; uint8_t v_cacheInferType_853_; lean_object* v___f_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v_keyedConfig_843_ = lean_ctor_get(v_a_838_, 0);
v_trackZetaDelta_844_ = lean_ctor_get_uint8(v_a_838_, sizeof(void*)*7);
v_zetaDeltaSet_845_ = lean_ctor_get(v_a_838_, 1);
v_lctx_846_ = lean_ctor_get(v_a_838_, 2);
v_localInstances_847_ = lean_ctor_get(v_a_838_, 3);
v_defEqCtx_x3f_848_ = lean_ctor_get(v_a_838_, 4);
v_synthPendingDepth_849_ = lean_ctor_get(v_a_838_, 5);
v_customCanUnfoldPredicate_x3f_850_ = lean_ctor_get(v_a_838_, 6);
v_univApprox_851_ = lean_ctor_get_uint8(v_a_838_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_852_ = lean_ctor_get_uint8(v_a_838_, sizeof(void*)*7 + 2);
v_cacheInferType_853_ = lean_ctor_get_uint8(v_a_838_, sizeof(void*)*7 + 3);
v___f_854_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___lam__0___boxed), 7, 2);
lean_closure_set(v___f_854_, 0, v_target_834_);
lean_closure_set(v___f_854_, 1, v_excluded_836_);
lean_inc_ref(v_keyedConfig_843_);
v___x_855_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_835_, v_keyedConfig_843_);
lean_inc(v_customCanUnfoldPredicate_x3f_850_);
lean_inc(v_synthPendingDepth_849_);
lean_inc(v_defEqCtx_x3f_848_);
lean_inc_ref(v_localInstances_847_);
lean_inc_ref(v_lctx_846_);
lean_inc(v_zetaDeltaSet_845_);
v___x_856_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_856_, 0, v___x_855_);
lean_ctor_set(v___x_856_, 1, v_zetaDeltaSet_845_);
lean_ctor_set(v___x_856_, 2, v_lctx_846_);
lean_ctor_set(v___x_856_, 3, v_localInstances_847_);
lean_ctor_set(v___x_856_, 4, v_defEqCtx_x3f_848_);
lean_ctor_set(v___x_856_, 5, v_synthPendingDepth_849_);
lean_ctor_set(v___x_856_, 6, v_customCanUnfoldPredicate_x3f_850_);
lean_ctor_set_uint8(v___x_856_, sizeof(void*)*7, v_trackZetaDelta_844_);
lean_ctor_set_uint8(v___x_856_, sizeof(void*)*7 + 1, v_univApprox_851_);
lean_ctor_set_uint8(v___x_856_, sizeof(void*)*7 + 2, v_inTypeClassResolution_852_);
lean_ctor_set_uint8(v___x_856_, sizeof(void*)*7 + 3, v_cacheInferType_853_);
v___x_857_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp_spec__3___redArg(v_goal_837_, v___f_854_, v___x_856_, v_a_839_, v_a_840_, v_a_841_);
lean_dec_ref_known(v___x_856_, 7);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp___boxed(lean_object* v_target_858_, lean_object* v_md_859_, lean_object* v_excluded_860_, lean_object* v_goal_861_, lean_object* v_a_862_, lean_object* v_a_863_, lean_object* v_a_864_, lean_object* v_a_865_, lean_object* v_a_866_){
_start:
{
uint8_t v_md_boxed_867_; lean_object* v_res_868_; 
v_md_boxed_867_ = lean_unbox(v_md_859_);
v_res_868_ = lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp(v_target_858_, v_md_boxed_867_, v_excluded_860_, v_goal_861_, v_a_862_, v_a_863_, v_a_864_, v_a_865_);
lean_dec(v_a_865_);
lean_dec_ref(v_a_864_);
lean_dec(v_a_863_);
lean_dec_ref(v_a_862_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__1(lean_object* v_x_869_, lean_object* v_x_870_){
_start:
{
if (lean_obj_tag(v_x_870_) == 0)
{
return v_x_869_;
}
else
{
lean_object* v_key_871_; lean_object* v_tail_872_; lean_object* v___x_873_; 
v_key_871_ = lean_ctor_get(v_x_870_, 0);
lean_inc(v_key_871_);
v_tail_872_ = lean_ctor_get(v_x_870_, 2);
lean_inc(v_tail_872_);
lean_dec_ref_known(v_x_870_, 3);
v___x_873_ = lean_array_push(v_x_869_, v_key_871_);
v_x_869_ = v___x_873_;
v_x_870_ = v_tail_872_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2(lean_object* v_as_875_, size_t v_i_876_, size_t v_stop_877_, lean_object* v_b_878_){
_start:
{
uint8_t v___x_879_; 
v___x_879_ = lean_usize_dec_eq(v_i_876_, v_stop_877_);
if (v___x_879_ == 0)
{
lean_object* v___x_880_; lean_object* v___x_881_; size_t v___x_882_; size_t v___x_883_; 
v___x_880_ = lean_array_uget_borrowed(v_as_875_, v_i_876_);
lean_inc(v___x_880_);
v___x_881_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__1(v_b_878_, v___x_880_);
v___x_882_ = ((size_t)1ULL);
v___x_883_ = lean_usize_add(v_i_876_, v___x_882_);
v_i_876_ = v___x_883_;
v_b_878_ = v___x_881_;
goto _start;
}
else
{
return v_b_878_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2___boxed(lean_object* v_as_885_, lean_object* v_i_886_, lean_object* v_stop_887_, lean_object* v_b_888_){
_start:
{
size_t v_i_boxed_889_; size_t v_stop_boxed_890_; lean_object* v_res_891_; 
v_i_boxed_889_ = lean_unbox_usize(v_i_886_);
lean_dec(v_i_886_);
v_stop_boxed_890_ = lean_unbox_usize(v_stop_887_);
lean_dec(v_stop_887_);
v_res_891_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2(v_as_885_, v_i_boxed_889_, v_stop_boxed_890_, v_b_888_);
lean_dec_ref(v_as_885_);
return v_res_891_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0(lean_object* v___x_892_, size_t v_sz_893_, size_t v_i_894_, lean_object* v_bs_895_){
_start:
{
uint8_t v___x_896_; 
v___x_896_ = lean_usize_dec_lt(v_i_894_, v_sz_893_);
if (v___x_896_ == 0)
{
return v_bs_895_;
}
else
{
lean_object* v_subst_897_; lean_object* v_v_898_; lean_object* v___x_899_; lean_object* v_bs_x27_900_; lean_object* v___y_902_; lean_object* v___x_907_; 
v_subst_897_ = lean_ctor_get(v___x_892_, 2);
v_v_898_ = lean_array_uget(v_bs_895_, v_i_894_);
v___x_899_ = lean_unsigned_to_nat(0u);
v_bs_x27_900_ = lean_array_uset(v_bs_895_, v_i_894_, v___x_899_);
lean_inc(v_v_898_);
v___x_907_ = l_Lean_Meta_FVarSubst_get(v_subst_897_, v_v_898_);
if (lean_obj_tag(v___x_907_) == 1)
{
lean_object* v_fvarId_908_; 
lean_dec(v_v_898_);
v_fvarId_908_ = lean_ctor_get(v___x_907_, 0);
lean_inc(v_fvarId_908_);
lean_dec_ref_known(v___x_907_, 1);
v___y_902_ = v_fvarId_908_;
goto v___jp_901_;
}
else
{
lean_dec_ref(v___x_907_);
v___y_902_ = v_v_898_;
goto v___jp_901_;
}
v___jp_901_:
{
size_t v___x_903_; size_t v___x_904_; lean_object* v___x_905_; 
v___x_903_ = ((size_t)1ULL);
v___x_904_ = lean_usize_add(v_i_894_, v___x_903_);
v___x_905_ = lean_array_uset(v_bs_x27_900_, v_i_894_, v___y_902_);
v_i_894_ = v___x_904_;
v_bs_895_ = v___x_905_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0___boxed(lean_object* v___x_909_, lean_object* v_sz_910_, lean_object* v_i_911_, lean_object* v_bs_912_){
_start:
{
size_t v_sz_boxed_913_; size_t v_i_boxed_914_; lean_object* v_res_915_; 
v_sz_boxed_913_ = lean_unbox_usize(v_sz_910_);
lean_dec(v_sz_910_);
v_i_boxed_914_ = lean_unbox_usize(v_i_911_);
lean_dec(v_i_911_);
v_res_915_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0(v___x_909_, v_sz_boxed_913_, v_i_boxed_914_, v_bs_912_);
lean_dec_ref(v___x_909_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg(lean_object* v_val_916_, lean_object* v_target_917_, uint8_t v_md_918_, uint8_t v_isRecursiveType_919_, lean_object* v_ctorNames_920_, lean_object* v_initialGoal_921_, lean_object* v_goal_922_, lean_object* v_excluded_923_, lean_object* v_range_924_, lean_object* v_b_925_, lean_object* v_i_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
lean_object* v_stop_934_; lean_object* v_step_935_; lean_object* v_a_937_; uint8_t v___x_940_; 
v_stop_934_ = lean_ctor_get(v_range_924_, 1);
v_step_935_ = lean_ctor_get(v_range_924_, 2);
v___x_940_ = lean_nat_dec_lt(v_i_926_, v_stop_934_);
if (v___x_940_ == 0)
{
lean_object* v___x_941_; 
lean_dec(v_i_926_);
lean_dec_ref(v_excluded_923_);
lean_dec(v_goal_922_);
lean_dec(v_initialGoal_921_);
lean_dec_ref(v_target_917_);
v___x_941_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_941_, 0, v_b_925_);
return v___x_941_;
}
else
{
lean_object* v___x_942_; lean_object* v_toInductionSubgoal_943_; lean_object* v_mvarId_944_; lean_object* v___x_945_; 
v___x_942_ = lean_array_fget_borrowed(v_val_916_, v_i_926_);
v_toInductionSubgoal_943_ = lean_ctor_get(v___x_942_, 0);
v_mvarId_944_ = lean_ctor_get(v_toInductionSubgoal_943_, 0);
lean_inc(v_mvarId_944_);
lean_inc(v_goal_922_);
v___x_945_ = lp_aesop_Aesop_diffGoals(v_goal_922_, v_mvarId_944_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
if (lean_obj_tag(v___x_945_) == 0)
{
lean_object* v_a_946_; lean_object* v_excluded_948_; lean_object* v___y_949_; lean_object* v___y_950_; lean_object* v___y_951_; lean_object* v___y_952_; lean_object* v___y_953_; lean_object* v___y_954_; 
v_a_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc(v_a_946_);
lean_dec_ref_known(v___x_945_, 1);
if (v_isRecursiveType_919_ == 0)
{
lean_dec(v_a_946_);
lean_inc_ref(v_excluded_923_);
v_excluded_948_ = v_excluded_923_;
v___y_949_ = v___y_927_;
v___y_950_ = v___y_928_;
v___y_951_ = v___y_929_;
v___y_952_ = v___y_930_;
v___y_953_ = v___y_931_;
v___y_954_ = v___y_932_;
goto v___jp_947_;
}
else
{
lean_object* v_addedFVars_977_; lean_object* v_size_978_; lean_object* v_buckets_979_; size_t v_sz_980_; size_t v___x_981_; lean_object* v___x_982_; lean_object* v___y_984_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; uint8_t v___x_989_; 
v_addedFVars_977_ = lean_ctor_get(v_a_946_, 2);
lean_inc_ref(v_addedFVars_977_);
lean_dec(v_a_946_);
v_size_978_ = lean_ctor_get(v_addedFVars_977_, 0);
lean_inc(v_size_978_);
v_buckets_979_ = lean_ctor_get(v_addedFVars_977_, 1);
lean_inc_ref(v_buckets_979_);
lean_dec_ref(v_addedFVars_977_);
v_sz_980_ = lean_array_size(v_excluded_923_);
v___x_981_ = ((size_t)0ULL);
lean_inc_ref(v_excluded_923_);
v___x_982_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__0(v_toInductionSubgoal_943_, v_sz_980_, v___x_981_, v_excluded_923_);
v___x_986_ = lean_mk_empty_array_with_capacity(v_size_978_);
lean_dec(v_size_978_);
v___x_987_ = lean_unsigned_to_nat(0u);
v___x_988_ = lean_array_get_size(v_buckets_979_);
v___x_989_ = lean_nat_dec_lt(v___x_987_, v___x_988_);
if (v___x_989_ == 0)
{
lean_dec_ref(v_buckets_979_);
v___y_984_ = v___x_986_;
goto v___jp_983_;
}
else
{
uint8_t v___x_990_; 
v___x_990_ = lean_nat_dec_le(v___x_988_, v___x_988_);
if (v___x_990_ == 0)
{
if (v___x_989_ == 0)
{
lean_dec_ref(v_buckets_979_);
v___y_984_ = v___x_986_;
goto v___jp_983_;
}
else
{
size_t v___x_991_; lean_object* v___x_992_; 
v___x_991_ = lean_usize_of_nat(v___x_988_);
v___x_992_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2(v_buckets_979_, v___x_981_, v___x_991_, v___x_986_);
lean_dec_ref(v_buckets_979_);
v___y_984_ = v___x_992_;
goto v___jp_983_;
}
}
else
{
size_t v___x_993_; lean_object* v___x_994_; 
v___x_993_ = lean_usize_of_nat(v___x_988_);
v___x_994_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__2(v_buckets_979_, v___x_981_, v___x_993_, v___x_986_);
lean_dec_ref(v_buckets_979_);
v___y_984_ = v___x_994_;
goto v___jp_983_;
}
}
v___jp_983_:
{
lean_object* v___x_985_; 
v___x_985_ = l_Array_append___redArg(v___x_982_, v___y_984_);
lean_dec_ref(v___y_984_);
v_excluded_948_ = v___x_985_;
v___y_949_ = v___y_927_;
v___y_950_ = v___y_928_;
v___y_951_ = v___y_929_;
v___y_952_ = v___y_930_;
v___y_953_ = v___y_931_;
v___y_954_ = v___y_932_;
goto v___jp_947_;
}
}
v___jp_947_:
{
lean_object* v___x_955_; 
lean_inc(v_mvarId_944_);
lean_inc_ref(v_b_925_);
lean_inc(v_initialGoal_921_);
lean_inc_ref(v_target_917_);
v___x_955_ = lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go(v_target_917_, v_md_918_, v_isRecursiveType_919_, v_ctorNames_920_, v_initialGoal_921_, v_b_925_, v_excluded_948_, v_mvarId_944_, v___y_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_);
if (lean_obj_tag(v___x_955_) == 0)
{
lean_object* v_a_956_; 
v_a_956_ = lean_ctor_get(v___x_955_, 0);
lean_inc(v_a_956_);
lean_dec_ref_known(v___x_955_, 1);
if (lean_obj_tag(v_a_956_) == 0)
{
lean_object* v___x_957_; 
lean_inc(v_mvarId_944_);
lean_inc(v_initialGoal_921_);
v___x_957_ = lp_aesop_Aesop_diffGoals(v_initialGoal_921_, v_mvarId_944_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_);
if (lean_obj_tag(v___x_957_) == 0)
{
lean_object* v_a_958_; lean_object* v___x_959_; 
v_a_958_ = lean_ctor_get(v___x_957_, 0);
lean_inc(v_a_958_);
lean_dec_ref_known(v___x_957_, 1);
v___x_959_ = lean_array_push(v_b_925_, v_a_958_);
v_a_937_ = v___x_959_;
goto v___jp_936_;
}
else
{
lean_object* v_a_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_967_; 
lean_dec(v_i_926_);
lean_dec_ref(v_b_925_);
lean_dec_ref(v_excluded_923_);
lean_dec(v_goal_922_);
lean_dec(v_initialGoal_921_);
lean_dec_ref(v_target_917_);
v_a_960_ = lean_ctor_get(v___x_957_, 0);
v_isSharedCheck_967_ = !lean_is_exclusive(v___x_957_);
if (v_isSharedCheck_967_ == 0)
{
v___x_962_ = v___x_957_;
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_a_960_);
lean_dec(v___x_957_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_965_; 
if (v_isShared_963_ == 0)
{
v___x_965_ = v___x_962_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_a_960_);
v___x_965_ = v_reuseFailAlloc_966_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
return v___x_965_;
}
}
}
}
else
{
lean_object* v_val_968_; 
lean_dec_ref(v_b_925_);
v_val_968_ = lean_ctor_get(v_a_956_, 0);
lean_inc(v_val_968_);
lean_dec_ref_known(v_a_956_, 1);
v_a_937_ = v_val_968_;
goto v___jp_936_;
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
lean_dec(v_i_926_);
lean_dec_ref(v_b_925_);
lean_dec_ref(v_excluded_923_);
lean_dec(v_goal_922_);
lean_dec(v_initialGoal_921_);
lean_dec_ref(v_target_917_);
v_a_969_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v___x_955_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_955_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
}
else
{
lean_object* v_a_995_; lean_object* v___x_997_; uint8_t v_isShared_998_; uint8_t v_isSharedCheck_1002_; 
lean_dec(v_i_926_);
lean_dec_ref(v_b_925_);
lean_dec_ref(v_excluded_923_);
lean_dec(v_goal_922_);
lean_dec(v_initialGoal_921_);
lean_dec_ref(v_target_917_);
v_a_995_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_997_ = v___x_945_;
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
else
{
lean_inc(v_a_995_);
lean_dec(v___x_945_);
v___x_997_ = lean_box(0);
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
v_resetjp_996_:
{
lean_object* v___x_1000_; 
if (v_isShared_998_ == 0)
{
v___x_1000_ = v___x_997_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1001_; 
v_reuseFailAlloc_1001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1001_, 0, v_a_995_);
v___x_1000_ = v_reuseFailAlloc_1001_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
return v___x_1000_;
}
}
}
}
v___jp_936_:
{
lean_object* v___x_938_; 
v___x_938_ = lean_nat_add(v_i_926_, v_step_935_);
lean_dec(v_i_926_);
v_b_925_ = v_a_937_;
v_i_926_ = v___x_938_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go(lean_object* v_target_1003_, uint8_t v_md_1004_, uint8_t v_isRecursiveType_1005_, lean_object* v_ctorNames_1006_, lean_object* v_initialGoal_1007_, lean_object* v_newGoals_1008_, lean_object* v_excluded_1009_, lean_object* v_goal_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_){
_start:
{
lean_object* v___x_1018_; 
lean_inc(v_goal_1010_);
lean_inc_ref(v_excluded_1009_);
lean_inc_ref(v_target_1003_);
v___x_1018_ = lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_findFirstApplicableHyp(v_target_1003_, v_md_1004_, v_excluded_1009_, v_goal_1010_, v_a_1013_, v_a_1014_, v_a_1015_, v_a_1016_);
if (lean_obj_tag(v___x_1018_) == 0)
{
lean_object* v_a_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1075_; 
v_a_1019_ = lean_ctor_get(v___x_1018_, 0);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1018_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1021_ = v___x_1018_;
v_isShared_1022_ = v_isSharedCheck_1075_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_a_1019_);
lean_dec(v___x_1018_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1075_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
if (lean_obj_tag(v_a_1019_) == 1)
{
lean_object* v_val_1023_; lean_object* v___x_1024_; 
lean_del_object(v___x_1021_);
v_val_1023_ = lean_ctor_get(v_a_1019_, 0);
lean_inc(v_val_1023_);
lean_dec_ref_known(v_a_1019_, 1);
lean_inc(v_goal_1010_);
v___x_1024_ = lp_aesop_Aesop_tryCasesS___redArg(v_goal_1010_, v_val_1023_, v_ctorNames_1006_, v_a_1011_, v_a_1013_, v_a_1014_, v_a_1015_, v_a_1016_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v_a_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1062_; 
v_a_1025_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1062_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1062_ == 0)
{
v___x_1027_ = v___x_1024_;
v_isShared_1028_ = v_isSharedCheck_1062_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_a_1025_);
lean_dec(v___x_1024_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1062_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
if (lean_obj_tag(v_a_1025_) == 1)
{
lean_object* v_val_1029_; lean_object* v___x_1031_; uint8_t v_isShared_1032_; uint8_t v_isSharedCheck_1057_; 
lean_del_object(v___x_1027_);
v_val_1029_ = lean_ctor_get(v_a_1025_, 0);
v_isSharedCheck_1057_ = !lean_is_exclusive(v_a_1025_);
if (v_isSharedCheck_1057_ == 0)
{
v___x_1031_ = v_a_1025_;
v_isShared_1032_ = v_isSharedCheck_1057_;
goto v_resetjp_1030_;
}
else
{
lean_inc(v_val_1029_);
lean_dec(v_a_1025_);
v___x_1031_ = lean_box(0);
v_isShared_1032_ = v_isSharedCheck_1057_;
goto v_resetjp_1030_;
}
v_resetjp_1030_:
{
lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; 
v___x_1033_ = lean_unsigned_to_nat(0u);
v___x_1034_ = lean_array_get_size(v_val_1029_);
v___x_1035_ = lean_unsigned_to_nat(1u);
v___x_1036_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1033_);
lean_ctor_set(v___x_1036_, 1, v___x_1034_);
lean_ctor_set(v___x_1036_, 2, v___x_1035_);
v___x_1037_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg(v_val_1029_, v_target_1003_, v_md_1004_, v_isRecursiveType_1005_, v_ctorNames_1006_, v_initialGoal_1007_, v_goal_1010_, v_excluded_1009_, v___x_1036_, v_newGoals_1008_, v___x_1033_, v_a_1011_, v_a_1012_, v_a_1013_, v_a_1014_, v_a_1015_, v_a_1016_);
lean_dec_ref_known(v___x_1036_, 3);
lean_dec(v_val_1029_);
if (lean_obj_tag(v___x_1037_) == 0)
{
lean_object* v_a_1038_; lean_object* v___x_1040_; uint8_t v_isShared_1041_; uint8_t v_isSharedCheck_1048_; 
v_a_1038_ = lean_ctor_get(v___x_1037_, 0);
v_isSharedCheck_1048_ = !lean_is_exclusive(v___x_1037_);
if (v_isSharedCheck_1048_ == 0)
{
v___x_1040_ = v___x_1037_;
v_isShared_1041_ = v_isSharedCheck_1048_;
goto v_resetjp_1039_;
}
else
{
lean_inc(v_a_1038_);
lean_dec(v___x_1037_);
v___x_1040_ = lean_box(0);
v_isShared_1041_ = v_isSharedCheck_1048_;
goto v_resetjp_1039_;
}
v_resetjp_1039_:
{
lean_object* v___x_1043_; 
if (v_isShared_1032_ == 0)
{
lean_ctor_set(v___x_1031_, 0, v_a_1038_);
v___x_1043_ = v___x_1031_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1047_; 
v_reuseFailAlloc_1047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1047_, 0, v_a_1038_);
v___x_1043_ = v_reuseFailAlloc_1047_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
lean_object* v___x_1045_; 
if (v_isShared_1041_ == 0)
{
lean_ctor_set(v___x_1040_, 0, v___x_1043_);
v___x_1045_ = v___x_1040_;
goto v_reusejp_1044_;
}
else
{
lean_object* v_reuseFailAlloc_1046_; 
v_reuseFailAlloc_1046_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1046_, 0, v___x_1043_);
v___x_1045_ = v_reuseFailAlloc_1046_;
goto v_reusejp_1044_;
}
v_reusejp_1044_:
{
return v___x_1045_;
}
}
}
}
else
{
lean_object* v_a_1049_; lean_object* v___x_1051_; uint8_t v_isShared_1052_; uint8_t v_isSharedCheck_1056_; 
lean_del_object(v___x_1031_);
v_a_1049_ = lean_ctor_get(v___x_1037_, 0);
v_isSharedCheck_1056_ = !lean_is_exclusive(v___x_1037_);
if (v_isSharedCheck_1056_ == 0)
{
v___x_1051_ = v___x_1037_;
v_isShared_1052_ = v_isSharedCheck_1056_;
goto v_resetjp_1050_;
}
else
{
lean_inc(v_a_1049_);
lean_dec(v___x_1037_);
v___x_1051_ = lean_box(0);
v_isShared_1052_ = v_isSharedCheck_1056_;
goto v_resetjp_1050_;
}
v_resetjp_1050_:
{
lean_object* v___x_1054_; 
if (v_isShared_1052_ == 0)
{
v___x_1054_ = v___x_1051_;
goto v_reusejp_1053_;
}
else
{
lean_object* v_reuseFailAlloc_1055_; 
v_reuseFailAlloc_1055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1055_, 0, v_a_1049_);
v___x_1054_ = v_reuseFailAlloc_1055_;
goto v_reusejp_1053_;
}
v_reusejp_1053_:
{
return v___x_1054_;
}
}
}
}
}
else
{
lean_object* v___x_1058_; lean_object* v___x_1060_; 
lean_dec(v_a_1025_);
lean_dec(v_goal_1010_);
lean_dec_ref(v_excluded_1009_);
lean_dec_ref(v_newGoals_1008_);
lean_dec(v_initialGoal_1007_);
lean_dec_ref(v_target_1003_);
v___x_1058_ = lean_box(0);
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 0, v___x_1058_);
v___x_1060_ = v___x_1027_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v___x_1058_);
v___x_1060_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
return v___x_1060_;
}
}
}
}
else
{
lean_object* v_a_1063_; lean_object* v___x_1065_; uint8_t v_isShared_1066_; uint8_t v_isSharedCheck_1070_; 
lean_dec(v_goal_1010_);
lean_dec_ref(v_excluded_1009_);
lean_dec_ref(v_newGoals_1008_);
lean_dec(v_initialGoal_1007_);
lean_dec_ref(v_target_1003_);
v_a_1063_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1065_ = v___x_1024_;
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
else
{
lean_inc(v_a_1063_);
lean_dec(v___x_1024_);
v___x_1065_ = lean_box(0);
v_isShared_1066_ = v_isSharedCheck_1070_;
goto v_resetjp_1064_;
}
v_resetjp_1064_:
{
lean_object* v___x_1068_; 
if (v_isShared_1066_ == 0)
{
v___x_1068_ = v___x_1065_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_a_1063_);
v___x_1068_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
return v___x_1068_;
}
}
}
}
else
{
lean_object* v___x_1071_; lean_object* v___x_1073_; 
lean_dec(v_a_1019_);
lean_dec(v_goal_1010_);
lean_dec_ref(v_excluded_1009_);
lean_dec_ref(v_newGoals_1008_);
lean_dec(v_initialGoal_1007_);
lean_dec_ref(v_target_1003_);
v___x_1071_ = lean_box(0);
if (v_isShared_1022_ == 0)
{
lean_ctor_set(v___x_1021_, 0, v___x_1071_);
v___x_1073_ = v___x_1021_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v___x_1071_);
v___x_1073_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
return v___x_1073_;
}
}
}
}
else
{
lean_object* v_a_1076_; lean_object* v___x_1078_; uint8_t v_isShared_1079_; uint8_t v_isSharedCheck_1083_; 
lean_dec(v_goal_1010_);
lean_dec_ref(v_excluded_1009_);
lean_dec_ref(v_newGoals_1008_);
lean_dec(v_initialGoal_1007_);
lean_dec_ref(v_target_1003_);
v_a_1076_ = lean_ctor_get(v___x_1018_, 0);
v_isSharedCheck_1083_ = !lean_is_exclusive(v___x_1018_);
if (v_isSharedCheck_1083_ == 0)
{
v___x_1078_ = v___x_1018_;
v_isShared_1079_ = v_isSharedCheck_1083_;
goto v_resetjp_1077_;
}
else
{
lean_inc(v_a_1076_);
lean_dec(v___x_1018_);
v___x_1078_ = lean_box(0);
v_isShared_1079_ = v_isSharedCheck_1083_;
goto v_resetjp_1077_;
}
v_resetjp_1077_:
{
lean_object* v___x_1081_; 
if (v_isShared_1079_ == 0)
{
v___x_1081_ = v___x_1078_;
goto v_reusejp_1080_;
}
else
{
lean_object* v_reuseFailAlloc_1082_; 
v_reuseFailAlloc_1082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1082_, 0, v_a_1076_);
v___x_1081_ = v_reuseFailAlloc_1082_;
goto v_reusejp_1080_;
}
v_reusejp_1080_:
{
return v___x_1081_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go___boxed(lean_object* v_target_1084_, lean_object* v_md_1085_, lean_object* v_isRecursiveType_1086_, lean_object* v_ctorNames_1087_, lean_object* v_initialGoal_1088_, lean_object* v_newGoals_1089_, lean_object* v_excluded_1090_, lean_object* v_goal_1091_, lean_object* v_a_1092_, lean_object* v_a_1093_, lean_object* v_a_1094_, lean_object* v_a_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_, lean_object* v_a_1098_){
_start:
{
uint8_t v_md_boxed_1099_; uint8_t v_isRecursiveType_boxed_1100_; lean_object* v_res_1101_; 
v_md_boxed_1099_ = lean_unbox(v_md_1085_);
v_isRecursiveType_boxed_1100_ = lean_unbox(v_isRecursiveType_1086_);
v_res_1101_ = lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go(v_target_1084_, v_md_boxed_1099_, v_isRecursiveType_boxed_1100_, v_ctorNames_1087_, v_initialGoal_1088_, v_newGoals_1089_, v_excluded_1090_, v_goal_1091_, v_a_1092_, v_a_1093_, v_a_1094_, v_a_1095_, v_a_1096_, v_a_1097_);
lean_dec(v_a_1097_);
lean_dec_ref(v_a_1096_);
lean_dec(v_a_1095_);
lean_dec_ref(v_a_1094_);
lean_dec(v_a_1093_);
lean_dec(v_a_1092_);
lean_dec_ref(v_ctorNames_1087_);
return v_res_1101_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg___boxed(lean_object** _args){
lean_object* v_val_1102_ = _args[0];
lean_object* v_target_1103_ = _args[1];
lean_object* v_md_1104_ = _args[2];
lean_object* v_isRecursiveType_1105_ = _args[3];
lean_object* v_ctorNames_1106_ = _args[4];
lean_object* v_initialGoal_1107_ = _args[5];
lean_object* v_goal_1108_ = _args[6];
lean_object* v_excluded_1109_ = _args[7];
lean_object* v_range_1110_ = _args[8];
lean_object* v_b_1111_ = _args[9];
lean_object* v_i_1112_ = _args[10];
lean_object* v___y_1113_ = _args[11];
lean_object* v___y_1114_ = _args[12];
lean_object* v___y_1115_ = _args[13];
lean_object* v___y_1116_ = _args[14];
lean_object* v___y_1117_ = _args[15];
lean_object* v___y_1118_ = _args[16];
lean_object* v___y_1119_ = _args[17];
_start:
{
uint8_t v_md_boxed_1120_; uint8_t v_isRecursiveType_boxed_1121_; lean_object* v_res_1122_; 
v_md_boxed_1120_ = lean_unbox(v_md_1104_);
v_isRecursiveType_boxed_1121_ = lean_unbox(v_isRecursiveType_1105_);
v_res_1122_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg(v_val_1102_, v_target_1103_, v_md_boxed_1120_, v_isRecursiveType_boxed_1121_, v_ctorNames_1106_, v_initialGoal_1107_, v_goal_1108_, v_excluded_1109_, v_range_1110_, v_b_1111_, v_i_1112_, v___y_1113_, v___y_1114_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_);
lean_dec(v___y_1118_);
lean_dec_ref(v___y_1117_);
lean_dec(v___y_1116_);
lean_dec_ref(v___y_1115_);
lean_dec(v___y_1114_);
lean_dec(v___y_1113_);
lean_dec_ref(v_range_1110_);
lean_dec_ref(v_ctorNames_1106_);
lean_dec_ref(v_val_1102_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3(lean_object* v_val_1123_, lean_object* v_target_1124_, uint8_t v_md_1125_, uint8_t v_isRecursiveType_1126_, lean_object* v_ctorNames_1127_, lean_object* v_initialGoal_1128_, lean_object* v_goal_1129_, lean_object* v_excluded_1130_, lean_object* v_range_1131_, lean_object* v_b_1132_, lean_object* v_i_1133_, lean_object* v_hs_1134_, lean_object* v_hl_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
lean_object* v___x_1143_; 
v___x_1143_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___redArg(v_val_1123_, v_target_1124_, v_md_1125_, v_isRecursiveType_1126_, v_ctorNames_1127_, v_initialGoal_1128_, v_goal_1129_, v_excluded_1130_, v_range_1131_, v_b_1132_, v_i_1133_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_);
return v___x_1143_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3___boxed(lean_object** _args){
lean_object* v_val_1144_ = _args[0];
lean_object* v_target_1145_ = _args[1];
lean_object* v_md_1146_ = _args[2];
lean_object* v_isRecursiveType_1147_ = _args[3];
lean_object* v_ctorNames_1148_ = _args[4];
lean_object* v_initialGoal_1149_ = _args[5];
lean_object* v_goal_1150_ = _args[6];
lean_object* v_excluded_1151_ = _args[7];
lean_object* v_range_1152_ = _args[8];
lean_object* v_b_1153_ = _args[9];
lean_object* v_i_1154_ = _args[10];
lean_object* v_hs_1155_ = _args[11];
lean_object* v_hl_1156_ = _args[12];
lean_object* v___y_1157_ = _args[13];
lean_object* v___y_1158_ = _args[14];
lean_object* v___y_1159_ = _args[15];
lean_object* v___y_1160_ = _args[16];
lean_object* v___y_1161_ = _args[17];
lean_object* v___y_1162_ = _args[18];
lean_object* v___y_1163_ = _args[19];
_start:
{
uint8_t v_md_boxed_1164_; uint8_t v_isRecursiveType_boxed_1165_; lean_object* v_res_1166_; 
v_md_boxed_1164_ = lean_unbox(v_md_1146_);
v_isRecursiveType_boxed_1165_ = lean_unbox(v_isRecursiveType_1147_);
v_res_1166_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go_spec__3(v_val_1144_, v_target_1145_, v_md_boxed_1164_, v_isRecursiveType_boxed_1165_, v_ctorNames_1148_, v_initialGoal_1149_, v_goal_1150_, v_excluded_1151_, v_range_1152_, v_b_1153_, v_i_1154_, v_hs_1155_, v_hl_1156_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_);
lean_dec(v___y_1162_);
lean_dec_ref(v___y_1161_);
lean_dec(v___y_1160_);
lean_dec_ref(v___y_1159_);
lean_dec(v___y_1158_);
lean_dec(v___y_1157_);
lean_dec_ref(v_range_1152_);
lean_dec_ref(v_ctorNames_1148_);
lean_dec_ref(v_val_1144_);
return v_res_1166_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg(lean_object* v_x_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_){
_start:
{
lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1176_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___closed__0));
v___x_1177_ = lean_st_mk_ref(v___x_1176_);
lean_inc(v___y_1174_);
lean_inc_ref(v___y_1173_);
lean_inc(v___y_1172_);
lean_inc_ref(v___y_1171_);
lean_inc(v___y_1170_);
lean_inc(v___x_1177_);
v___x_1178_ = lean_apply_7(v_x_1169_, v___x_1177_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, lean_box(0));
if (lean_obj_tag(v___x_1178_) == 0)
{
lean_object* v_a_1179_; lean_object* v___x_1181_; uint8_t v_isShared_1182_; uint8_t v_isSharedCheck_1188_; 
v_a_1179_ = lean_ctor_get(v___x_1178_, 0);
v_isSharedCheck_1188_ = !lean_is_exclusive(v___x_1178_);
if (v_isSharedCheck_1188_ == 0)
{
v___x_1181_ = v___x_1178_;
v_isShared_1182_ = v_isSharedCheck_1188_;
goto v_resetjp_1180_;
}
else
{
lean_inc(v_a_1179_);
lean_dec(v___x_1178_);
v___x_1181_ = lean_box(0);
v_isShared_1182_ = v_isSharedCheck_1188_;
goto v_resetjp_1180_;
}
v_resetjp_1180_:
{
lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1186_; 
v___x_1183_ = lean_st_ref_get(v___x_1177_);
lean_dec(v___x_1177_);
v___x_1184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1184_, 0, v_a_1179_);
lean_ctor_set(v___x_1184_, 1, v___x_1183_);
if (v_isShared_1182_ == 0)
{
lean_ctor_set(v___x_1181_, 0, v___x_1184_);
v___x_1186_ = v___x_1181_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1187_; 
v_reuseFailAlloc_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1187_, 0, v___x_1184_);
v___x_1186_ = v_reuseFailAlloc_1187_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
return v___x_1186_;
}
}
}
else
{
lean_object* v_a_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1196_; 
lean_dec(v___x_1177_);
v_a_1189_ = lean_ctor_get(v___x_1178_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1178_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1191_ = v___x_1178_;
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_a_1189_);
lean_dec(v___x_1178_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v___x_1194_; 
if (v_isShared_1192_ == 0)
{
v___x_1194_ = v___x_1191_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_a_1189_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg___boxed(lean_object* v_x_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_){
_start:
{
lean_object* v_res_1204_; 
v_res_1204_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg(v_x_1197_, v___y_1198_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v___y_1198_);
return v_res_1204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0(lean_object* v_00_u03b1_1205_, lean_object* v_x_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_){
_start:
{
lean_object* v___x_1213_; 
v___x_1213_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg(v_x_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___boxed(lean_object* v_00_u03b1_1214_, lean_object* v_x_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_){
_start:
{
lean_object* v_res_1222_; 
v_res_1222_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0(v_00_u03b1_1214_, v_x_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
lean_dec(v___y_1218_);
lean_dec_ref(v___y_1217_);
lean_dec(v___y_1216_);
return v_res_1222_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1(lean_object* v_msgData_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_){
_start:
{
lean_object* v___x_1229_; lean_object* v_env_1230_; lean_object* v___x_1231_; lean_object* v_mctx_1232_; lean_object* v_lctx_1233_; lean_object* v_options_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; 
v___x_1229_ = lean_st_ref_get(v___y_1227_);
v_env_1230_ = lean_ctor_get(v___x_1229_, 0);
lean_inc_ref(v_env_1230_);
lean_dec(v___x_1229_);
v___x_1231_ = lean_st_ref_get(v___y_1225_);
v_mctx_1232_ = lean_ctor_get(v___x_1231_, 0);
lean_inc_ref(v_mctx_1232_);
lean_dec(v___x_1231_);
v_lctx_1233_ = lean_ctor_get(v___y_1224_, 2);
v_options_1234_ = lean_ctor_get(v___y_1226_, 2);
lean_inc_ref(v_options_1234_);
lean_inc_ref(v_lctx_1233_);
v___x_1235_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1235_, 0, v_env_1230_);
lean_ctor_set(v___x_1235_, 1, v_mctx_1232_);
lean_ctor_set(v___x_1235_, 2, v_lctx_1233_);
lean_ctor_set(v___x_1235_, 3, v_options_1234_);
v___x_1236_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1236_, 0, v___x_1235_);
lean_ctor_set(v___x_1236_, 1, v_msgData_1223_);
v___x_1237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1237_, 0, v___x_1236_);
return v___x_1237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1___boxed(lean_object* v_msgData_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_){
_start:
{
lean_object* v_res_1244_; 
v_res_1244_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1(v_msgData_1238_, v___y_1239_, v___y_1240_, v___y_1241_, v___y_1242_);
lean_dec(v___y_1242_);
lean_dec_ref(v___y_1241_);
lean_dec(v___y_1240_);
lean_dec_ref(v___y_1239_);
return v_res_1244_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg(lean_object* v_msg_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_){
_start:
{
lean_object* v_ref_1251_; lean_object* v___x_1252_; lean_object* v_a_1253_; lean_object* v___x_1255_; uint8_t v_isShared_1256_; uint8_t v_isSharedCheck_1261_; 
v_ref_1251_ = lean_ctor_get(v___y_1248_, 5);
v___x_1252_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_cases_spec__1_spec__1(v_msg_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_);
v_a_1253_ = lean_ctor_get(v___x_1252_, 0);
v_isSharedCheck_1261_ = !lean_is_exclusive(v___x_1252_);
if (v_isSharedCheck_1261_ == 0)
{
v___x_1255_ = v___x_1252_;
v_isShared_1256_ = v_isSharedCheck_1261_;
goto v_resetjp_1254_;
}
else
{
lean_inc(v_a_1253_);
lean_dec(v___x_1252_);
v___x_1255_ = lean_box(0);
v_isShared_1256_ = v_isSharedCheck_1261_;
goto v_resetjp_1254_;
}
v_resetjp_1254_:
{
lean_object* v___x_1257_; lean_object* v___x_1259_; 
lean_inc(v_ref_1251_);
v___x_1257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1257_, 0, v_ref_1251_);
lean_ctor_set(v___x_1257_, 1, v_a_1253_);
if (v_isShared_1256_ == 0)
{
lean_ctor_set_tag(v___x_1255_, 1);
lean_ctor_set(v___x_1255_, 0, v___x_1257_);
v___x_1259_ = v___x_1255_;
goto v_reusejp_1258_;
}
else
{
lean_object* v_reuseFailAlloc_1260_; 
v_reuseFailAlloc_1260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1260_, 0, v___x_1257_);
v___x_1259_ = v_reuseFailAlloc_1260_;
goto v_reusejp_1258_;
}
v_reusejp_1258_:
{
return v___x_1259_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg___boxed(lean_object* v_msg_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_){
_start:
{
lean_object* v_res_1268_; 
v_res_1268_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg(v_msg_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_);
lean_dec(v___y_1266_);
lean_dec_ref(v___y_1265_);
lean_dec(v___y_1264_);
lean_dec_ref(v___y_1263_);
return v_res_1268_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_cases___closed__2(void){
_start:
{
lean_object* v___x_1272_; lean_object* v___x_1273_; 
v___x_1272_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_cases___closed__1));
v___x_1273_ = l_Lean_stringToMessageData(v___x_1272_);
return v___x_1273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_cases(lean_object* v_target_1274_, uint8_t v_md_1275_, uint8_t v_isRecursiveType_1276_, lean_object* v_ctorNames_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_fst_1286_; lean_object* v_fst_1287_; lean_object* v_snd_1288_; lean_object* v_goal_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v_goal_1310_ = lean_ctor_get(v_a_1278_, 0);
lean_inc_n(v_goal_1310_, 2);
lean_dec_ref(v_a_1278_);
v___x_1311_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_cases___closed__0));
v___x_1312_ = lean_box(v_md_1275_);
v___x_1313_ = lean_box(v_isRecursiveType_1276_);
v___x_1314_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_RuleTac_Cases_0__Aesop_RuleTac_cases_go___boxed), 15, 8);
lean_closure_set(v___x_1314_, 0, v_target_1274_);
lean_closure_set(v___x_1314_, 1, v___x_1312_);
lean_closure_set(v___x_1314_, 2, v___x_1313_);
lean_closure_set(v___x_1314_, 3, v_ctorNames_1277_);
lean_closure_set(v___x_1314_, 4, v_goal_1310_);
lean_closure_set(v___x_1314_, 5, v___x_1311_);
lean_closure_set(v___x_1314_, 6, v___x_1311_);
lean_closure_set(v___x_1314_, 7, v_goal_1310_);
v___x_1315_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_cases_spec__0___redArg(v___x_1314_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_);
if (lean_obj_tag(v___x_1315_) == 0)
{
lean_object* v_a_1316_; lean_object* v_fst_1317_; 
v_a_1316_ = lean_ctor_get(v___x_1315_, 0);
lean_inc(v_a_1316_);
lean_dec_ref_known(v___x_1315_, 1);
v_fst_1317_ = lean_ctor_get(v_a_1316_, 0);
lean_inc(v_fst_1317_);
if (lean_obj_tag(v_fst_1317_) == 0)
{
lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v_a_1320_; lean_object* v___x_1322_; uint8_t v_isShared_1323_; uint8_t v_isSharedCheck_1327_; 
lean_dec(v_a_1316_);
v___x_1318_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_cases___closed__2, &lp_aesop_Aesop_RuleTac_cases___closed__2_once, _init_lp_aesop_Aesop_RuleTac_cases___closed__2);
v___x_1319_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg(v___x_1318_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_);
v_a_1320_ = lean_ctor_get(v___x_1319_, 0);
v_isSharedCheck_1327_ = !lean_is_exclusive(v___x_1319_);
if (v_isSharedCheck_1327_ == 0)
{
v___x_1322_ = v___x_1319_;
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
else
{
lean_inc(v_a_1320_);
lean_dec(v___x_1319_);
v___x_1322_ = lean_box(0);
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
v_resetjp_1321_:
{
lean_object* v___x_1325_; 
if (v_isShared_1323_ == 0)
{
v___x_1325_ = v___x_1322_;
goto v_reusejp_1324_;
}
else
{
lean_object* v_reuseFailAlloc_1326_; 
v_reuseFailAlloc_1326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1326_, 0, v_a_1320_);
v___x_1325_ = v_reuseFailAlloc_1326_;
goto v_reusejp_1324_;
}
v_reusejp_1324_:
{
return v___x_1325_;
}
}
}
else
{
lean_object* v_snd_1328_; lean_object* v_val_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1337_; 
v_snd_1328_ = lean_ctor_get(v_a_1316_, 1);
lean_inc(v_snd_1328_);
lean_dec(v_a_1316_);
v_val_1329_ = lean_ctor_get(v_fst_1317_, 0);
v_isSharedCheck_1337_ = !lean_is_exclusive(v_fst_1317_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1331_ = v_fst_1317_;
v_isShared_1332_ = v_isSharedCheck_1337_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_val_1329_);
lean_dec(v_fst_1317_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1337_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1334_; 
if (v_isShared_1332_ == 0)
{
lean_ctor_set(v___x_1331_, 0, v_snd_1328_);
v___x_1334_ = v___x_1331_;
goto v_reusejp_1333_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_snd_1328_);
v___x_1334_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1333_;
}
v_reusejp_1333_:
{
lean_object* v___x_1335_; 
v___x_1335_ = lean_box(0);
v_fst_1286_ = v_val_1329_;
v_fst_1287_ = v___x_1334_;
v_snd_1288_ = v___x_1335_;
goto v___jp_1285_;
}
}
}
}
else
{
lean_object* v_a_1338_; lean_object* v___x_1340_; uint8_t v_isShared_1341_; uint8_t v_isSharedCheck_1345_; 
v_a_1338_ = lean_ctor_get(v___x_1315_, 0);
v_isSharedCheck_1345_ = !lean_is_exclusive(v___x_1315_);
if (v_isSharedCheck_1345_ == 0)
{
v___x_1340_ = v___x_1315_;
v_isShared_1341_ = v_isSharedCheck_1345_;
goto v_resetjp_1339_;
}
else
{
lean_inc(v_a_1338_);
lean_dec(v___x_1315_);
v___x_1340_ = lean_box(0);
v_isShared_1341_ = v_isSharedCheck_1345_;
goto v_resetjp_1339_;
}
v_resetjp_1339_:
{
lean_object* v___x_1343_; 
if (v_isShared_1341_ == 0)
{
v___x_1343_ = v___x_1340_;
goto v_reusejp_1342_;
}
else
{
lean_object* v_reuseFailAlloc_1344_; 
v_reuseFailAlloc_1344_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1344_, 0, v_a_1338_);
v___x_1343_ = v_reuseFailAlloc_1344_;
goto v_reusejp_1342_;
}
v_reusejp_1342_:
{
return v___x_1343_;
}
}
}
v___jp_1285_:
{
lean_object* v___x_1289_; 
v___x_1289_ = l_Lean_Meta_saveState___redArg(v_a_1281_, v_a_1283_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1301_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
v_isSharedCheck_1301_ = !lean_is_exclusive(v___x_1289_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1292_ = v___x_1289_;
v_isShared_1293_ = v_isSharedCheck_1301_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_a_1290_);
lean_dec(v___x_1289_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1301_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1299_; 
lean_inc(v_snd_1288_);
v___x_1294_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1294_, 0, v_fst_1286_);
lean_ctor_set(v___x_1294_, 1, v_a_1290_);
lean_ctor_set(v___x_1294_, 2, v_fst_1287_);
lean_ctor_set(v___x_1294_, 3, v_snd_1288_);
v___x_1295_ = lean_unsigned_to_nat(1u);
v___x_1296_ = lean_mk_empty_array_with_capacity(v___x_1295_);
v___x_1297_ = lean_array_push(v___x_1296_, v___x_1294_);
if (v_isShared_1293_ == 0)
{
lean_ctor_set(v___x_1292_, 0, v___x_1297_);
v___x_1299_ = v___x_1292_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v___x_1297_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
}
else
{
lean_object* v_a_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1309_; 
lean_dec(v_fst_1287_);
lean_dec_ref(v_fst_1286_);
v_a_1302_ = lean_ctor_get(v___x_1289_, 0);
v_isSharedCheck_1309_ = !lean_is_exclusive(v___x_1289_);
if (v_isSharedCheck_1309_ == 0)
{
v___x_1304_ = v___x_1289_;
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_a_1302_);
lean_dec(v___x_1289_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1307_; 
if (v_isShared_1305_ == 0)
{
v___x_1307_ = v___x_1304_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1308_; 
v_reuseFailAlloc_1308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1308_, 0, v_a_1302_);
v___x_1307_ = v_reuseFailAlloc_1308_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
return v___x_1307_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_cases___boxed(lean_object* v_target_1346_, lean_object* v_md_1347_, lean_object* v_isRecursiveType_1348_, lean_object* v_ctorNames_1349_, lean_object* v_a_1350_, lean_object* v_a_1351_, lean_object* v_a_1352_, lean_object* v_a_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_){
_start:
{
uint8_t v_md_boxed_1357_; uint8_t v_isRecursiveType_boxed_1358_; lean_object* v_res_1359_; 
v_md_boxed_1357_ = lean_unbox(v_md_1347_);
v_isRecursiveType_boxed_1358_ = lean_unbox(v_isRecursiveType_1348_);
v_res_1359_ = lp_aesop_Aesop_RuleTac_cases(v_target_1346_, v_md_boxed_1357_, v_isRecursiveType_boxed_1358_, v_ctorNames_1349_, v_a_1350_, v_a_1351_, v_a_1352_, v_a_1353_, v_a_1354_, v_a_1355_);
lean_dec(v_a_1355_);
lean_dec_ref(v_a_1354_);
lean_dec(v_a_1353_);
lean_dec_ref(v_a_1352_);
lean_dec(v_a_1351_);
return v_res_1359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1(lean_object* v_00_u03b1_1360_, lean_object* v_msg_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_){
_start:
{
lean_object* v___x_1368_; 
v___x_1368_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___redArg(v_msg_1361_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1___boxed(lean_object* v_00_u03b1_1369_, lean_object* v_msg_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_){
_start:
{
lean_object* v_res_1377_; 
v_res_1377_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_cases_spec__1(v_00_u03b1_1369_, v_msg_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_);
lean_dec(v___y_1375_);
lean_dec_ref(v___y_1374_);
lean_dec(v___y_1373_);
lean_dec_ref(v___y_1372_);
lean_dec(v___y_1371_);
return v_res_1377_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Cases(builtin);
}
#ifdef __cplusplus
}
#endif
