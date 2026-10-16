// Lean compiler output
// Module: Aesop.Util.Unfold
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Simp.Main import Lean.Meta.Tactic.Delta import Lean.Meta.WHNF
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
extern lean_object* l_Lean_Meta_Simp_neutralConfig;
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Expr_getAppFn_x27(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_reduceMatcher_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_delta_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isRflTheorem(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_Simp_tryTheorem_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Meta_applySimpResultToLocalDecl(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_applySimpResultToTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static const lean_array_object lp_aesop_Aesop_mkUnfoldSimpContext___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_mkUnfoldSimpContext___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0_value;
static const lean_array_object lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_unfoldManyCore___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_unfoldManyCore___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_unfoldManyCore___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_unfoldManyCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_unfoldManyCore___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_unfoldManyCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_unfoldManyCore___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_unfoldManyCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_unfoldManyCore___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_unfoldManyCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_unfoldManyCore___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_unfoldManyCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_unfoldManyCore___lam__3___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_unfoldManyCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_unfoldManyCore___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_unfoldManyCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_unfoldManyCore___lam__4___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_unfoldManyCore___closed__3 = (const lean_object*)&lp_aesop_Aesop_unfoldManyCore___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__4;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__5;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__6;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__7;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__8;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__9;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyCore___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyCore___closed__10;
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_unfoldMany___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_unfoldMany___closed__0 = (const lean_object*)&lp_aesop_Aesop_unfoldMany___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_unfoldManyTarget___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_unfoldManyTarget___closed__0 = (const lean_object*)&lp_aesop_Aesop_unfoldManyTarget___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_unfoldManyAt___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "aesop_unfold"};
static const lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__0 = (const lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_unfoldManyAt___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(114, 241, 238, 81, 41, 35, 11, 5)}};
static const lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__1 = (const lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__1_value;
static const lean_string_object lp_aesop_Aesop_unfoldManyAt___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "internal error: unexpected result of applySimpResultToLocalDecl"};
static const lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__2 = (const lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_unfoldManyAt___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__3 = (const lean_object*)&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4;
static lean_once_cell_t lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___redArg(lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_a_8_);
lean_dec_ref_known(v___x_7_, 1);
v___x_9_ = l_Lean_Meta_Simp_neutralConfig;
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_mkUnfoldSimpContext___redArg___closed__0));
v___x_11_ = l_Lean_Options_empty;
v___x_12_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_9_, v___x_10_, v_a_8_, v___x_11_, v_a_3_, v_a_4_, v_a_5_);
return v___x_12_;
}
else
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_20_; 
v_a_13_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_20_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_20_ == 0)
{
v___x_15_ = v___x_7_;
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_7_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_18_; 
if (v_isShared_16_ == 0)
{
v___x_18_ = v___x_15_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v_a_13_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___redArg___boxed(lean_object* v_a_21_, lean_object* v_a_22_, lean_object* v_a_23_, lean_object* v_a_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_mkUnfoldSimpContext___redArg(v_a_21_, v_a_22_, v_a_23_);
lean_dec(v_a_23_);
lean_dec_ref(v_a_22_);
lean_dec_ref(v_a_21_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext(lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_aesop_Aesop_mkUnfoldSimpContext___redArg(v_a_26_, v_a_28_, v_a_29_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkUnfoldSimpContext___boxed(lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_aesop_Aesop_mkUnfoldSimpContext(v_a_32_, v_a_33_, v_a_34_, v_a_35_);
lean_dec(v_a_35_);
lean_dec_ref(v_a_34_);
lean_dec(v_a_33_);
lean_dec_ref(v_a_32_);
return v_res_37_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0(lean_object* v_val_38_, lean_object* v_n_39_){
_start:
{
uint8_t v___x_40_; 
v___x_40_ = lean_name_eq(v_n_39_, v_val_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0___boxed(lean_object* v_val_41_, lean_object* v_n_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0(v_val_41_, v_n_42_);
lean_dec(v_n_42_);
lean_dec(v_val_41_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre(lean_object* v_unfold_x3f_49_, lean_object* v_e_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = l_Lean_Expr_getAppFn_x27(v_e_50_);
v___x_64_ = l_Lean_Expr_constName_x3f(v___x_63_);
lean_dec_ref(v___x_63_);
if (lean_obj_tag(v___x_64_) == 1)
{
lean_object* v_val_65_; lean_object* v_a_67_; lean_object* v___x_119_; 
v_val_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc_n(v_val_65_, 2);
lean_dec_ref_known(v___x_64_, 1);
v___x_119_ = lean_apply_1(v_unfold_x3f_49_, v_val_65_);
if (lean_obj_tag(v___x_119_) == 0)
{
lean_dec(v_val_65_);
lean_dec_ref(v_e_50_);
goto v___jp_60_;
}
else
{
lean_object* v_val_120_; 
v_val_120_ = lean_ctor_get(v___x_119_, 0);
lean_inc(v_val_120_);
lean_dec_ref_known(v___x_119_, 1);
if (lean_obj_tag(v_val_120_) == 0)
{
lean_object* v___f_121_; uint8_t v___x_122_; lean_object* v___x_123_; 
lean_inc(v_val_65_);
v___f_121_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___lam__0___boxed), 2, 1);
lean_closure_set(v___f_121_, 0, v_val_65_);
v___x_122_ = 0;
v___x_123_ = l_Lean_Meta_delta_x3f(v_e_50_, v___f_121_, v___x_122_, v_a_57_, v_a_58_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_149_; 
v_a_124_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_149_ == 0)
{
v___x_126_ = v___x_123_;
v_isShared_127_ = v_isSharedCheck_149_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_123_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_149_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
if (lean_obj_tag(v_a_124_) == 1)
{
lean_object* v_val_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_144_; 
v_val_128_ = lean_ctor_get(v_a_124_, 0);
v_isSharedCheck_144_ = !lean_is_exclusive(v_a_124_);
if (v_isSharedCheck_144_ == 0)
{
v___x_130_ = v_a_124_;
v_isShared_131_ = v_isSharedCheck_144_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_val_128_);
lean_dec(v_a_124_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_144_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; uint8_t v___x_136_; lean_object* v___x_137_; lean_object* v___x_139_; 
v___x_132_ = lean_st_ref_take(v_a_51_);
v___x_133_ = lean_array_push(v___x_132_, v_val_65_);
v___x_134_ = lean_st_ref_set(v_a_51_, v___x_133_);
v___x_135_ = lean_box(0);
v___x_136_ = 1;
v___x_137_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_137_, 0, v_val_128_);
lean_ctor_set(v___x_137_, 1, v___x_135_);
lean_ctor_set_uint8(v___x_137_, sizeof(void*)*2, v___x_136_);
if (v_isShared_131_ == 0)
{
lean_ctor_set_tag(v___x_130_, 0);
lean_ctor_set(v___x_130_, 0, v___x_137_);
v___x_139_ = v___x_130_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v___x_137_);
v___x_139_ = v_reuseFailAlloc_143_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
lean_object* v___x_141_; 
if (v_isShared_127_ == 0)
{
lean_ctor_set(v___x_126_, 0, v___x_139_);
v___x_141_ = v___x_126_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_139_);
v___x_141_ = v_reuseFailAlloc_142_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
return v___x_141_;
}
}
}
}
else
{
lean_object* v___x_145_; lean_object* v___x_147_; 
lean_dec(v_a_124_);
lean_dec(v_val_65_);
v___x_145_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0));
if (v_isShared_127_ == 0)
{
lean_ctor_set(v___x_126_, 0, v___x_145_);
v___x_147_ = v___x_126_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
else
{
lean_object* v_a_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_157_; 
lean_dec(v_val_65_);
v_a_150_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_157_ == 0)
{
v___x_152_ = v___x_123_;
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_a_150_);
lean_dec(v___x_123_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_155_; 
if (v_isShared_153_ == 0)
{
v___x_155_ = v___x_152_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v_a_150_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
}
else
{
lean_object* v_val_158_; lean_object* v___x_159_; 
v_val_158_ = lean_ctor_get(v_val_120_, 0);
lean_inc_n(v_val_158_, 2);
lean_dec_ref_known(v_val_120_, 1);
v___x_159_ = l_Lean_Meta_isRflTheorem(v_val_158_, v_a_57_, v_a_58_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; uint8_t v___x_165_; uint8_t v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; uint8_t v___x_169_; lean_object* v_keyedConfig_170_; uint8_t v_trackZetaDelta_171_; lean_object* v_zetaDeltaSet_172_; lean_object* v_lctx_173_; lean_object* v_localInstances_174_; lean_object* v_defEqCtx_x3f_175_; lean_object* v_synthPendingDepth_176_; lean_object* v_customCanUnfoldPredicate_x3f_177_; uint8_t v_univApprox_178_; uint8_t v_inTypeClassResolution_179_; uint8_t v_cacheInferType_180_; uint8_t v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_a_160_);
lean_dec_ref_known(v___x_159_, 1);
v___x_161_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__1));
v___x_162_ = lean_box(0);
lean_inc(v_val_158_);
v___x_163_ = l_Lean_mkConst(v_val_158_, v___x_162_);
v___x_164_ = lean_unsigned_to_nat(1000u);
v___x_165_ = 1;
v___x_166_ = 0;
v___x_167_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_167_, 0, v_val_158_);
lean_ctor_set_uint8(v___x_167_, sizeof(void*)*1, v___x_165_);
lean_ctor_set_uint8(v___x_167_, sizeof(void*)*1 + 1, v___x_166_);
v___x_168_ = lean_alloc_ctor(0, 5, 4);
lean_ctor_set(v___x_168_, 0, v___x_161_);
lean_ctor_set(v___x_168_, 1, v___x_161_);
lean_ctor_set(v___x_168_, 2, v___x_163_);
lean_ctor_set(v___x_168_, 3, v___x_164_);
lean_ctor_set(v___x_168_, 4, v___x_167_);
lean_ctor_set_uint8(v___x_168_, sizeof(void*)*5, v___x_165_);
lean_ctor_set_uint8(v___x_168_, sizeof(void*)*5 + 1, v___x_166_);
v___x_169_ = lean_unbox(v_a_160_);
lean_dec(v_a_160_);
lean_ctor_set_uint8(v___x_168_, sizeof(void*)*5 + 2, v___x_169_);
lean_ctor_set_uint8(v___x_168_, sizeof(void*)*5 + 3, v___x_166_);
v_keyedConfig_170_ = lean_ctor_get(v_a_55_, 0);
v_trackZetaDelta_171_ = lean_ctor_get_uint8(v_a_55_, sizeof(void*)*7);
v_zetaDeltaSet_172_ = lean_ctor_get(v_a_55_, 1);
v_lctx_173_ = lean_ctor_get(v_a_55_, 2);
v_localInstances_174_ = lean_ctor_get(v_a_55_, 3);
v_defEqCtx_x3f_175_ = lean_ctor_get(v_a_55_, 4);
v_synthPendingDepth_176_ = lean_ctor_get(v_a_55_, 5);
v_customCanUnfoldPredicate_x3f_177_ = lean_ctor_get(v_a_55_, 6);
v_univApprox_178_ = lean_ctor_get_uint8(v_a_55_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_179_ = lean_ctor_get_uint8(v_a_55_, sizeof(void*)*7 + 2);
v_cacheInferType_180_ = lean_ctor_get_uint8(v_a_55_, sizeof(void*)*7 + 3);
v___x_181_ = 2;
lean_inc_ref(v_keyedConfig_170_);
v___x_182_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_181_, v_keyedConfig_170_);
lean_inc(v_customCanUnfoldPredicate_x3f_177_);
lean_inc(v_synthPendingDepth_176_);
lean_inc(v_defEqCtx_x3f_175_);
lean_inc_ref(v_localInstances_174_);
lean_inc_ref(v_lctx_173_);
lean_inc(v_zetaDeltaSet_172_);
v___x_183_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_183_, 0, v___x_182_);
lean_ctor_set(v___x_183_, 1, v_zetaDeltaSet_172_);
lean_ctor_set(v___x_183_, 2, v_lctx_173_);
lean_ctor_set(v___x_183_, 3, v_localInstances_174_);
lean_ctor_set(v___x_183_, 4, v_defEqCtx_x3f_175_);
lean_ctor_set(v___x_183_, 5, v_synthPendingDepth_176_);
lean_ctor_set(v___x_183_, 6, v_customCanUnfoldPredicate_x3f_177_);
lean_ctor_set_uint8(v___x_183_, sizeof(void*)*7, v_trackZetaDelta_171_);
lean_ctor_set_uint8(v___x_183_, sizeof(void*)*7 + 1, v_univApprox_178_);
lean_ctor_set_uint8(v___x_183_, sizeof(void*)*7 + 2, v_inTypeClassResolution_179_);
lean_ctor_set_uint8(v___x_183_, sizeof(void*)*7 + 3, v_cacheInferType_180_);
v___x_184_ = l_Lean_Meta_Simp_tryTheorem_x3f(v_e_50_, v___x_168_, v_a_52_, v_a_53_, v_a_54_, v___x_183_, v_a_56_, v_a_57_, v_a_58_);
lean_dec_ref_known(v___x_183_, 7);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_185_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_185_);
lean_dec_ref_known(v___x_184_, 1);
v_a_67_ = v_a_185_;
goto v___jp_66_;
}
else
{
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_186_; 
v_a_186_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_186_);
lean_dec_ref_known(v___x_184_, 1);
v_a_67_ = v_a_186_;
goto v___jp_66_;
}
else
{
lean_object* v_a_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_194_; 
lean_dec(v_val_65_);
v_a_187_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_194_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_194_ == 0)
{
v___x_189_ = v___x_184_;
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_a_187_);
lean_dec(v___x_184_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_192_; 
if (v_isShared_190_ == 0)
{
v___x_192_ = v___x_189_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_a_187_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
}
}
else
{
lean_object* v_a_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_202_; 
lean_dec(v_val_158_);
lean_dec(v_val_65_);
lean_dec_ref(v_e_50_);
v_a_195_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_202_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_202_ == 0)
{
v___x_197_ = v___x_159_;
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_a_195_);
lean_dec(v___x_159_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v___x_200_; 
if (v_isShared_198_ == 0)
{
v___x_200_ = v___x_197_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_a_195_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
}
}
}
v___jp_66_:
{
if (lean_obj_tag(v_a_67_) == 0)
{
lean_dec(v_val_65_);
goto v___jp_60_;
}
else
{
lean_object* v_val_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_118_; 
v_val_68_ = lean_ctor_get(v_a_67_, 0);
v_isSharedCheck_118_ = !lean_is_exclusive(v_a_67_);
if (v_isSharedCheck_118_ == 0)
{
v___x_70_ = v_a_67_;
v_isShared_71_ = v_isSharedCheck_118_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_val_68_);
lean_dec(v_a_67_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_118_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v_expr_75_; lean_object* v_proof_x3f_76_; uint8_t v_cache_77_; lean_object* v___x_78_; 
v___x_72_ = lean_st_ref_take(v_a_51_);
v___x_73_ = lean_array_push(v___x_72_, v_val_65_);
v___x_74_ = lean_st_ref_set(v_a_51_, v___x_73_);
v_expr_75_ = lean_ctor_get(v_val_68_, 0);
v_proof_x3f_76_ = lean_ctor_get(v_val_68_, 1);
v_cache_77_ = lean_ctor_get_uint8(v_val_68_, sizeof(void*)*2);
v___x_78_ = l_Lean_Meta_reduceMatcher_x3f(v_expr_75_, v_a_55_, v_a_56_, v_a_57_, v_a_58_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v_a_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_109_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_109_ == 0)
{
v___x_81_ = v___x_78_;
v_isShared_82_ = v_isSharedCheck_109_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_a_79_);
lean_dec(v___x_78_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_109_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
if (lean_obj_tag(v_a_79_) == 0)
{
lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_100_; 
lean_inc(v_proof_x3f_76_);
lean_del_object(v___x_70_);
v_isSharedCheck_100_ = !lean_is_exclusive(v_val_68_);
if (v_isSharedCheck_100_ == 0)
{
lean_object* v_unused_101_; lean_object* v_unused_102_; 
v_unused_101_ = lean_ctor_get(v_val_68_, 1);
lean_dec(v_unused_101_);
v_unused_102_ = lean_ctor_get(v_val_68_, 0);
lean_dec(v_unused_102_);
v___x_84_ = v_val_68_;
v_isShared_85_ = v_isSharedCheck_100_;
goto v_resetjp_83_;
}
else
{
lean_dec(v_val_68_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_100_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v_val_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_99_; 
v_val_86_ = lean_ctor_get(v_a_79_, 0);
v_isSharedCheck_99_ = !lean_is_exclusive(v_a_79_);
if (v_isSharedCheck_99_ == 0)
{
v___x_88_ = v_a_79_;
v_isShared_89_ = v_isSharedCheck_99_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_val_86_);
lean_dec(v_a_79_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_99_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_91_; 
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 0, v_val_86_);
v___x_91_ = v___x_84_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_val_86_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_proof_x3f_76_);
lean_ctor_set_uint8(v_reuseFailAlloc_98_, sizeof(void*)*2, v_cache_77_);
v___x_91_ = v_reuseFailAlloc_98_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
lean_object* v___x_93_; 
if (v_isShared_89_ == 0)
{
lean_ctor_set(v___x_88_, 0, v___x_91_);
v___x_93_ = v___x_88_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v___x_91_);
v___x_93_ = v_reuseFailAlloc_97_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
lean_object* v___x_95_; 
if (v_isShared_82_ == 0)
{
lean_ctor_set(v___x_81_, 0, v___x_93_);
v___x_95_ = v___x_81_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v___x_93_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
}
}
else
{
lean_object* v___x_104_; 
lean_dec(v_a_79_);
if (v_isShared_71_ == 0)
{
lean_ctor_set_tag(v___x_70_, 0);
v___x_104_ = v___x_70_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v_val_68_);
v___x_104_ = v_reuseFailAlloc_108_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
lean_object* v___x_106_; 
if (v_isShared_82_ == 0)
{
lean_ctor_set(v___x_81_, 0, v___x_104_);
v___x_106_ = v___x_81_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v___x_104_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
}
}
else
{
lean_object* v_a_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_117_; 
lean_del_object(v___x_70_);
lean_dec(v_val_68_);
v_a_110_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_117_ == 0)
{
v___x_112_ = v___x_78_;
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_a_110_);
lean_dec(v___x_78_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_115_; 
if (v_isShared_113_ == 0)
{
v___x_115_ = v___x_112_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_a_110_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_203_; lean_object* v___x_204_; 
lean_dec(v___x_64_);
lean_dec_ref(v_e_50_);
lean_dec_ref(v_unfold_x3f_49_);
v___x_203_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0));
v___x_204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
return v___x_204_;
}
v___jp_60_:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = ((lean_object*)(lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___closed__0));
v___x_62_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre___boxed(lean_object* v_unfold_x3f_205_, lean_object* v_e_206_, lean_object* v_a_207_, lean_object* v_a_208_, lean_object* v_a_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre(v_unfold_x3f_205_, v_e_206_, v_a_207_, v_a_208_, v_a_209_, v_a_210_, v_a_211_, v_a_212_, v_a_213_, v_a_214_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
lean_dec(v_a_212_);
lean_dec_ref(v_a_211_);
lean_dec(v_a_210_);
lean_dec_ref(v_a_209_);
lean_dec(v_a_208_);
lean_dec(v_a_207_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__0(lean_object* v_unfold_x3f_217_, lean_object* v_usedDeclsRef_218_, lean_object* v_x_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre(v_unfold_x3f_217_, v_x_219_, v_usedDeclsRef_218_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__0___boxed(lean_object* v_unfold_x3f_229_, lean_object* v_usedDeclsRef_230_, lean_object* v_x_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_aesop_Aesop_unfoldManyCore___lam__0(v_unfold_x3f_229_, v_usedDeclsRef_230_, v_x_231_, v___y_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
lean_dec(v___y_232_);
lean_dec(v_usedDeclsRef_230_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__1(lean_object* v_e_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v___x_250_; uint8_t v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_250_ = lean_box(0);
v___x_251_ = 1;
v___x_252_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_252_, 0, v_e_241_);
lean_ctor_set(v___x_252_, 1, v___x_250_);
lean_ctor_set_uint8(v___x_252_, sizeof(void*)*2, v___x_251_);
v___x_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
v___x_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__1___boxed(lean_object* v_e_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_aesop_Aesop_unfoldManyCore___lam__1(v_e_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
lean_dec(v___y_256_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__2(lean_object* v_x_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_276_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___lam__2___closed__0));
v___x_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__2___boxed(lean_object* v_x_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_aesop_Aesop_unfoldManyCore___lam__2(v_x_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v_x_278_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__3(lean_object* v_e_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_297_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_297_, 0, v_e_288_);
v___x_298_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_298_, 0, v___x_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__3___boxed(lean_object* v_e_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_aesop_Aesop_unfoldManyCore___lam__3(v_e_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__4(lean_object* v_x_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_box(0);
v___x_319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_319_, 0, v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___lam__4___boxed(lean_object* v_x_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_aesop_Aesop_unfoldManyCore___lam__4(v_x_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec_ref(v_x_320_);
return v_res_329_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__4(void){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_334_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__5(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__4, &lp_aesop_Aesop_unfoldManyCore___closed__4_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__4);
v___x_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
return v___x_336_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__6(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_337_ = lean_unsigned_to_nat(0u);
v___x_338_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__5, &lp_aesop_Aesop_unfoldManyCore___closed__5_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__5);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v___x_337_);
return v___x_339_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__7(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_340_ = lean_unsigned_to_nat(32u);
v___x_341_ = lean_mk_empty_array_with_capacity(v___x_340_);
v___x_342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
return v___x_342_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__8(void){
_start:
{
size_t v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_343_ = ((size_t)5ULL);
v___x_344_ = lean_unsigned_to_nat(0u);
v___x_345_ = lean_unsigned_to_nat(32u);
v___x_346_ = lean_mk_empty_array_with_capacity(v___x_345_);
v___x_347_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__7, &lp_aesop_Aesop_unfoldManyCore___closed__7_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__7);
v___x_348_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
lean_ctor_set(v___x_348_, 2, v___x_344_);
lean_ctor_set(v___x_348_, 3, v___x_344_);
lean_ctor_set_usize(v___x_348_, 4, v___x_343_);
return v___x_348_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__9(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__8, &lp_aesop_Aesop_unfoldManyCore___closed__8_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__8);
v___x_350_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__5, &lp_aesop_Aesop_unfoldManyCore___closed__5_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__5);
v___x_351_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
lean_ctor_set(v___x_351_, 2, v___x_350_);
lean_ctor_set(v___x_351_, 3, v___x_349_);
return v___x_351_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyCore___closed__10(void){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_352_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__9, &lp_aesop_Aesop_unfoldManyCore___closed__9_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__9);
v___x_353_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__6, &lp_aesop_Aesop_unfoldManyCore___closed__6_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__6);
v___x_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
lean_ctor_set(v___x_354_, 1, v___x_352_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore(lean_object* v_ctx_355_, lean_object* v_unfold_x3f_356_, lean_object* v_e_357_, lean_object* v_usedDeclsRef_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_){
_start:
{
lean_object* v___f_364_; lean_object* v___f_365_; lean_object* v___f_366_; lean_object* v___f_367_; lean_object* v___f_368_; lean_object* v___x_369_; uint8_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
lean_inc(v_usedDeclsRef_358_);
v___f_364_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldManyCore___lam__0___boxed), 11, 2);
lean_closure_set(v___f_364_, 0, v_unfold_x3f_356_);
lean_closure_set(v___f_364_, 1, v_usedDeclsRef_358_);
v___f_365_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__0));
v___f_366_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__1));
v___f_367_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__2));
v___f_368_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__3));
v___x_369_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__10, &lp_aesop_Aesop_unfoldManyCore___closed__10_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__10);
v___x_370_ = 1;
v___x_371_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_371_, 0, v___f_364_);
lean_ctor_set(v___x_371_, 1, v___f_365_);
lean_ctor_set(v___x_371_, 2, v___f_366_);
lean_ctor_set(v___x_371_, 3, v___f_367_);
lean_ctor_set(v___x_371_, 4, v___f_368_);
lean_ctor_set_uint8(v___x_371_, sizeof(void*)*5, v___x_370_);
v___x_372_ = l_Lean_Meta_Simp_main(v_e_357_, v_ctx_355_, v___x_369_, v___x_371_, v_a_359_, v_a_360_, v_a_361_, v_a_362_);
if (lean_obj_tag(v___x_372_) == 0)
{
lean_object* v_a_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_381_; 
v_a_373_ = lean_ctor_get(v___x_372_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_381_ == 0)
{
v___x_375_ = v___x_372_;
v_isShared_376_ = v_isSharedCheck_381_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_a_373_);
lean_dec(v___x_372_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_381_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
lean_object* v_fst_377_; lean_object* v___x_379_; 
v_fst_377_ = lean_ctor_get(v_a_373_, 0);
lean_inc(v_fst_377_);
lean_dec(v_a_373_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 0, v_fst_377_);
v___x_379_ = v___x_375_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_fst_377_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
else
{
lean_object* v_a_382_; lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_389_; 
v_a_382_ = lean_ctor_get(v___x_372_, 0);
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_389_ == 0)
{
v___x_384_ = v___x_372_;
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
else
{
lean_inc(v_a_382_);
lean_dec(v___x_372_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v___x_387_; 
if (v_isShared_385_ == 0)
{
v___x_387_ = v___x_384_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v_a_382_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyCore___boxed(lean_object* v_ctx_390_, lean_object* v_unfold_x3f_391_, lean_object* v_e_392_, lean_object* v_usedDeclsRef_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_, lean_object* v_a_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_aesop_Aesop_unfoldManyCore(v_ctx_390_, v_unfold_x3f_391_, v_e_392_, v_usedDeclsRef_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
lean_dec(v_a_397_);
lean_dec_ref(v_a_396_);
lean_dec(v_a_395_);
lean_dec_ref(v_a_394_);
lean_dec(v_usedDeclsRef_393_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(lean_object* v_e_400_, lean_object* v___y_401_){
_start:
{
uint8_t v___x_403_; 
v___x_403_ = l_Lean_Expr_hasMVar(v_e_400_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; 
v___x_404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_404_, 0, v_e_400_);
return v___x_404_;
}
else
{
lean_object* v___x_405_; lean_object* v_mctx_406_; lean_object* v___x_407_; lean_object* v_fst_408_; lean_object* v_snd_409_; lean_object* v___x_410_; lean_object* v_cache_411_; lean_object* v_zetaDeltaFVarIds_412_; lean_object* v_postponed_413_; lean_object* v_diag_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_423_; 
v___x_405_ = lean_st_ref_get(v___y_401_);
v_mctx_406_ = lean_ctor_get(v___x_405_, 0);
lean_inc_ref(v_mctx_406_);
lean_dec(v___x_405_);
v___x_407_ = l_Lean_instantiateMVarsCore(v_mctx_406_, v_e_400_);
v_fst_408_ = lean_ctor_get(v___x_407_, 0);
lean_inc(v_fst_408_);
v_snd_409_ = lean_ctor_get(v___x_407_, 1);
lean_inc(v_snd_409_);
lean_dec_ref(v___x_407_);
v___x_410_ = lean_st_ref_take(v___y_401_);
v_cache_411_ = lean_ctor_get(v___x_410_, 1);
v_zetaDeltaFVarIds_412_ = lean_ctor_get(v___x_410_, 2);
v_postponed_413_ = lean_ctor_get(v___x_410_, 3);
v_diag_414_ = lean_ctor_get(v___x_410_, 4);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_423_ == 0)
{
lean_object* v_unused_424_; 
v_unused_424_ = lean_ctor_get(v___x_410_, 0);
lean_dec(v_unused_424_);
v___x_416_ = v___x_410_;
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_diag_414_);
lean_inc(v_postponed_413_);
lean_inc(v_zetaDeltaFVarIds_412_);
lean_inc(v_cache_411_);
lean_dec(v___x_410_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_419_; 
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 0, v_snd_409_);
v___x_419_ = v___x_416_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v_snd_409_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v_cache_411_);
lean_ctor_set(v_reuseFailAlloc_422_, 2, v_zetaDeltaFVarIds_412_);
lean_ctor_set(v_reuseFailAlloc_422_, 3, v_postponed_413_);
lean_ctor_set(v_reuseFailAlloc_422_, 4, v_diag_414_);
v___x_419_ = v_reuseFailAlloc_422_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
lean_object* v___x_420_; lean_object* v___x_421_; 
v___x_420_ = lean_st_ref_set(v___y_401_, v___x_419_);
v___x_421_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_421_, 0, v_fst_408_);
return v___x_421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg___boxed(lean_object* v_e_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_e_425_, v___y_426_);
lean_dec(v___y_426_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0(lean_object* v_e_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_e_429_, v___y_431_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___boxed(lean_object* v_e_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0(v_e_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___lam__4(lean_object* v_unfold_x3f_443_, lean_object* v_val_444_, lean_object* v_x_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_aesop___private_Aesop_Util_Unfold_0__Aesop_unfoldManyCore_pre(v_unfold_x3f_443_, v_x_445_, v_val_444_, v___y_446_, v___y_447_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___lam__4___boxed(lean_object* v_unfold_x3f_455_, lean_object* v_val_456_, lean_object* v_x_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_aesop_Aesop_unfoldMany___lam__4(v_unfold_x3f_455_, v_val_456_, v_x_457_, v___y_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_, v___y_463_, v___y_464_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
lean_dec(v___y_460_);
lean_dec_ref(v___y_459_);
lean_dec(v___y_458_);
lean_dec(v_val_456_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany(lean_object* v_unfold_x3f_469_, lean_object* v_e_470_, lean_object* v_a_471_, lean_object* v_a_472_, lean_object* v_a_473_, lean_object* v_a_474_){
_start:
{
lean_object* v___x_476_; lean_object* v_a_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_539_; 
v___x_476_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_e_470_, v_a_472_);
v_a_477_ = lean_ctor_get(v___x_476_, 0);
v_isSharedCheck_539_ = !lean_is_exclusive(v___x_476_);
if (v_isSharedCheck_539_ == 0)
{
v___x_479_ = v___x_476_;
v_isShared_480_ = v_isSharedCheck_539_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_a_477_);
lean_dec(v___x_476_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_539_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; 
v___x_481_ = lp_aesop_Aesop_mkUnfoldSimpContext___redArg(v_a_471_, v_a_473_, v_a_474_);
if (lean_obj_tag(v___x_481_) == 0)
{
lean_object* v_a_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___f_485_; lean_object* v___f_486_; lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___f_489_; lean_object* v___x_490_; uint8_t v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_a_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_a_482_);
lean_dec_ref_known(v___x_481_, 1);
v___x_483_ = ((lean_object*)(lp_aesop_Aesop_unfoldMany___closed__0));
v___x_484_ = lean_st_mk_ref(v___x_483_);
v___f_485_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__0));
v___f_486_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__1));
v___f_487_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__2));
v___f_488_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__3));
lean_inc(v___x_484_);
v___f_489_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldMany___lam__4___boxed), 11, 2);
lean_closure_set(v___f_489_, 0, v_unfold_x3f_469_);
lean_closure_set(v___f_489_, 1, v___x_484_);
v___x_490_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__10, &lp_aesop_Aesop_unfoldManyCore___closed__10_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__10);
v___x_491_ = 1;
v___x_492_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_492_, 0, v___f_489_);
lean_ctor_set(v___x_492_, 1, v___f_485_);
lean_ctor_set(v___x_492_, 2, v___f_486_);
lean_ctor_set(v___x_492_, 3, v___f_487_);
lean_ctor_set(v___x_492_, 4, v___f_488_);
lean_ctor_set_uint8(v___x_492_, sizeof(void*)*5, v___x_491_);
lean_inc(v_a_477_);
v___x_493_ = l_Lean_Meta_Simp_main(v_a_477_, v_a_482_, v___x_490_, v___x_492_, v_a_471_, v_a_472_, v_a_473_, v_a_474_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_495_; lean_object* v_fst_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_521_; 
v_a_494_ = lean_ctor_get(v___x_493_, 0);
lean_inc(v_a_494_);
lean_dec_ref_known(v___x_493_, 1);
v___x_495_ = lean_st_ref_get(v___x_484_);
lean_dec(v___x_484_);
v_fst_496_ = lean_ctor_get(v_a_494_, 0);
v_isSharedCheck_521_ = !lean_is_exclusive(v_a_494_);
if (v_isSharedCheck_521_ == 0)
{
lean_object* v_unused_522_; 
v_unused_522_ = lean_ctor_get(v_a_494_, 1);
lean_dec(v_unused_522_);
v___x_498_ = v_a_494_;
v_isShared_499_ = v_isSharedCheck_521_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_fst_496_);
lean_dec(v_a_494_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_521_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v_expr_500_; lean_object* v___x_501_; lean_object* v_a_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_520_; 
v_expr_500_ = lean_ctor_get(v_fst_496_, 0);
lean_inc_ref_n(v_expr_500_, 2);
lean_dec(v_fst_496_);
v___x_501_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_expr_500_, v_a_472_);
v_a_502_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_520_ == 0)
{
v___x_504_ = v___x_501_;
v_isShared_505_ = v_isSharedCheck_520_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_a_502_);
lean_dec(v___x_501_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_520_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
uint8_t v___x_506_; 
v___x_506_ = lean_expr_eqv(v_a_502_, v_a_477_);
lean_dec(v_a_477_);
lean_dec(v_a_502_);
if (v___x_506_ == 0)
{
lean_object* v___x_508_; 
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 1, v___x_495_);
lean_ctor_set(v___x_498_, 0, v_expr_500_);
v___x_508_ = v___x_498_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_expr_500_);
lean_ctor_set(v_reuseFailAlloc_515_, 1, v___x_495_);
v___x_508_ = v_reuseFailAlloc_515_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
lean_object* v___x_510_; 
if (v_isShared_480_ == 0)
{
lean_ctor_set_tag(v___x_479_, 1);
lean_ctor_set(v___x_479_, 0, v___x_508_);
v___x_510_ = v___x_479_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_508_);
v___x_510_ = v_reuseFailAlloc_514_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
lean_object* v___x_512_; 
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 0, v___x_510_);
v___x_512_ = v___x_504_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v___x_510_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
else
{
lean_object* v___x_516_; lean_object* v___x_518_; 
lean_dec_ref(v_expr_500_);
lean_del_object(v___x_498_);
lean_dec(v___x_495_);
lean_del_object(v___x_479_);
v___x_516_ = lean_box(0);
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 0, v___x_516_);
v___x_518_ = v___x_504_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v___x_516_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
}
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
lean_dec(v___x_484_);
lean_del_object(v___x_479_);
lean_dec(v_a_477_);
v_a_523_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___x_493_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_493_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
else
{
lean_object* v_a_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_538_; 
lean_del_object(v___x_479_);
lean_dec(v_a_477_);
lean_dec_ref(v_unfold_x3f_469_);
v_a_531_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_538_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_538_ == 0)
{
v___x_533_ = v___x_481_;
v_isShared_534_ = v_isSharedCheck_538_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_a_531_);
lean_dec(v___x_481_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_538_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_536_; 
if (v_isShared_534_ == 0)
{
v___x_536_ = v___x_533_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v_a_531_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldMany___boxed(lean_object* v_unfold_x3f_540_, lean_object* v_e_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_, lean_object* v_a_545_, lean_object* v_a_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_aesop_Aesop_unfoldMany(v_unfold_x3f_540_, v_e_541_, v_a_542_, v_a_543_, v_a_544_, v_a_545_);
lean_dec(v_a_545_);
lean_dec_ref(v_a_544_);
lean_dec(v_a_543_);
lean_dec_ref(v_a_542_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTarget(lean_object* v_unfold_x3f_550_, lean_object* v_goal_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_){
_start:
{
lean_object* v___x_557_; 
lean_inc(v_goal_551_);
v___x_557_ = l_Lean_MVarId_getType(v_goal_551_, v_a_552_, v_a_553_, v_a_554_, v_a_555_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_object* v_a_558_; lean_object* v___x_559_; lean_object* v_a_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_634_; 
v_a_558_ = lean_ctor_get(v___x_557_, 0);
lean_inc(v_a_558_);
lean_dec_ref_known(v___x_557_, 1);
v___x_559_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_a_558_, v_a_553_);
v_a_560_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_634_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_634_ == 0)
{
v___x_562_ = v___x_559_;
v_isShared_563_ = v_isSharedCheck_634_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_a_560_);
lean_dec(v___x_559_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_634_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v___x_564_; 
v___x_564_ = lp_aesop_Aesop_mkUnfoldSimpContext___redArg(v_a_552_, v_a_554_, v_a_555_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v_a_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___f_568_; lean_object* v___f_569_; lean_object* v___f_570_; lean_object* v___f_571_; lean_object* v___f_572_; lean_object* v___x_573_; uint8_t v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v_a_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_565_);
lean_dec_ref_known(v___x_564_, 1);
v___x_566_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyTarget___closed__0));
v___x_567_ = lean_st_mk_ref(v___x_566_);
v___f_568_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__0));
v___f_569_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__1));
v___f_570_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__2));
v___f_571_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__3));
lean_inc(v___x_567_);
v___f_572_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldMany___lam__4___boxed), 11, 2);
lean_closure_set(v___f_572_, 0, v_unfold_x3f_550_);
lean_closure_set(v___f_572_, 1, v___x_567_);
v___x_573_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__10, &lp_aesop_Aesop_unfoldManyCore___closed__10_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__10);
v___x_574_ = 1;
v___x_575_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_575_, 0, v___f_572_);
lean_ctor_set(v___x_575_, 1, v___f_568_);
lean_ctor_set(v___x_575_, 2, v___f_569_);
lean_ctor_set(v___x_575_, 3, v___f_570_);
lean_ctor_set(v___x_575_, 4, v___f_571_);
lean_ctor_set_uint8(v___x_575_, sizeof(void*)*5, v___x_574_);
lean_inc(v_a_560_);
v___x_576_ = l_Lean_Meta_Simp_main(v_a_560_, v_a_565_, v___x_573_, v___x_575_, v_a_552_, v_a_553_, v_a_554_, v_a_555_);
if (lean_obj_tag(v___x_576_) == 0)
{
lean_object* v_a_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_617_; 
v_a_577_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_617_ == 0)
{
v___x_579_ = v___x_576_;
v_isShared_580_ = v_isSharedCheck_617_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_a_577_);
lean_dec(v___x_576_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_617_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v___x_581_; lean_object* v_fst_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_615_; 
v___x_581_ = lean_st_ref_get(v___x_567_);
lean_dec(v___x_567_);
v_fst_582_ = lean_ctor_get(v_a_577_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v_a_577_);
if (v_isSharedCheck_615_ == 0)
{
lean_object* v_unused_616_; 
v_unused_616_ = lean_ctor_get(v_a_577_, 1);
lean_dec(v_unused_616_);
v___x_584_ = v_a_577_;
v_isShared_585_ = v_isSharedCheck_615_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_fst_582_);
lean_dec(v_a_577_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_615_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v_expr_586_; uint8_t v___x_587_; 
v_expr_586_ = lean_ctor_get(v_fst_582_, 0);
v___x_587_ = lean_expr_eqv(v_expr_586_, v_a_560_);
if (v___x_587_ == 0)
{
lean_object* v___x_588_; 
lean_del_object(v___x_579_);
v___x_588_ = l_Lean_Meta_applySimpResultToTarget(v_goal_551_, v_a_560_, v_fst_582_, v_a_552_, v_a_553_, v_a_554_, v_a_555_);
lean_dec(v_a_560_);
if (lean_obj_tag(v___x_588_) == 0)
{
lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_602_; 
v_a_589_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_602_ == 0)
{
v___x_591_ = v___x_588_;
v_isShared_592_ = v_isSharedCheck_602_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_588_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_602_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_594_; 
if (v_isShared_585_ == 0)
{
lean_ctor_set(v___x_584_, 1, v___x_581_);
lean_ctor_set(v___x_584_, 0, v_a_589_);
v___x_594_ = v___x_584_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_a_589_);
lean_ctor_set(v_reuseFailAlloc_601_, 1, v___x_581_);
v___x_594_ = v_reuseFailAlloc_601_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
lean_object* v___x_596_; 
if (v_isShared_563_ == 0)
{
lean_ctor_set_tag(v___x_562_, 1);
lean_ctor_set(v___x_562_, 0, v___x_594_);
v___x_596_ = v___x_562_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v___x_594_);
v___x_596_ = v_reuseFailAlloc_600_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
lean_object* v___x_598_; 
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_596_);
v___x_598_ = v___x_591_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v___x_596_);
v___x_598_ = v_reuseFailAlloc_599_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
return v___x_598_;
}
}
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_del_object(v___x_584_);
lean_dec(v___x_581_);
lean_del_object(v___x_562_);
v_a_603_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_588_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_588_);
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
else
{
lean_object* v___x_611_; lean_object* v___x_613_; 
lean_del_object(v___x_584_);
lean_dec(v_fst_582_);
lean_dec(v___x_581_);
lean_del_object(v___x_562_);
lean_dec(v_a_560_);
lean_dec(v_goal_551_);
v___x_611_ = lean_box(0);
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 0, v___x_611_);
v___x_613_ = v___x_579_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v___x_611_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
}
}
}
}
}
else
{
lean_object* v_a_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
lean_dec(v___x_567_);
lean_del_object(v___x_562_);
lean_dec(v_a_560_);
lean_dec(v_goal_551_);
v_a_618_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_625_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_625_ == 0)
{
v___x_620_ = v___x_576_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_a_618_);
lean_dec(v___x_576_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_623_; 
if (v_isShared_621_ == 0)
{
v___x_623_ = v___x_620_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_a_618_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
}
else
{
lean_object* v_a_626_; lean_object* v___x_628_; uint8_t v_isShared_629_; uint8_t v_isSharedCheck_633_; 
lean_del_object(v___x_562_);
lean_dec(v_a_560_);
lean_dec(v_goal_551_);
lean_dec_ref(v_unfold_x3f_550_);
v_a_626_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_633_ == 0)
{
v___x_628_ = v___x_564_;
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
else
{
lean_inc(v_a_626_);
lean_dec(v___x_564_);
v___x_628_ = lean_box(0);
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
v_resetjp_627_:
{
lean_object* v___x_631_; 
if (v_isShared_629_ == 0)
{
v___x_631_ = v___x_628_;
goto v_reusejp_630_;
}
else
{
lean_object* v_reuseFailAlloc_632_; 
v_reuseFailAlloc_632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_632_, 0, v_a_626_);
v___x_631_ = v_reuseFailAlloc_632_;
goto v_reusejp_630_;
}
v_reusejp_630_:
{
return v___x_631_;
}
}
}
}
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_642_; 
lean_dec(v_goal_551_);
lean_dec_ref(v_unfold_x3f_550_);
v_a_635_ = lean_ctor_get(v___x_557_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v___x_557_);
if (v_isSharedCheck_642_ == 0)
{
v___x_637_ = v___x_557_;
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_557_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_640_; 
if (v_isShared_638_ == 0)
{
v___x_640_ = v___x_637_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_641_; 
v_reuseFailAlloc_641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_641_, 0, v_a_635_);
v___x_640_ = v_reuseFailAlloc_641_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
return v___x_640_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTarget___boxed(lean_object* v_unfold_x3f_643_, lean_object* v_goal_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_aesop_Aesop_unfoldManyTarget(v_unfold_x3f_643_, v_goal_644_, v_a_645_, v_a_646_, v_a_647_, v_a_648_);
lean_dec(v_a_648_);
lean_dec_ref(v_a_647_);
lean_dec(v_a_646_);
lean_dec_ref(v_a_645_);
return v_res_650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(lean_object* v_mvarId_651_, lean_object* v_x_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_651_, v_x_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_);
if (lean_obj_tag(v___x_658_) == 0)
{
lean_object* v_a_659_; lean_object* v___x_661_; uint8_t v_isShared_662_; uint8_t v_isSharedCheck_666_; 
v_a_659_ = lean_ctor_get(v___x_658_, 0);
v_isSharedCheck_666_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_666_ == 0)
{
v___x_661_ = v___x_658_;
v_isShared_662_ = v_isSharedCheck_666_;
goto v_resetjp_660_;
}
else
{
lean_inc(v_a_659_);
lean_dec(v___x_658_);
v___x_661_ = lean_box(0);
v_isShared_662_ = v_isSharedCheck_666_;
goto v_resetjp_660_;
}
v_resetjp_660_:
{
lean_object* v___x_664_; 
if (v_isShared_662_ == 0)
{
v___x_664_ = v___x_661_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v_a_659_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
else
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_674_; 
v_a_667_ = lean_ctor_get(v___x_658_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_674_ == 0)
{
v___x_669_ = v___x_658_;
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___x_658_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___x_672_; 
if (v_isShared_670_ == 0)
{
v___x_672_ = v___x_669_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_a_667_);
v___x_672_ = v_reuseFailAlloc_673_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
return v___x_672_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg___boxed(lean_object* v_mvarId_675_, lean_object* v_x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_){
_start:
{
lean_object* v_res_682_; 
v_res_682_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(v_mvarId_675_, v_x_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0(lean_object* v_00_u03b1_683_, lean_object* v_mvarId_684_, lean_object* v_x_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(v_mvarId_684_, v_x_685_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___boxed(lean_object* v_00_u03b1_692_, lean_object* v_mvarId_693_, lean_object* v_x_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0(v_00_u03b1_692_, v_mvarId_693_, v_x_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
return v_res_700_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4(void){
_start:
{
lean_object* v___x_707_; lean_object* v___x_708_; 
v___x_707_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyAt___lam__5___closed__3));
v___x_708_ = l_Lean_MessageData_ofFormat(v___x_707_);
return v___x_708_;
}
}
static lean_object* _init_lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5(void){
_start:
{
lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_709_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4, &lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4_once, _init_lp_aesop_Aesop_unfoldManyAt___lam__5___closed__4);
v___x_710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5(lean_object* v_fvarId_711_, lean_object* v_unfold_x3f_712_, lean_object* v___f_713_, lean_object* v___f_714_, lean_object* v___f_715_, lean_object* v___f_716_, lean_object* v_goal_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_723_; 
lean_inc(v_fvarId_711_);
v___x_723_ = l_Lean_FVarId_getType___redArg(v_fvarId_711_, v___y_718_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_723_) == 0)
{
lean_object* v_a_724_; lean_object* v___x_725_; lean_object* v_a_726_; lean_object* v___x_727_; 
v_a_724_ = lean_ctor_get(v___x_723_, 0);
lean_inc(v_a_724_);
lean_dec_ref_known(v___x_723_, 1);
v___x_725_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_unfoldMany_spec__0___redArg(v_a_724_, v___y_719_);
v_a_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc(v_a_726_);
lean_dec_ref(v___x_725_);
v___x_727_ = lp_aesop_Aesop_mkUnfoldSimpContext___redArg(v___y_718_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_object* v_a_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___f_731_; lean_object* v___x_732_; uint8_t v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v_a_728_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_a_728_);
lean_dec_ref_known(v___x_727_, 1);
v___x_729_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyTarget___closed__0));
v___x_730_ = lean_st_mk_ref(v___x_729_);
lean_inc(v___x_730_);
v___f_731_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldMany___lam__4___boxed), 11, 2);
lean_closure_set(v___f_731_, 0, v_unfold_x3f_712_);
lean_closure_set(v___f_731_, 1, v___x_730_);
v___x_732_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyCore___closed__10, &lp_aesop_Aesop_unfoldManyCore___closed__10_once, _init_lp_aesop_Aesop_unfoldManyCore___closed__10);
v___x_733_ = 1;
v___x_734_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_734_, 0, v___f_731_);
lean_ctor_set(v___x_734_, 1, v___f_713_);
lean_ctor_set(v___x_734_, 2, v___f_714_);
lean_ctor_set(v___x_734_, 3, v___f_715_);
lean_ctor_set(v___x_734_, 4, v___f_716_);
lean_ctor_set_uint8(v___x_734_, sizeof(void*)*5, v___x_733_);
lean_inc(v_a_726_);
v___x_735_ = l_Lean_Meta_Simp_main(v_a_726_, v_a_728_, v___x_732_, v___x_734_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_735_) == 0)
{
lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_785_; 
v_a_736_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_785_ == 0)
{
v___x_738_ = v___x_735_;
v_isShared_739_ = v_isSharedCheck_785_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_735_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_785_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_740_; lean_object* v_fst_741_; lean_object* v_expr_742_; uint8_t v___x_743_; 
v___x_740_ = lean_st_ref_get(v___x_730_);
lean_dec(v___x_730_);
v_fst_741_ = lean_ctor_get(v_a_736_, 0);
lean_inc(v_fst_741_);
lean_dec(v_a_736_);
v_expr_742_ = lean_ctor_get(v_fst_741_, 0);
v___x_743_ = lean_expr_eqv(v_expr_742_, v_a_726_);
lean_dec(v_a_726_);
if (v___x_743_ == 0)
{
lean_object* v___x_744_; 
lean_del_object(v___x_738_);
lean_inc(v_goal_717_);
v___x_744_ = l_Lean_Meta_applySimpResultToLocalDecl(v_goal_717_, v_fvarId_711_, v_fst_741_, v___x_743_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_744_) == 0)
{
lean_object* v_a_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_772_; 
v_a_745_ = lean_ctor_get(v___x_744_, 0);
v_isSharedCheck_772_ = !lean_is_exclusive(v___x_744_);
if (v_isSharedCheck_772_ == 0)
{
v___x_747_ = v___x_744_;
v_isShared_748_ = v_isSharedCheck_772_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_a_745_);
lean_dec(v___x_744_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_772_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
if (lean_obj_tag(v_a_745_) == 1)
{
lean_object* v_val_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_768_; 
lean_dec(v_goal_717_);
v_val_749_ = lean_ctor_get(v_a_745_, 0);
v_isSharedCheck_768_ = !lean_is_exclusive(v_a_745_);
if (v_isSharedCheck_768_ == 0)
{
v___x_751_ = v_a_745_;
v_isShared_752_ = v_isSharedCheck_768_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_val_749_);
lean_dec(v_a_745_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_768_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
lean_object* v_snd_753_; lean_object* v___x_755_; uint8_t v_isShared_756_; uint8_t v_isSharedCheck_766_; 
v_snd_753_ = lean_ctor_get(v_val_749_, 1);
v_isSharedCheck_766_ = !lean_is_exclusive(v_val_749_);
if (v_isSharedCheck_766_ == 0)
{
lean_object* v_unused_767_; 
v_unused_767_ = lean_ctor_get(v_val_749_, 0);
lean_dec(v_unused_767_);
v___x_755_ = v_val_749_;
v_isShared_756_ = v_isSharedCheck_766_;
goto v_resetjp_754_;
}
else
{
lean_inc(v_snd_753_);
lean_dec(v_val_749_);
v___x_755_ = lean_box(0);
v_isShared_756_ = v_isSharedCheck_766_;
goto v_resetjp_754_;
}
v_resetjp_754_:
{
lean_object* v___x_758_; 
if (v_isShared_756_ == 0)
{
lean_ctor_set(v___x_755_, 1, v___x_740_);
lean_ctor_set(v___x_755_, 0, v_snd_753_);
v___x_758_ = v___x_755_;
goto v_reusejp_757_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_snd_753_);
lean_ctor_set(v_reuseFailAlloc_765_, 1, v___x_740_);
v___x_758_ = v_reuseFailAlloc_765_;
goto v_reusejp_757_;
}
v_reusejp_757_:
{
lean_object* v___x_760_; 
if (v_isShared_752_ == 0)
{
lean_ctor_set(v___x_751_, 0, v___x_758_);
v___x_760_ = v___x_751_;
goto v_reusejp_759_;
}
else
{
lean_object* v_reuseFailAlloc_764_; 
v_reuseFailAlloc_764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_764_, 0, v___x_758_);
v___x_760_ = v_reuseFailAlloc_764_;
goto v_reusejp_759_;
}
v_reusejp_759_:
{
lean_object* v___x_762_; 
if (v_isShared_748_ == 0)
{
lean_ctor_set(v___x_747_, 0, v___x_760_);
v___x_762_ = v___x_747_;
goto v_reusejp_761_;
}
else
{
lean_object* v_reuseFailAlloc_763_; 
v_reuseFailAlloc_763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_763_, 0, v___x_760_);
v___x_762_ = v_reuseFailAlloc_763_;
goto v_reusejp_761_;
}
v_reusejp_761_:
{
return v___x_762_;
}
}
}
}
}
}
else
{
lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
lean_del_object(v___x_747_);
lean_dec(v_a_745_);
lean_dec(v___x_740_);
v___x_769_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyAt___lam__5___closed__1));
v___x_770_ = lean_obj_once(&lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5, &lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5_once, _init_lp_aesop_Aesop_unfoldManyAt___lam__5___closed__5);
v___x_771_ = l_Lean_Meta_throwTacticEx___redArg(v___x_769_, v_goal_717_, v___x_770_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
return v___x_771_;
}
}
}
else
{
lean_object* v_a_773_; lean_object* v___x_775_; uint8_t v_isShared_776_; uint8_t v_isSharedCheck_780_; 
lean_dec(v___x_740_);
lean_dec(v_goal_717_);
v_a_773_ = lean_ctor_get(v___x_744_, 0);
v_isSharedCheck_780_ = !lean_is_exclusive(v___x_744_);
if (v_isSharedCheck_780_ == 0)
{
v___x_775_ = v___x_744_;
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
else
{
lean_inc(v_a_773_);
lean_dec(v___x_744_);
v___x_775_ = lean_box(0);
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
v_resetjp_774_:
{
lean_object* v___x_778_; 
if (v_isShared_776_ == 0)
{
v___x_778_ = v___x_775_;
goto v_reusejp_777_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v_a_773_);
v___x_778_ = v_reuseFailAlloc_779_;
goto v_reusejp_777_;
}
v_reusejp_777_:
{
return v___x_778_;
}
}
}
}
else
{
lean_object* v___x_781_; lean_object* v___x_783_; 
lean_dec(v_fst_741_);
lean_dec(v___x_740_);
lean_dec(v_goal_717_);
lean_dec(v_fvarId_711_);
v___x_781_ = lean_box(0);
if (v_isShared_739_ == 0)
{
lean_ctor_set(v___x_738_, 0, v___x_781_);
v___x_783_ = v___x_738_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v___x_781_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
else
{
lean_object* v_a_786_; lean_object* v___x_788_; uint8_t v_isShared_789_; uint8_t v_isSharedCheck_793_; 
lean_dec(v___x_730_);
lean_dec(v_a_726_);
lean_dec(v_goal_717_);
lean_dec(v_fvarId_711_);
v_a_786_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_793_ == 0)
{
v___x_788_ = v___x_735_;
v_isShared_789_ = v_isSharedCheck_793_;
goto v_resetjp_787_;
}
else
{
lean_inc(v_a_786_);
lean_dec(v___x_735_);
v___x_788_ = lean_box(0);
v_isShared_789_ = v_isSharedCheck_793_;
goto v_resetjp_787_;
}
v_resetjp_787_:
{
lean_object* v___x_791_; 
if (v_isShared_789_ == 0)
{
v___x_791_ = v___x_788_;
goto v_reusejp_790_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v_a_786_);
v___x_791_ = v_reuseFailAlloc_792_;
goto v_reusejp_790_;
}
v_reusejp_790_:
{
return v___x_791_;
}
}
}
}
else
{
lean_object* v_a_794_; lean_object* v___x_796_; uint8_t v_isShared_797_; uint8_t v_isSharedCheck_801_; 
lean_dec(v_a_726_);
lean_dec(v_goal_717_);
lean_dec_ref(v___f_716_);
lean_dec_ref(v___f_715_);
lean_dec_ref(v___f_714_);
lean_dec_ref(v___f_713_);
lean_dec_ref(v_unfold_x3f_712_);
lean_dec(v_fvarId_711_);
v_a_794_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_801_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_801_ == 0)
{
v___x_796_ = v___x_727_;
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
else
{
lean_inc(v_a_794_);
lean_dec(v___x_727_);
v___x_796_ = lean_box(0);
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
v_resetjp_795_:
{
lean_object* v___x_799_; 
if (v_isShared_797_ == 0)
{
v___x_799_ = v___x_796_;
goto v_reusejp_798_;
}
else
{
lean_object* v_reuseFailAlloc_800_; 
v_reuseFailAlloc_800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_800_, 0, v_a_794_);
v___x_799_ = v_reuseFailAlloc_800_;
goto v_reusejp_798_;
}
v_reusejp_798_:
{
return v___x_799_;
}
}
}
}
else
{
lean_object* v_a_802_; lean_object* v___x_804_; uint8_t v_isShared_805_; uint8_t v_isSharedCheck_809_; 
lean_dec(v_goal_717_);
lean_dec_ref(v___f_716_);
lean_dec_ref(v___f_715_);
lean_dec_ref(v___f_714_);
lean_dec_ref(v___f_713_);
lean_dec_ref(v_unfold_x3f_712_);
lean_dec(v_fvarId_711_);
v_a_802_ = lean_ctor_get(v___x_723_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_723_);
if (v_isSharedCheck_809_ == 0)
{
v___x_804_ = v___x_723_;
v_isShared_805_ = v_isSharedCheck_809_;
goto v_resetjp_803_;
}
else
{
lean_inc(v_a_802_);
lean_dec(v___x_723_);
v___x_804_ = lean_box(0);
v_isShared_805_ = v_isSharedCheck_809_;
goto v_resetjp_803_;
}
v_resetjp_803_:
{
lean_object* v___x_807_; 
if (v_isShared_805_ == 0)
{
v___x_807_ = v___x_804_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v_a_802_);
v___x_807_ = v_reuseFailAlloc_808_;
goto v_reusejp_806_;
}
v_reusejp_806_:
{
return v___x_807_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___lam__5___boxed(lean_object* v_fvarId_810_, lean_object* v_unfold_x3f_811_, lean_object* v___f_812_, lean_object* v___f_813_, lean_object* v___f_814_, lean_object* v___f_815_, lean_object* v_goal_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_aesop_Aesop_unfoldManyAt___lam__5(v_fvarId_810_, v_unfold_x3f_811_, v___f_812_, v___f_813_, v___f_814_, v___f_815_, v_goal_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt(lean_object* v_unfold_x3f_823_, lean_object* v_goal_824_, lean_object* v_fvarId_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v___f_831_; lean_object* v___f_832_; lean_object* v___f_833_; lean_object* v___f_834_; lean_object* v___f_835_; lean_object* v___x_836_; 
v___f_831_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__0));
v___f_832_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__1));
v___f_833_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__2));
v___f_834_ = ((lean_object*)(lp_aesop_Aesop_unfoldManyCore___closed__3));
lean_inc(v_goal_824_);
v___f_835_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldManyAt___lam__5___boxed), 12, 7);
lean_closure_set(v___f_835_, 0, v_fvarId_825_);
lean_closure_set(v___f_835_, 1, v_unfold_x3f_823_);
lean_closure_set(v___f_835_, 2, v___f_831_);
lean_closure_set(v___f_835_, 3, v___f_832_);
lean_closure_set(v___f_835_, 4, v___f_833_);
lean_closure_set(v___f_835_, 5, v___f_834_);
lean_closure_set(v___f_835_, 6, v_goal_824_);
v___x_836_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(v_goal_824_, v___f_835_, v_a_826_, v_a_827_, v_a_828_, v_a_829_);
return v___x_836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAt___boxed(lean_object* v_unfold_x3f_837_, lean_object* v_goal_838_, lean_object* v_fvarId_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_, lean_object* v_a_843_, lean_object* v_a_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_837_, v_goal_838_, v_fvarId_839_, v_a_840_, v_a_841_, v_a_842_, v_a_843_);
lean_dec(v_a_843_);
lean_dec_ref(v_a_842_);
lean_dec(v_a_841_);
lean_dec_ref(v_a_840_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3(lean_object* v_unfold_x3f_846_, lean_object* v_as_847_, size_t v_sz_848_, size_t v_i_849_, lean_object* v_b_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_){
_start:
{
uint8_t v___x_856_; 
v___x_856_ = lean_usize_dec_lt(v_i_849_, v_sz_848_);
if (v___x_856_ == 0)
{
lean_object* v___x_857_; 
lean_dec_ref(v_unfold_x3f_846_);
v___x_857_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_857_, 0, v_b_850_);
return v___x_857_;
}
else
{
lean_object* v_snd_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_887_; 
v_snd_858_ = lean_ctor_get(v_b_850_, 1);
v_isSharedCheck_887_ = !lean_is_exclusive(v_b_850_);
if (v_isSharedCheck_887_ == 0)
{
lean_object* v_unused_888_; 
v_unused_888_ = lean_ctor_get(v_b_850_, 0);
lean_dec(v_unused_888_);
v___x_860_ = v_b_850_;
v_isShared_861_ = v_isSharedCheck_887_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_snd_858_);
lean_dec(v_b_850_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_887_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_862_; lean_object* v_a_864_; lean_object* v_a_871_; 
v___x_862_ = lean_box(0);
v_a_871_ = lean_array_uget_borrowed(v_as_847_, v_i_849_);
if (lean_obj_tag(v_a_871_) == 0)
{
v_a_864_ = v_snd_858_;
goto v___jp_863_;
}
else
{
lean_object* v_val_872_; uint8_t v___x_873_; 
v_val_872_ = lean_ctor_get(v_a_871_, 0);
v___x_873_ = l_Lean_LocalDecl_isImplementationDetail(v_val_872_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; lean_object* v___x_875_; 
v___x_874_ = l_Lean_LocalDecl_fvarId(v_val_872_);
lean_inc(v_snd_858_);
lean_inc_ref(v_unfold_x3f_846_);
v___x_875_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_846_, v_snd_858_, v___x_874_, v___y_851_, v___y_852_, v___y_853_, v___y_854_);
if (lean_obj_tag(v___x_875_) == 0)
{
lean_object* v_a_876_; 
v_a_876_ = lean_ctor_get(v___x_875_, 0);
lean_inc(v_a_876_);
lean_dec_ref_known(v___x_875_, 1);
if (lean_obj_tag(v_a_876_) == 1)
{
lean_object* v_val_877_; lean_object* v_fst_878_; 
lean_dec(v_snd_858_);
v_val_877_ = lean_ctor_get(v_a_876_, 0);
lean_inc(v_val_877_);
lean_dec_ref_known(v_a_876_, 1);
v_fst_878_ = lean_ctor_get(v_val_877_, 0);
lean_inc(v_fst_878_);
lean_dec(v_val_877_);
v_a_864_ = v_fst_878_;
goto v___jp_863_;
}
else
{
lean_dec(v_a_876_);
v_a_864_ = v_snd_858_;
goto v___jp_863_;
}
}
else
{
lean_object* v_a_879_; lean_object* v___x_881_; uint8_t v_isShared_882_; uint8_t v_isSharedCheck_886_; 
lean_del_object(v___x_860_);
lean_dec(v_snd_858_);
lean_dec_ref(v_unfold_x3f_846_);
v_a_879_ = lean_ctor_get(v___x_875_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_875_);
if (v_isSharedCheck_886_ == 0)
{
v___x_881_ = v___x_875_;
v_isShared_882_ = v_isSharedCheck_886_;
goto v_resetjp_880_;
}
else
{
lean_inc(v_a_879_);
lean_dec(v___x_875_);
v___x_881_ = lean_box(0);
v_isShared_882_ = v_isSharedCheck_886_;
goto v_resetjp_880_;
}
v_resetjp_880_:
{
lean_object* v___x_884_; 
if (v_isShared_882_ == 0)
{
v___x_884_ = v___x_881_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v_a_879_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
}
}
else
{
v_a_864_ = v_snd_858_;
goto v___jp_863_;
}
}
v___jp_863_:
{
lean_object* v___x_866_; 
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 1, v_a_864_);
lean_ctor_set(v___x_860_, 0, v___x_862_);
v___x_866_ = v___x_860_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v___x_862_);
lean_ctor_set(v_reuseFailAlloc_870_, 1, v_a_864_);
v___x_866_ = v_reuseFailAlloc_870_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
size_t v___x_867_; size_t v___x_868_; 
v___x_867_ = ((size_t)1ULL);
v___x_868_ = lean_usize_add(v_i_849_, v___x_867_);
v_i_849_ = v___x_868_;
v_b_850_ = v___x_866_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_unfold_x3f_889_, lean_object* v_as_890_, lean_object* v_sz_891_, lean_object* v_i_892_, lean_object* v_b_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
size_t v_sz_boxed_899_; size_t v_i_boxed_900_; lean_object* v_res_901_; 
v_sz_boxed_899_ = lean_unbox_usize(v_sz_891_);
lean_dec(v_sz_891_);
v_i_boxed_900_ = lean_unbox_usize(v_i_892_);
lean_dec(v_i_892_);
v_res_901_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3(v_unfold_x3f_889_, v_as_890_, v_sz_boxed_899_, v_i_boxed_900_, v_b_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
lean_dec(v___y_897_);
lean_dec_ref(v___y_896_);
lean_dec(v___y_895_);
lean_dec_ref(v___y_894_);
lean_dec_ref(v_as_890_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2(lean_object* v_unfold_x3f_902_, lean_object* v_as_903_, size_t v_sz_904_, size_t v_i_905_, lean_object* v_b_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_){
_start:
{
uint8_t v___x_912_; 
v___x_912_ = lean_usize_dec_lt(v_i_905_, v_sz_904_);
if (v___x_912_ == 0)
{
lean_object* v___x_913_; 
lean_dec_ref(v_unfold_x3f_902_);
v___x_913_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_913_, 0, v_b_906_);
return v___x_913_;
}
else
{
lean_object* v_snd_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_943_; 
v_snd_914_ = lean_ctor_get(v_b_906_, 1);
v_isSharedCheck_943_ = !lean_is_exclusive(v_b_906_);
if (v_isSharedCheck_943_ == 0)
{
lean_object* v_unused_944_; 
v_unused_944_ = lean_ctor_get(v_b_906_, 0);
lean_dec(v_unused_944_);
v___x_916_ = v_b_906_;
v_isShared_917_ = v_isSharedCheck_943_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_snd_914_);
lean_dec(v_b_906_);
v___x_916_ = lean_box(0);
v_isShared_917_ = v_isSharedCheck_943_;
goto v_resetjp_915_;
}
v_resetjp_915_:
{
lean_object* v___x_918_; lean_object* v_a_920_; lean_object* v_a_927_; 
v___x_918_ = lean_box(0);
v_a_927_ = lean_array_uget_borrowed(v_as_903_, v_i_905_);
if (lean_obj_tag(v_a_927_) == 0)
{
v_a_920_ = v_snd_914_;
goto v___jp_919_;
}
else
{
lean_object* v_val_928_; uint8_t v___x_929_; 
v_val_928_ = lean_ctor_get(v_a_927_, 0);
v___x_929_ = l_Lean_LocalDecl_isImplementationDetail(v_val_928_);
if (v___x_929_ == 0)
{
lean_object* v___x_930_; lean_object* v___x_931_; 
v___x_930_ = l_Lean_LocalDecl_fvarId(v_val_928_);
lean_inc(v_snd_914_);
lean_inc_ref(v_unfold_x3f_902_);
v___x_931_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_902_, v_snd_914_, v___x_930_, v___y_907_, v___y_908_, v___y_909_, v___y_910_);
if (lean_obj_tag(v___x_931_) == 0)
{
lean_object* v_a_932_; 
v_a_932_ = lean_ctor_get(v___x_931_, 0);
lean_inc(v_a_932_);
lean_dec_ref_known(v___x_931_, 1);
if (lean_obj_tag(v_a_932_) == 1)
{
lean_object* v_val_933_; lean_object* v_fst_934_; 
lean_dec(v_snd_914_);
v_val_933_ = lean_ctor_get(v_a_932_, 0);
lean_inc(v_val_933_);
lean_dec_ref_known(v_a_932_, 1);
v_fst_934_ = lean_ctor_get(v_val_933_, 0);
lean_inc(v_fst_934_);
lean_dec(v_val_933_);
v_a_920_ = v_fst_934_;
goto v___jp_919_;
}
else
{
lean_dec(v_a_932_);
v_a_920_ = v_snd_914_;
goto v___jp_919_;
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
lean_del_object(v___x_916_);
lean_dec(v_snd_914_);
lean_dec_ref(v_unfold_x3f_902_);
v_a_935_ = lean_ctor_get(v___x_931_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_931_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_931_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_931_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
lean_object* v___x_940_; 
if (v_isShared_938_ == 0)
{
v___x_940_ = v___x_937_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_a_935_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
}
}
else
{
v_a_920_ = v_snd_914_;
goto v___jp_919_;
}
}
v___jp_919_:
{
lean_object* v___x_922_; 
if (v_isShared_917_ == 0)
{
lean_ctor_set(v___x_916_, 1, v_a_920_);
lean_ctor_set(v___x_916_, 0, v___x_918_);
v___x_922_ = v___x_916_;
goto v_reusejp_921_;
}
else
{
lean_object* v_reuseFailAlloc_926_; 
v_reuseFailAlloc_926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_926_, 0, v___x_918_);
lean_ctor_set(v_reuseFailAlloc_926_, 1, v_a_920_);
v___x_922_ = v_reuseFailAlloc_926_;
goto v_reusejp_921_;
}
v_reusejp_921_:
{
size_t v___x_923_; size_t v___x_924_; lean_object* v___x_925_; 
v___x_923_ = ((size_t)1ULL);
v___x_924_ = lean_usize_add(v_i_905_, v___x_923_);
v___x_925_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2_spec__3(v_unfold_x3f_902_, v_as_903_, v_sz_904_, v___x_924_, v___x_922_, v___y_907_, v___y_908_, v___y_909_, v___y_910_);
return v___x_925_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2___boxed(lean_object* v_unfold_x3f_945_, lean_object* v_as_946_, lean_object* v_sz_947_, lean_object* v_i_948_, lean_object* v_b_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_){
_start:
{
size_t v_sz_boxed_955_; size_t v_i_boxed_956_; lean_object* v_res_957_; 
v_sz_boxed_955_ = lean_unbox_usize(v_sz_947_);
lean_dec(v_sz_947_);
v_i_boxed_956_ = lean_unbox_usize(v_i_948_);
lean_dec(v_i_948_);
v_res_957_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2(v_unfold_x3f_945_, v_as_946_, v_sz_boxed_955_, v_i_boxed_956_, v_b_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_);
lean_dec(v___y_953_);
lean_dec_ref(v___y_952_);
lean_dec(v___y_951_);
lean_dec_ref(v___y_950_);
lean_dec_ref(v_as_946_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0(lean_object* v_init_958_, lean_object* v_unfold_x3f_959_, lean_object* v_n_960_, lean_object* v_b_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
if (lean_obj_tag(v_n_960_) == 0)
{
lean_object* v_cs_967_; lean_object* v___x_968_; lean_object* v___x_969_; size_t v_sz_970_; size_t v___x_971_; lean_object* v___x_972_; 
v_cs_967_ = lean_ctor_get(v_n_960_, 0);
v___x_968_ = lean_box(0);
v___x_969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_969_, 0, v___x_968_);
lean_ctor_set(v___x_969_, 1, v_b_961_);
v_sz_970_ = lean_array_size(v_cs_967_);
v___x_971_ = ((size_t)0ULL);
v___x_972_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1(v_init_958_, v_unfold_x3f_959_, v_cs_967_, v_sz_970_, v___x_971_, v___x_969_, v___y_962_, v___y_963_, v___y_964_, v___y_965_);
if (lean_obj_tag(v___x_972_) == 0)
{
lean_object* v_a_973_; lean_object* v___x_975_; uint8_t v_isShared_976_; uint8_t v_isSharedCheck_987_; 
v_a_973_ = lean_ctor_get(v___x_972_, 0);
v_isSharedCheck_987_ = !lean_is_exclusive(v___x_972_);
if (v_isSharedCheck_987_ == 0)
{
v___x_975_ = v___x_972_;
v_isShared_976_ = v_isSharedCheck_987_;
goto v_resetjp_974_;
}
else
{
lean_inc(v_a_973_);
lean_dec(v___x_972_);
v___x_975_ = lean_box(0);
v_isShared_976_ = v_isSharedCheck_987_;
goto v_resetjp_974_;
}
v_resetjp_974_:
{
lean_object* v_fst_977_; 
v_fst_977_ = lean_ctor_get(v_a_973_, 0);
if (lean_obj_tag(v_fst_977_) == 0)
{
lean_object* v_snd_978_; lean_object* v___x_979_; lean_object* v___x_981_; 
v_snd_978_ = lean_ctor_get(v_a_973_, 1);
lean_inc(v_snd_978_);
lean_dec(v_a_973_);
v___x_979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_979_, 0, v_snd_978_);
if (v_isShared_976_ == 0)
{
lean_ctor_set(v___x_975_, 0, v___x_979_);
v___x_981_ = v___x_975_;
goto v_reusejp_980_;
}
else
{
lean_object* v_reuseFailAlloc_982_; 
v_reuseFailAlloc_982_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_982_, 0, v___x_979_);
v___x_981_ = v_reuseFailAlloc_982_;
goto v_reusejp_980_;
}
v_reusejp_980_:
{
return v___x_981_;
}
}
else
{
lean_object* v_val_983_; lean_object* v___x_985_; 
lean_inc_ref(v_fst_977_);
lean_dec(v_a_973_);
v_val_983_ = lean_ctor_get(v_fst_977_, 0);
lean_inc(v_val_983_);
lean_dec_ref_known(v_fst_977_, 1);
if (v_isShared_976_ == 0)
{
lean_ctor_set(v___x_975_, 0, v_val_983_);
v___x_985_ = v___x_975_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v_val_983_);
v___x_985_ = v_reuseFailAlloc_986_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
return v___x_985_;
}
}
}
}
else
{
lean_object* v_a_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_995_; 
v_a_988_ = lean_ctor_get(v___x_972_, 0);
v_isSharedCheck_995_ = !lean_is_exclusive(v___x_972_);
if (v_isSharedCheck_995_ == 0)
{
v___x_990_ = v___x_972_;
v_isShared_991_ = v_isSharedCheck_995_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_a_988_);
lean_dec(v___x_972_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_995_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___x_993_; 
if (v_isShared_991_ == 0)
{
v___x_993_ = v___x_990_;
goto v_reusejp_992_;
}
else
{
lean_object* v_reuseFailAlloc_994_; 
v_reuseFailAlloc_994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_994_, 0, v_a_988_);
v___x_993_ = v_reuseFailAlloc_994_;
goto v_reusejp_992_;
}
v_reusejp_992_:
{
return v___x_993_;
}
}
}
}
else
{
lean_object* v_vs_996_; lean_object* v___x_997_; lean_object* v___x_998_; size_t v_sz_999_; size_t v___x_1000_; lean_object* v___x_1001_; 
v_vs_996_ = lean_ctor_get(v_n_960_, 0);
v___x_997_ = lean_box(0);
v___x_998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_998_, 0, v___x_997_);
lean_ctor_set(v___x_998_, 1, v_b_961_);
v_sz_999_ = lean_array_size(v_vs_996_);
v___x_1000_ = ((size_t)0ULL);
v___x_1001_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__2(v_unfold_x3f_959_, v_vs_996_, v_sz_999_, v___x_1000_, v___x_998_, v___y_962_, v___y_963_, v___y_964_, v___y_965_);
if (lean_obj_tag(v___x_1001_) == 0)
{
lean_object* v_a_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1016_; 
v_a_1002_ = lean_ctor_get(v___x_1001_, 0);
v_isSharedCheck_1016_ = !lean_is_exclusive(v___x_1001_);
if (v_isSharedCheck_1016_ == 0)
{
v___x_1004_ = v___x_1001_;
v_isShared_1005_ = v_isSharedCheck_1016_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_a_1002_);
lean_dec(v___x_1001_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1016_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
lean_object* v_fst_1006_; 
v_fst_1006_ = lean_ctor_get(v_a_1002_, 0);
if (lean_obj_tag(v_fst_1006_) == 0)
{
lean_object* v_snd_1007_; lean_object* v___x_1008_; lean_object* v___x_1010_; 
v_snd_1007_ = lean_ctor_get(v_a_1002_, 1);
lean_inc(v_snd_1007_);
lean_dec(v_a_1002_);
v___x_1008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1008_, 0, v_snd_1007_);
if (v_isShared_1005_ == 0)
{
lean_ctor_set(v___x_1004_, 0, v___x_1008_);
v___x_1010_ = v___x_1004_;
goto v_reusejp_1009_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_1008_);
v___x_1010_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1009_;
}
v_reusejp_1009_:
{
return v___x_1010_;
}
}
else
{
lean_object* v_val_1012_; lean_object* v___x_1014_; 
lean_inc_ref(v_fst_1006_);
lean_dec(v_a_1002_);
v_val_1012_ = lean_ctor_get(v_fst_1006_, 0);
lean_inc(v_val_1012_);
lean_dec_ref_known(v_fst_1006_, 1);
if (v_isShared_1005_ == 0)
{
lean_ctor_set(v___x_1004_, 0, v_val_1012_);
v___x_1014_ = v___x_1004_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v_val_1012_);
v___x_1014_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
return v___x_1014_;
}
}
}
}
else
{
lean_object* v_a_1017_; lean_object* v___x_1019_; uint8_t v_isShared_1020_; uint8_t v_isSharedCheck_1024_; 
v_a_1017_ = lean_ctor_get(v___x_1001_, 0);
v_isSharedCheck_1024_ = !lean_is_exclusive(v___x_1001_);
if (v_isSharedCheck_1024_ == 0)
{
v___x_1019_ = v___x_1001_;
v_isShared_1020_ = v_isSharedCheck_1024_;
goto v_resetjp_1018_;
}
else
{
lean_inc(v_a_1017_);
lean_dec(v___x_1001_);
v___x_1019_ = lean_box(0);
v_isShared_1020_ = v_isSharedCheck_1024_;
goto v_resetjp_1018_;
}
v_resetjp_1018_:
{
lean_object* v___x_1022_; 
if (v_isShared_1020_ == 0)
{
v___x_1022_ = v___x_1019_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v_a_1017_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
return v___x_1022_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1(lean_object* v_init_1025_, lean_object* v_unfold_x3f_1026_, lean_object* v_as_1027_, size_t v_sz_1028_, size_t v_i_1029_, lean_object* v_b_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
uint8_t v___x_1036_; 
v___x_1036_ = lean_usize_dec_lt(v_i_1029_, v_sz_1028_);
if (v___x_1036_ == 0)
{
lean_object* v___x_1037_; 
lean_dec_ref(v_unfold_x3f_1026_);
v___x_1037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1037_, 0, v_b_1030_);
return v___x_1037_;
}
else
{
lean_object* v_snd_1038_; lean_object* v___x_1040_; uint8_t v_isShared_1041_; uint8_t v_isSharedCheck_1072_; 
v_snd_1038_ = lean_ctor_get(v_b_1030_, 1);
v_isSharedCheck_1072_ = !lean_is_exclusive(v_b_1030_);
if (v_isSharedCheck_1072_ == 0)
{
lean_object* v_unused_1073_; 
v_unused_1073_ = lean_ctor_get(v_b_1030_, 0);
lean_dec(v_unused_1073_);
v___x_1040_ = v_b_1030_;
v_isShared_1041_ = v_isSharedCheck_1072_;
goto v_resetjp_1039_;
}
else
{
lean_inc(v_snd_1038_);
lean_dec(v_b_1030_);
v___x_1040_ = lean_box(0);
v_isShared_1041_ = v_isSharedCheck_1072_;
goto v_resetjp_1039_;
}
v_resetjp_1039_:
{
lean_object* v_a_1042_; lean_object* v___x_1043_; 
v_a_1042_ = lean_array_uget_borrowed(v_as_1027_, v_i_1029_);
lean_inc(v_snd_1038_);
lean_inc_ref(v_unfold_x3f_1026_);
v___x_1043_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0(v_init_1025_, v_unfold_x3f_1026_, v_a_1042_, v_snd_1038_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_);
if (lean_obj_tag(v___x_1043_) == 0)
{
lean_object* v_a_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1063_; 
v_a_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1063_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1063_ == 0)
{
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1063_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_a_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1063_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
if (lean_obj_tag(v_a_1044_) == 0)
{
lean_object* v___x_1048_; lean_object* v___x_1050_; 
lean_dec_ref(v_unfold_x3f_1026_);
v___x_1048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1048_, 0, v_a_1044_);
if (v_isShared_1041_ == 0)
{
lean_ctor_set(v___x_1040_, 0, v___x_1048_);
v___x_1050_ = v___x_1040_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v___x_1048_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v_snd_1038_);
v___x_1050_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
lean_object* v___x_1052_; 
if (v_isShared_1047_ == 0)
{
lean_ctor_set(v___x_1046_, 0, v___x_1050_);
v___x_1052_ = v___x_1046_;
goto v_reusejp_1051_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v___x_1050_);
v___x_1052_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1051_;
}
v_reusejp_1051_:
{
return v___x_1052_;
}
}
}
else
{
lean_object* v_a_1055_; lean_object* v___x_1056_; lean_object* v___x_1058_; 
lean_del_object(v___x_1046_);
lean_dec(v_snd_1038_);
v_a_1055_ = lean_ctor_get(v_a_1044_, 0);
lean_inc(v_a_1055_);
lean_dec_ref_known(v_a_1044_, 1);
v___x_1056_ = lean_box(0);
if (v_isShared_1041_ == 0)
{
lean_ctor_set(v___x_1040_, 1, v_a_1055_);
lean_ctor_set(v___x_1040_, 0, v___x_1056_);
v___x_1058_ = v___x_1040_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1062_; 
v_reuseFailAlloc_1062_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1062_, 0, v___x_1056_);
lean_ctor_set(v_reuseFailAlloc_1062_, 1, v_a_1055_);
v___x_1058_ = v_reuseFailAlloc_1062_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
size_t v___x_1059_; size_t v___x_1060_; 
v___x_1059_ = ((size_t)1ULL);
v___x_1060_ = lean_usize_add(v_i_1029_, v___x_1059_);
v_i_1029_ = v___x_1060_;
v_b_1030_ = v___x_1058_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1071_; 
lean_del_object(v___x_1040_);
lean_dec(v_snd_1038_);
lean_dec_ref(v_unfold_x3f_1026_);
v_a_1064_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1071_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1066_ = v___x_1043_;
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v___x_1043_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1069_; 
if (v_isShared_1067_ == 0)
{
v___x_1069_ = v___x_1066_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_a_1064_);
v___x_1069_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
return v___x_1069_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1___boxed(lean_object* v_init_1074_, lean_object* v_unfold_x3f_1075_, lean_object* v_as_1076_, lean_object* v_sz_1077_, lean_object* v_i_1078_, lean_object* v_b_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_){
_start:
{
size_t v_sz_boxed_1085_; size_t v_i_boxed_1086_; lean_object* v_res_1087_; 
v_sz_boxed_1085_ = lean_unbox_usize(v_sz_1077_);
lean_dec(v_sz_1077_);
v_i_boxed_1086_ = lean_unbox_usize(v_i_1078_);
lean_dec(v_i_1078_);
v_res_1087_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0_spec__1(v_init_1074_, v_unfold_x3f_1075_, v_as_1076_, v_sz_boxed_1085_, v_i_boxed_1086_, v_b_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
lean_dec(v___y_1083_);
lean_dec_ref(v___y_1082_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
lean_dec_ref(v_as_1076_);
lean_dec(v_init_1074_);
return v_res_1087_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0___boxed(lean_object* v_init_1088_, lean_object* v_unfold_x3f_1089_, lean_object* v_n_1090_, lean_object* v_b_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
lean_object* v_res_1097_; 
v_res_1097_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0(v_init_1088_, v_unfold_x3f_1089_, v_n_1090_, v_b_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
lean_dec(v___y_1093_);
lean_dec_ref(v___y_1092_);
lean_dec_ref(v_n_1090_);
lean_dec(v_init_1088_);
return v_res_1097_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4(lean_object* v_unfold_x3f_1098_, lean_object* v_as_1099_, size_t v_sz_1100_, size_t v_i_1101_, lean_object* v_b_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
uint8_t v___x_1108_; 
v___x_1108_ = lean_usize_dec_lt(v_i_1101_, v_sz_1100_);
if (v___x_1108_ == 0)
{
lean_object* v___x_1109_; 
lean_dec_ref(v_unfold_x3f_1098_);
v___x_1109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1109_, 0, v_b_1102_);
return v___x_1109_;
}
else
{
lean_object* v_snd_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1139_; 
v_snd_1110_ = lean_ctor_get(v_b_1102_, 1);
v_isSharedCheck_1139_ = !lean_is_exclusive(v_b_1102_);
if (v_isSharedCheck_1139_ == 0)
{
lean_object* v_unused_1140_; 
v_unused_1140_ = lean_ctor_get(v_b_1102_, 0);
lean_dec(v_unused_1140_);
v___x_1112_ = v_b_1102_;
v_isShared_1113_ = v_isSharedCheck_1139_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_snd_1110_);
lean_dec(v_b_1102_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1139_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1114_; lean_object* v_a_1116_; lean_object* v_a_1123_; 
v___x_1114_ = lean_box(0);
v_a_1123_ = lean_array_uget_borrowed(v_as_1099_, v_i_1101_);
if (lean_obj_tag(v_a_1123_) == 0)
{
v_a_1116_ = v_snd_1110_;
goto v___jp_1115_;
}
else
{
lean_object* v_val_1124_; uint8_t v___x_1125_; 
v_val_1124_ = lean_ctor_get(v_a_1123_, 0);
v___x_1125_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1124_);
if (v___x_1125_ == 0)
{
lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1126_ = l_Lean_LocalDecl_fvarId(v_val_1124_);
lean_inc(v_snd_1110_);
lean_inc_ref(v_unfold_x3f_1098_);
v___x_1127_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_1098_, v_snd_1110_, v___x_1126_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
if (lean_obj_tag(v___x_1127_) == 0)
{
lean_object* v_a_1128_; 
v_a_1128_ = lean_ctor_get(v___x_1127_, 0);
lean_inc(v_a_1128_);
lean_dec_ref_known(v___x_1127_, 1);
if (lean_obj_tag(v_a_1128_) == 1)
{
lean_object* v_val_1129_; lean_object* v_fst_1130_; 
lean_dec(v_snd_1110_);
v_val_1129_ = lean_ctor_get(v_a_1128_, 0);
lean_inc(v_val_1129_);
lean_dec_ref_known(v_a_1128_, 1);
v_fst_1130_ = lean_ctor_get(v_val_1129_, 0);
lean_inc(v_fst_1130_);
lean_dec(v_val_1129_);
v_a_1116_ = v_fst_1130_;
goto v___jp_1115_;
}
else
{
lean_dec(v_a_1128_);
v_a_1116_ = v_snd_1110_;
goto v___jp_1115_;
}
}
else
{
lean_object* v_a_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1138_; 
lean_del_object(v___x_1112_);
lean_dec(v_snd_1110_);
lean_dec_ref(v_unfold_x3f_1098_);
v_a_1131_ = lean_ctor_get(v___x_1127_, 0);
v_isSharedCheck_1138_ = !lean_is_exclusive(v___x_1127_);
if (v_isSharedCheck_1138_ == 0)
{
v___x_1133_ = v___x_1127_;
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_a_1131_);
lean_dec(v___x_1127_);
v___x_1133_ = lean_box(0);
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
v_resetjp_1132_:
{
lean_object* v___x_1136_; 
if (v_isShared_1134_ == 0)
{
v___x_1136_ = v___x_1133_;
goto v_reusejp_1135_;
}
else
{
lean_object* v_reuseFailAlloc_1137_; 
v_reuseFailAlloc_1137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1137_, 0, v_a_1131_);
v___x_1136_ = v_reuseFailAlloc_1137_;
goto v_reusejp_1135_;
}
v_reusejp_1135_:
{
return v___x_1136_;
}
}
}
}
else
{
v_a_1116_ = v_snd_1110_;
goto v___jp_1115_;
}
}
v___jp_1115_:
{
lean_object* v___x_1118_; 
if (v_isShared_1113_ == 0)
{
lean_ctor_set(v___x_1112_, 1, v_a_1116_);
lean_ctor_set(v___x_1112_, 0, v___x_1114_);
v___x_1118_ = v___x_1112_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1122_; 
v_reuseFailAlloc_1122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1122_, 0, v___x_1114_);
lean_ctor_set(v_reuseFailAlloc_1122_, 1, v_a_1116_);
v___x_1118_ = v_reuseFailAlloc_1122_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
size_t v___x_1119_; size_t v___x_1120_; 
v___x_1119_ = ((size_t)1ULL);
v___x_1120_ = lean_usize_add(v_i_1101_, v___x_1119_);
v_i_1101_ = v___x_1120_;
v_b_1102_ = v___x_1118_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4___boxed(lean_object* v_unfold_x3f_1141_, lean_object* v_as_1142_, lean_object* v_sz_1143_, lean_object* v_i_1144_, lean_object* v_b_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_){
_start:
{
size_t v_sz_boxed_1151_; size_t v_i_boxed_1152_; lean_object* v_res_1153_; 
v_sz_boxed_1151_ = lean_unbox_usize(v_sz_1143_);
lean_dec(v_sz_1143_);
v_i_boxed_1152_ = lean_unbox_usize(v_i_1144_);
lean_dec(v_i_1144_);
v_res_1153_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4(v_unfold_x3f_1141_, v_as_1142_, v_sz_boxed_1151_, v_i_boxed_1152_, v_b_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_);
lean_dec(v___y_1149_);
lean_dec_ref(v___y_1148_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
lean_dec_ref(v_as_1142_);
return v_res_1153_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1(lean_object* v_unfold_x3f_1154_, lean_object* v_as_1155_, size_t v_sz_1156_, size_t v_i_1157_, lean_object* v_b_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_){
_start:
{
uint8_t v___x_1164_; 
v___x_1164_ = lean_usize_dec_lt(v_i_1157_, v_sz_1156_);
if (v___x_1164_ == 0)
{
lean_object* v___x_1165_; 
lean_dec_ref(v_unfold_x3f_1154_);
v___x_1165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1165_, 0, v_b_1158_);
return v___x_1165_;
}
else
{
lean_object* v_snd_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1195_; 
v_snd_1166_ = lean_ctor_get(v_b_1158_, 1);
v_isSharedCheck_1195_ = !lean_is_exclusive(v_b_1158_);
if (v_isSharedCheck_1195_ == 0)
{
lean_object* v_unused_1196_; 
v_unused_1196_ = lean_ctor_get(v_b_1158_, 0);
lean_dec(v_unused_1196_);
v___x_1168_ = v_b_1158_;
v_isShared_1169_ = v_isSharedCheck_1195_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_snd_1166_);
lean_dec(v_b_1158_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1195_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
lean_object* v___x_1170_; lean_object* v_a_1172_; lean_object* v_a_1179_; 
v___x_1170_ = lean_box(0);
v_a_1179_ = lean_array_uget_borrowed(v_as_1155_, v_i_1157_);
if (lean_obj_tag(v_a_1179_) == 0)
{
v_a_1172_ = v_snd_1166_;
goto v___jp_1171_;
}
else
{
lean_object* v_val_1180_; uint8_t v___x_1181_; 
v_val_1180_ = lean_ctor_get(v_a_1179_, 0);
v___x_1181_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1180_);
if (v___x_1181_ == 0)
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1182_ = l_Lean_LocalDecl_fvarId(v_val_1180_);
lean_inc(v_snd_1166_);
lean_inc_ref(v_unfold_x3f_1154_);
v___x_1183_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_1154_, v_snd_1166_, v___x_1182_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_);
if (lean_obj_tag(v___x_1183_) == 0)
{
lean_object* v_a_1184_; 
v_a_1184_ = lean_ctor_get(v___x_1183_, 0);
lean_inc(v_a_1184_);
lean_dec_ref_known(v___x_1183_, 1);
if (lean_obj_tag(v_a_1184_) == 1)
{
lean_object* v_val_1185_; lean_object* v_fst_1186_; 
lean_dec(v_snd_1166_);
v_val_1185_ = lean_ctor_get(v_a_1184_, 0);
lean_inc(v_val_1185_);
lean_dec_ref_known(v_a_1184_, 1);
v_fst_1186_ = lean_ctor_get(v_val_1185_, 0);
lean_inc(v_fst_1186_);
lean_dec(v_val_1185_);
v_a_1172_ = v_fst_1186_;
goto v___jp_1171_;
}
else
{
lean_dec(v_a_1184_);
v_a_1172_ = v_snd_1166_;
goto v___jp_1171_;
}
}
else
{
lean_object* v_a_1187_; lean_object* v___x_1189_; uint8_t v_isShared_1190_; uint8_t v_isSharedCheck_1194_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec_ref(v_unfold_x3f_1154_);
v_a_1187_ = lean_ctor_get(v___x_1183_, 0);
v_isSharedCheck_1194_ = !lean_is_exclusive(v___x_1183_);
if (v_isSharedCheck_1194_ == 0)
{
v___x_1189_ = v___x_1183_;
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
else
{
lean_inc(v_a_1187_);
lean_dec(v___x_1183_);
v___x_1189_ = lean_box(0);
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
v_resetjp_1188_:
{
lean_object* v___x_1192_; 
if (v_isShared_1190_ == 0)
{
v___x_1192_ = v___x_1189_;
goto v_reusejp_1191_;
}
else
{
lean_object* v_reuseFailAlloc_1193_; 
v_reuseFailAlloc_1193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1193_, 0, v_a_1187_);
v___x_1192_ = v_reuseFailAlloc_1193_;
goto v_reusejp_1191_;
}
v_reusejp_1191_:
{
return v___x_1192_;
}
}
}
}
else
{
v_a_1172_ = v_snd_1166_;
goto v___jp_1171_;
}
}
v___jp_1171_:
{
lean_object* v___x_1174_; 
if (v_isShared_1169_ == 0)
{
lean_ctor_set(v___x_1168_, 1, v_a_1172_);
lean_ctor_set(v___x_1168_, 0, v___x_1170_);
v___x_1174_ = v___x_1168_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1178_; 
v_reuseFailAlloc_1178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1178_, 0, v___x_1170_);
lean_ctor_set(v_reuseFailAlloc_1178_, 1, v_a_1172_);
v___x_1174_ = v_reuseFailAlloc_1178_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
size_t v___x_1175_; size_t v___x_1176_; lean_object* v___x_1177_; 
v___x_1175_ = ((size_t)1ULL);
v___x_1176_ = lean_usize_add(v_i_1157_, v___x_1175_);
v___x_1177_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1_spec__4(v_unfold_x3f_1154_, v_as_1155_, v_sz_1156_, v___x_1176_, v___x_1174_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_);
return v___x_1177_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1___boxed(lean_object* v_unfold_x3f_1197_, lean_object* v_as_1198_, lean_object* v_sz_1199_, lean_object* v_i_1200_, lean_object* v_b_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
size_t v_sz_boxed_1207_; size_t v_i_boxed_1208_; lean_object* v_res_1209_; 
v_sz_boxed_1207_ = lean_unbox_usize(v_sz_1199_);
lean_dec(v_sz_1199_);
v_i_boxed_1208_ = lean_unbox_usize(v_i_1200_);
lean_dec(v_i_1200_);
v_res_1209_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1(v_unfold_x3f_1197_, v_as_1198_, v_sz_boxed_1207_, v_i_boxed_1208_, v_b_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
lean_dec_ref(v_as_1198_);
return v_res_1209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0(lean_object* v_unfold_x3f_1210_, lean_object* v_t_1211_, lean_object* v_init_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_){
_start:
{
lean_object* v_root_1218_; lean_object* v_tail_1219_; lean_object* v___x_1220_; 
v_root_1218_ = lean_ctor_get(v_t_1211_, 0);
v_tail_1219_ = lean_ctor_get(v_t_1211_, 1);
lean_inc_ref(v_unfold_x3f_1210_);
lean_inc(v_init_1212_);
v___x_1220_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__0(v_init_1212_, v_unfold_x3f_1210_, v_root_1218_, v_init_1212_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_);
lean_dec(v_init_1212_);
if (lean_obj_tag(v___x_1220_) == 0)
{
lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1257_; 
v_a_1221_ = lean_ctor_get(v___x_1220_, 0);
v_isSharedCheck_1257_ = !lean_is_exclusive(v___x_1220_);
if (v_isSharedCheck_1257_ == 0)
{
v___x_1223_ = v___x_1220_;
v_isShared_1224_ = v_isSharedCheck_1257_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1220_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1257_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
if (lean_obj_tag(v_a_1221_) == 0)
{
lean_object* v_a_1225_; lean_object* v___x_1227_; 
lean_dec_ref(v_unfold_x3f_1210_);
v_a_1225_ = lean_ctor_get(v_a_1221_, 0);
lean_inc(v_a_1225_);
lean_dec_ref_known(v_a_1221_, 1);
if (v_isShared_1224_ == 0)
{
lean_ctor_set(v___x_1223_, 0, v_a_1225_);
v___x_1227_ = v___x_1223_;
goto v_reusejp_1226_;
}
else
{
lean_object* v_reuseFailAlloc_1228_; 
v_reuseFailAlloc_1228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1228_, 0, v_a_1225_);
v___x_1227_ = v_reuseFailAlloc_1228_;
goto v_reusejp_1226_;
}
v_reusejp_1226_:
{
return v___x_1227_;
}
}
else
{
lean_object* v_a_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; size_t v_sz_1232_; size_t v___x_1233_; lean_object* v___x_1234_; 
lean_del_object(v___x_1223_);
v_a_1229_ = lean_ctor_get(v_a_1221_, 0);
lean_inc(v_a_1229_);
lean_dec_ref_known(v_a_1221_, 1);
v___x_1230_ = lean_box(0);
v___x_1231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
lean_ctor_set(v___x_1231_, 1, v_a_1229_);
v_sz_1232_ = lean_array_size(v_tail_1219_);
v___x_1233_ = ((size_t)0ULL);
v___x_1234_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0_spec__1(v_unfold_x3f_1210_, v_tail_1219_, v_sz_1232_, v___x_1233_, v___x_1231_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_);
if (lean_obj_tag(v___x_1234_) == 0)
{
lean_object* v_a_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1248_; 
v_a_1235_ = lean_ctor_get(v___x_1234_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1234_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1237_ = v___x_1234_;
v_isShared_1238_ = v_isSharedCheck_1248_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_a_1235_);
lean_dec(v___x_1234_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1248_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v_fst_1239_; 
v_fst_1239_ = lean_ctor_get(v_a_1235_, 0);
if (lean_obj_tag(v_fst_1239_) == 0)
{
lean_object* v_snd_1240_; lean_object* v___x_1242_; 
v_snd_1240_ = lean_ctor_get(v_a_1235_, 1);
lean_inc(v_snd_1240_);
lean_dec(v_a_1235_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set(v___x_1237_, 0, v_snd_1240_);
v___x_1242_ = v___x_1237_;
goto v_reusejp_1241_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v_snd_1240_);
v___x_1242_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1241_;
}
v_reusejp_1241_:
{
return v___x_1242_;
}
}
else
{
lean_object* v_val_1244_; lean_object* v___x_1246_; 
lean_inc_ref(v_fst_1239_);
lean_dec(v_a_1235_);
v_val_1244_ = lean_ctor_get(v_fst_1239_, 0);
lean_inc(v_val_1244_);
lean_dec_ref_known(v_fst_1239_, 1);
if (v_isShared_1238_ == 0)
{
lean_ctor_set(v___x_1237_, 0, v_val_1244_);
v___x_1246_ = v___x_1237_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_val_1244_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
}
else
{
lean_object* v_a_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1256_; 
v_a_1249_ = lean_ctor_get(v___x_1234_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v___x_1234_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v___x_1234_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v___x_1234_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1254_; 
if (v_isShared_1252_ == 0)
{
v___x_1254_ = v___x_1251_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v_a_1249_);
v___x_1254_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1253_;
}
v_reusejp_1253_:
{
return v___x_1254_;
}
}
}
}
}
}
else
{
lean_object* v_a_1258_; lean_object* v___x_1260_; uint8_t v_isShared_1261_; uint8_t v_isSharedCheck_1265_; 
lean_dec_ref(v_unfold_x3f_1210_);
v_a_1258_ = lean_ctor_get(v___x_1220_, 0);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1220_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1260_ = v___x_1220_;
v_isShared_1261_ = v_isSharedCheck_1265_;
goto v_resetjp_1259_;
}
else
{
lean_inc(v_a_1258_);
lean_dec(v___x_1220_);
v___x_1260_ = lean_box(0);
v_isShared_1261_ = v_isSharedCheck_1265_;
goto v_resetjp_1259_;
}
v_resetjp_1259_:
{
lean_object* v___x_1263_; 
if (v_isShared_1261_ == 0)
{
v___x_1263_ = v___x_1260_;
goto v_reusejp_1262_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v_a_1258_);
v___x_1263_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1262_;
}
v_reusejp_1262_:
{
return v___x_1263_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0___boxed(lean_object* v_unfold_x3f_1266_, lean_object* v_t_1267_, lean_object* v_init_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0(v_unfold_x3f_1266_, v_t_1267_, v_init_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
lean_dec_ref(v_t_1267_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___lam__0(lean_object* v_unfold_x3f_1275_, lean_object* v_goal_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_){
_start:
{
lean_object* v___x_1282_; 
lean_inc(v_goal_1276_);
lean_inc_ref(v_unfold_x3f_1275_);
v___x_1282_ = lp_aesop_Aesop_unfoldManyTarget(v_unfold_x3f_1275_, v_goal_1276_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_);
if (lean_obj_tag(v___x_1282_) == 0)
{
lean_object* v_a_1283_; lean_object* v_goal_1285_; lean_object* v___y_1286_; lean_object* v___y_1287_; lean_object* v___y_1288_; lean_object* v___y_1289_; 
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref_known(v___x_1282_, 1);
if (lean_obj_tag(v_a_1283_) == 1)
{
lean_object* v_val_1325_; lean_object* v_fst_1326_; 
v_val_1325_ = lean_ctor_get(v_a_1283_, 0);
lean_inc(v_val_1325_);
lean_dec_ref_known(v_a_1283_, 1);
v_fst_1326_ = lean_ctor_get(v_val_1325_, 0);
lean_inc(v_fst_1326_);
lean_dec(v_val_1325_);
v_goal_1285_ = v_fst_1326_;
v___y_1286_ = v___y_1277_;
v___y_1287_ = v___y_1278_;
v___y_1288_ = v___y_1279_;
v___y_1289_ = v___y_1280_;
goto v___jp_1284_;
}
else
{
lean_dec(v_a_1283_);
lean_inc(v_goal_1276_);
v_goal_1285_ = v_goal_1276_;
v___y_1286_ = v___y_1277_;
v___y_1287_ = v___y_1278_;
v___y_1288_ = v___y_1279_;
v___y_1289_ = v___y_1280_;
goto v___jp_1284_;
}
v___jp_1284_:
{
lean_object* v___x_1290_; 
lean_inc(v_goal_1285_);
v___x_1290_ = l_Lean_MVarId_getDecl(v_goal_1285_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1290_) == 0)
{
lean_object* v_a_1291_; lean_object* v_lctx_1292_; lean_object* v_decls_1293_; lean_object* v___x_1294_; 
v_a_1291_ = lean_ctor_get(v___x_1290_, 0);
lean_inc(v_a_1291_);
lean_dec_ref_known(v___x_1290_, 1);
v_lctx_1292_ = lean_ctor_get(v_a_1291_, 1);
lean_inc_ref(v_lctx_1292_);
lean_dec(v_a_1291_);
v_decls_1293_ = lean_ctor_get(v_lctx_1292_, 1);
lean_inc_ref(v_decls_1293_);
lean_dec_ref(v_lctx_1292_);
v___x_1294_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStar_spec__0(v_unfold_x3f_1275_, v_decls_1293_, v_goal_1285_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
lean_dec_ref(v_decls_1293_);
if (lean_obj_tag(v___x_1294_) == 0)
{
lean_object* v_a_1295_; lean_object* v___x_1297_; uint8_t v_isShared_1298_; uint8_t v_isSharedCheck_1308_; 
v_a_1295_ = lean_ctor_get(v___x_1294_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v___x_1294_);
if (v_isSharedCheck_1308_ == 0)
{
v___x_1297_ = v___x_1294_;
v_isShared_1298_ = v_isSharedCheck_1308_;
goto v_resetjp_1296_;
}
else
{
lean_inc(v_a_1295_);
lean_dec(v___x_1294_);
v___x_1297_ = lean_box(0);
v_isShared_1298_ = v_isSharedCheck_1308_;
goto v_resetjp_1296_;
}
v_resetjp_1296_:
{
uint8_t v___x_1299_; 
v___x_1299_ = l_Lean_instBEqMVarId_beq(v_a_1295_, v_goal_1276_);
lean_dec(v_goal_1276_);
if (v___x_1299_ == 0)
{
lean_object* v___x_1300_; lean_object* v___x_1302_; 
v___x_1300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1300_, 0, v_a_1295_);
if (v_isShared_1298_ == 0)
{
lean_ctor_set(v___x_1297_, 0, v___x_1300_);
v___x_1302_ = v___x_1297_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v___x_1300_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
else
{
lean_object* v___x_1304_; lean_object* v___x_1306_; 
lean_dec(v_a_1295_);
v___x_1304_ = lean_box(0);
if (v_isShared_1298_ == 0)
{
lean_ctor_set(v___x_1297_, 0, v___x_1304_);
v___x_1306_ = v___x_1297_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v___x_1304_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
}
}
else
{
lean_object* v_a_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1316_; 
lean_dec(v_goal_1276_);
v_a_1309_ = lean_ctor_get(v___x_1294_, 0);
v_isSharedCheck_1316_ = !lean_is_exclusive(v___x_1294_);
if (v_isSharedCheck_1316_ == 0)
{
v___x_1311_ = v___x_1294_;
v_isShared_1312_ = v_isSharedCheck_1316_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_a_1309_);
lean_dec(v___x_1294_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1316_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1314_; 
if (v_isShared_1312_ == 0)
{
v___x_1314_ = v___x_1311_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1315_; 
v_reuseFailAlloc_1315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1315_, 0, v_a_1309_);
v___x_1314_ = v_reuseFailAlloc_1315_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
return v___x_1314_;
}
}
}
}
else
{
lean_object* v_a_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1324_; 
lean_dec(v_goal_1285_);
lean_dec(v_goal_1276_);
lean_dec_ref(v_unfold_x3f_1275_);
v_a_1317_ = lean_ctor_get(v___x_1290_, 0);
v_isSharedCheck_1324_ = !lean_is_exclusive(v___x_1290_);
if (v_isSharedCheck_1324_ == 0)
{
v___x_1319_ = v___x_1290_;
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_a_1317_);
lean_dec(v___x_1290_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1322_; 
if (v_isShared_1320_ == 0)
{
v___x_1322_ = v___x_1319_;
goto v_reusejp_1321_;
}
else
{
lean_object* v_reuseFailAlloc_1323_; 
v_reuseFailAlloc_1323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1323_, 0, v_a_1317_);
v___x_1322_ = v_reuseFailAlloc_1323_;
goto v_reusejp_1321_;
}
v_reusejp_1321_:
{
return v___x_1322_;
}
}
}
}
}
else
{
lean_object* v_a_1327_; lean_object* v___x_1329_; uint8_t v_isShared_1330_; uint8_t v_isSharedCheck_1334_; 
lean_dec(v_goal_1276_);
lean_dec_ref(v_unfold_x3f_1275_);
v_a_1327_ = lean_ctor_get(v___x_1282_, 0);
v_isSharedCheck_1334_ = !lean_is_exclusive(v___x_1282_);
if (v_isSharedCheck_1334_ == 0)
{
v___x_1329_ = v___x_1282_;
v_isShared_1330_ = v_isSharedCheck_1334_;
goto v_resetjp_1328_;
}
else
{
lean_inc(v_a_1327_);
lean_dec(v___x_1282_);
v___x_1329_ = lean_box(0);
v_isShared_1330_ = v_isSharedCheck_1334_;
goto v_resetjp_1328_;
}
v_resetjp_1328_:
{
lean_object* v___x_1332_; 
if (v_isShared_1330_ == 0)
{
v___x_1332_ = v___x_1329_;
goto v_reusejp_1331_;
}
else
{
lean_object* v_reuseFailAlloc_1333_; 
v_reuseFailAlloc_1333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1333_, 0, v_a_1327_);
v___x_1332_ = v_reuseFailAlloc_1333_;
goto v_reusejp_1331_;
}
v_reusejp_1331_:
{
return v___x_1332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___lam__0___boxed(lean_object* v_unfold_x3f_1335_, lean_object* v_goal_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v_res_1342_; 
v_res_1342_ = lp_aesop_Aesop_unfoldManyStar___lam__0(v_unfold_x3f_1335_, v_goal_1336_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_);
lean_dec(v___y_1340_);
lean_dec_ref(v___y_1339_);
lean_dec(v___y_1338_);
lean_dec_ref(v___y_1337_);
return v_res_1342_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar(lean_object* v_unfold_x3f_1343_, lean_object* v_goal_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_){
_start:
{
lean_object* v___f_1350_; lean_object* v___x_1351_; 
lean_inc(v_goal_1344_);
v___f_1350_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldManyStar___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1350_, 0, v_unfold_x3f_1343_);
lean_closure_set(v___f_1350_, 1, v_goal_1344_);
v___x_1351_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyAt_spec__0___redArg(v_goal_1344_, v___f_1350_, v_a_1345_, v_a_1346_, v_a_1347_, v_a_1348_);
return v___x_1351_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStar___boxed(lean_object* v_unfold_x3f_1352_, lean_object* v_goal_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_, lean_object* v_a_1358_){
_start:
{
lean_object* v_res_1359_; 
v_res_1359_ = lp_aesop_Aesop_unfoldManyStar(v_unfold_x3f_1352_, v_goal_1353_, v_a_1354_, v_a_1355_, v_a_1356_, v_a_1357_);
lean_dec(v_a_1357_);
lean_dec_ref(v_a_1356_);
lean_dec(v_a_1355_);
lean_dec_ref(v_a_1354_);
return v_res_1359_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_WHNF(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_Unfold(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_WHNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_Unfold(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
lean_object* initialize_Lean_Meta_WHNF(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_Unfold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_WHNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_Unfold(builtin);
}
#ifdef __cplusplus
}
#endif
