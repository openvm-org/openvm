// Lean compiler output
// Module: Aesop.Search.Expansion.Simp
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Simp.SimpAll
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
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addLetDeclToUnfold(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_hasValue(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setSimpTheorems(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setFailIfUnchanged(lean_object*, uint8_t);
lean_object* l_Lean_Meta_simpGoal(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_simpAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_solved_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_solved_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_unchanged_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_unchanged_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_simplified_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_simplified_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_newGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_newGoal_x3f___boxed(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5;
static lean_once_cell_t lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoal(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_aesop_Aesop_SimpResult_ctorIdx(v_x_5_);
lean_dec(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim___redArg(lean_object* v_t_7_, lean_object* v_k_8_){
_start:
{
switch(lean_obj_tag(v_t_7_))
{
case 0:
{
lean_object* v_usedTheorems_9_; lean_object* v___x_10_; 
v_usedTheorems_9_ = lean_ctor_get(v_t_7_, 0);
lean_inc_ref(v_usedTheorems_9_);
lean_dec_ref_known(v_t_7_, 1);
v___x_10_ = lean_apply_1(v_k_8_, v_usedTheorems_9_);
return v___x_10_;
}
case 1:
{
return v_k_8_;
}
default: 
{
lean_object* v_newGoal_11_; lean_object* v_usedTheorems_12_; lean_object* v___x_13_; 
v_newGoal_11_ = lean_ctor_get(v_t_7_, 0);
lean_inc(v_newGoal_11_);
v_usedTheorems_12_ = lean_ctor_get(v_t_7_, 1);
lean_inc_ref(v_usedTheorems_12_);
lean_dec_ref_known(v_t_7_, 2);
v___x_13_ = lean_apply_2(v_k_8_, v_newGoal_11_, v_usedTheorems_12_);
return v___x_13_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim(lean_object* v_motive_14_, lean_object* v_ctorIdx_15_, lean_object* v_t_16_, lean_object* v_h_17_, lean_object* v_k_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_16_, v_k_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_ctorElim___boxed(lean_object* v_motive_20_, lean_object* v_ctorIdx_21_, lean_object* v_t_22_, lean_object* v_h_23_, lean_object* v_k_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_SimpResult_ctorElim(v_motive_20_, v_ctorIdx_21_, v_t_22_, v_h_23_, v_k_24_);
lean_dec(v_ctorIdx_21_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_solved_elim___redArg(lean_object* v_t_26_, lean_object* v_solved_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_26_, v_solved_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_solved_elim(lean_object* v_motive_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_solved_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_30_, v_solved_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_unchanged_elim___redArg(lean_object* v_t_34_, lean_object* v_unchanged_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_34_, v_unchanged_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_unchanged_elim(lean_object* v_motive_37_, lean_object* v_t_38_, lean_object* v_h_39_, lean_object* v_unchanged_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_38_, v_unchanged_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_simplified_elim___redArg(lean_object* v_t_42_, lean_object* v_simplified_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_42_, v_simplified_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_simplified_elim(lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_simplified_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_aesop_Aesop_SimpResult_ctorElim___redArg(v_t_46_, v_simplified_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_newGoal_x3f(lean_object* v_x_50_){
_start:
{
if (lean_obj_tag(v_x_50_) == 2)
{
lean_object* v_newGoal_51_; lean_object* v___x_52_; 
v_newGoal_51_ = lean_ctor_get(v_x_50_, 0);
lean_inc(v_newGoal_51_);
v___x_52_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_52_, 0, v_newGoal_51_);
return v___x_52_;
}
else
{
lean_object* v___x_53_; 
v___x_53_ = lean_box(0);
return v___x_53_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SimpResult_newGoal_x3f___boxed(lean_object* v_x_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_aesop_Aesop_SimpResult_newGoal_x3f(v_x_54_);
lean_dec(v_x_54_);
return v_res_55_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0(void){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_56_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__0);
v___x_58_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1(lean_object* v_00_u03b2_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1___closed__1);
return v___x_60_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0(void){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_61_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__0);
v___x_63_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2(lean_object* v_00_u03b2_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2___closed__1);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg(lean_object* v_as_66_, size_t v_sz_67_, size_t v_i_68_, lean_object* v_b_69_){
_start:
{
uint8_t v___x_71_; 
v___x_71_ = lean_usize_dec_lt(v_i_68_, v_sz_67_);
if (v___x_71_ == 0)
{
lean_object* v___x_72_; 
v___x_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_72_, 0, v_b_69_);
return v___x_72_;
}
else
{
lean_object* v_snd_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_102_; 
v_snd_73_ = lean_ctor_get(v_b_69_, 1);
v_isSharedCheck_102_ = !lean_is_exclusive(v_b_69_);
if (v_isSharedCheck_102_ == 0)
{
lean_object* v_unused_103_; 
v_unused_103_ = lean_ctor_get(v_b_69_, 0);
lean_dec(v_unused_103_);
v___x_75_ = v_b_69_;
v_isShared_76_ = v_isSharedCheck_102_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_snd_73_);
lean_dec(v_b_69_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_102_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_77_; lean_object* v_a_79_; lean_object* v_a_86_; 
v___x_77_ = lean_box(0);
v_a_86_ = lean_array_uget_borrowed(v_as_66_, v_i_68_);
if (lean_obj_tag(v_a_86_) == 0)
{
v_a_79_ = v_snd_73_;
goto v___jp_78_;
}
else
{
lean_object* v_val_87_; uint8_t v___y_89_; uint8_t v___x_99_; uint8_t v___x_100_; 
v_val_87_ = lean_ctor_get(v_a_86_, 0);
v___x_99_ = 0;
v___x_100_ = l_Lean_LocalDecl_hasValue(v_val_87_, v___x_99_);
if (v___x_100_ == 0)
{
v___y_89_ = v___x_100_;
goto v___jp_88_;
}
else
{
uint8_t v___x_101_; 
v___x_101_ = l_Lean_LocalDecl_isImplementationDetail(v_val_87_);
if (v___x_101_ == 0)
{
v___y_89_ = v___x_100_;
goto v___jp_88_;
}
else
{
v_a_79_ = v_snd_73_;
goto v___jp_78_;
}
}
v___jp_88_:
{
if (v___y_89_ == 0)
{
v_a_79_ = v_snd_73_;
goto v___jp_78_;
}
else
{
lean_object* v___x_90_; lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_90_ = lean_unsigned_to_nat(0u);
v___x_91_ = lean_array_get_size(v_snd_73_);
v___x_92_ = lean_nat_dec_lt(v___x_90_, v___x_91_);
if (v___x_92_ == 0)
{
v_a_79_ = v_snd_73_;
goto v___jp_78_;
}
else
{
lean_object* v_v_93_; lean_object* v___x_94_; lean_object* v_xs_x27_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v_v_93_ = lean_array_fget(v_snd_73_, v___x_90_);
v___x_94_ = lean_box(0);
v_xs_x27_95_ = lean_array_fset(v_snd_73_, v___x_90_, v___x_94_);
v___x_96_ = l_Lean_LocalDecl_fvarId(v_val_87_);
v___x_97_ = l_Lean_Meta_SimpTheorems_addLetDeclToUnfold(v_v_93_, v___x_96_);
v___x_98_ = lean_array_fset(v_xs_x27_95_, v___x_90_, v___x_97_);
v_a_79_ = v___x_98_;
goto v___jp_78_;
}
}
}
}
v___jp_78_:
{
lean_object* v___x_81_; 
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 1, v_a_79_);
lean_ctor_set(v___x_75_, 0, v___x_77_);
v___x_81_ = v___x_75_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___x_77_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v_a_79_);
v___x_81_ = v_reuseFailAlloc_85_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
size_t v___x_82_; size_t v___x_83_; 
v___x_82_ = ((size_t)1ULL);
v___x_83_ = lean_usize_add(v_i_68_, v___x_82_);
v_i_68_ = v___x_83_;
v_b_69_ = v___x_81_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg___boxed(lean_object* v_as_104_, lean_object* v_sz_105_, lean_object* v_i_106_, lean_object* v_b_107_, lean_object* v___y_108_){
_start:
{
size_t v_sz_boxed_109_; size_t v_i_boxed_110_; lean_object* v_res_111_; 
v_sz_boxed_109_ = lean_unbox_usize(v_sz_105_);
lean_dec(v_sz_105_);
v_i_boxed_110_ = lean_unbox_usize(v_i_106_);
lean_dec(v_i_106_);
v_res_111_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg(v_as_104_, v_sz_boxed_109_, v_i_boxed_110_, v_b_107_);
lean_dec_ref(v_as_104_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1(lean_object* v_as_112_, size_t v_sz_113_, size_t v_i_114_, lean_object* v_b_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
uint8_t v___x_121_; 
v___x_121_ = lean_usize_dec_lt(v_i_114_, v_sz_113_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
v___x_122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_122_, 0, v_b_115_);
return v___x_122_;
}
else
{
lean_object* v_snd_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_152_; 
v_snd_123_ = lean_ctor_get(v_b_115_, 1);
v_isSharedCheck_152_ = !lean_is_exclusive(v_b_115_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; 
v_unused_153_ = lean_ctor_get(v_b_115_, 0);
lean_dec(v_unused_153_);
v___x_125_ = v_b_115_;
v_isShared_126_ = v_isSharedCheck_152_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_snd_123_);
lean_dec(v_b_115_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_152_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_127_; lean_object* v_a_129_; lean_object* v_a_136_; 
v___x_127_ = lean_box(0);
v_a_136_ = lean_array_uget_borrowed(v_as_112_, v_i_114_);
if (lean_obj_tag(v_a_136_) == 0)
{
v_a_129_ = v_snd_123_;
goto v___jp_128_;
}
else
{
lean_object* v_val_137_; uint8_t v___y_139_; uint8_t v___x_149_; uint8_t v___x_150_; 
v_val_137_ = lean_ctor_get(v_a_136_, 0);
v___x_149_ = 0;
v___x_150_ = l_Lean_LocalDecl_hasValue(v_val_137_, v___x_149_);
if (v___x_150_ == 0)
{
v___y_139_ = v___x_150_;
goto v___jp_138_;
}
else
{
uint8_t v___x_151_; 
v___x_151_ = l_Lean_LocalDecl_isImplementationDetail(v_val_137_);
if (v___x_151_ == 0)
{
v___y_139_ = v___x_150_;
goto v___jp_138_;
}
else
{
v_a_129_ = v_snd_123_;
goto v___jp_128_;
}
}
v___jp_138_:
{
if (v___y_139_ == 0)
{
v_a_129_ = v_snd_123_;
goto v___jp_128_;
}
else
{
lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_140_ = lean_unsigned_to_nat(0u);
v___x_141_ = lean_array_get_size(v_snd_123_);
v___x_142_ = lean_nat_dec_lt(v___x_140_, v___x_141_);
if (v___x_142_ == 0)
{
v_a_129_ = v_snd_123_;
goto v___jp_128_;
}
else
{
lean_object* v_v_143_; lean_object* v___x_144_; lean_object* v_xs_x27_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_v_143_ = lean_array_fget(v_snd_123_, v___x_140_);
v___x_144_ = lean_box(0);
v_xs_x27_145_ = lean_array_fset(v_snd_123_, v___x_140_, v___x_144_);
v___x_146_ = l_Lean_LocalDecl_fvarId(v_val_137_);
v___x_147_ = l_Lean_Meta_SimpTheorems_addLetDeclToUnfold(v_v_143_, v___x_146_);
v___x_148_ = lean_array_fset(v_xs_x27_145_, v___x_140_, v___x_147_);
v_a_129_ = v___x_148_;
goto v___jp_128_;
}
}
}
}
v___jp_128_:
{
lean_object* v___x_131_; 
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 1, v_a_129_);
lean_ctor_set(v___x_125_, 0, v___x_127_);
v___x_131_ = v___x_125_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_127_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v_a_129_);
v___x_131_ = v_reuseFailAlloc_135_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
size_t v___x_132_; size_t v___x_133_; lean_object* v___x_134_; 
v___x_132_ = ((size_t)1ULL);
v___x_133_ = lean_usize_add(v_i_114_, v___x_132_);
v___x_134_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg(v_as_112_, v_sz_113_, v___x_133_, v___x_131_);
return v___x_134_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1___boxed(lean_object* v_as_154_, lean_object* v_sz_155_, lean_object* v_i_156_, lean_object* v_b_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
size_t v_sz_boxed_163_; size_t v_i_boxed_164_; lean_object* v_res_165_; 
v_sz_boxed_163_ = lean_unbox_usize(v_sz_155_);
lean_dec(v_sz_155_);
v_i_boxed_164_ = lean_unbox_usize(v_i_156_);
lean_dec(v_i_156_);
v_res_165_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1(v_as_154_, v_sz_boxed_163_, v_i_boxed_164_, v_b_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
lean_dec_ref(v_as_154_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg(lean_object* v_as_166_, size_t v_sz_167_, size_t v_i_168_, lean_object* v_b_169_){
_start:
{
uint8_t v___x_171_; 
v___x_171_ = lean_usize_dec_lt(v_i_168_, v_sz_167_);
if (v___x_171_ == 0)
{
lean_object* v___x_172_; 
v___x_172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_172_, 0, v_b_169_);
return v___x_172_;
}
else
{
lean_object* v_snd_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_202_; 
v_snd_173_ = lean_ctor_get(v_b_169_, 1);
v_isSharedCheck_202_ = !lean_is_exclusive(v_b_169_);
if (v_isSharedCheck_202_ == 0)
{
lean_object* v_unused_203_; 
v_unused_203_ = lean_ctor_get(v_b_169_, 0);
lean_dec(v_unused_203_);
v___x_175_ = v_b_169_;
v_isShared_176_ = v_isSharedCheck_202_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_snd_173_);
lean_dec(v_b_169_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_202_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_177_; lean_object* v_a_179_; lean_object* v_a_186_; 
v___x_177_ = lean_box(0);
v_a_186_ = lean_array_uget_borrowed(v_as_166_, v_i_168_);
if (lean_obj_tag(v_a_186_) == 0)
{
v_a_179_ = v_snd_173_;
goto v___jp_178_;
}
else
{
lean_object* v_val_187_; uint8_t v___y_189_; uint8_t v___x_199_; uint8_t v___x_200_; 
v_val_187_ = lean_ctor_get(v_a_186_, 0);
v___x_199_ = 0;
v___x_200_ = l_Lean_LocalDecl_hasValue(v_val_187_, v___x_199_);
if (v___x_200_ == 0)
{
v___y_189_ = v___x_200_;
goto v___jp_188_;
}
else
{
uint8_t v___x_201_; 
v___x_201_ = l_Lean_LocalDecl_isImplementationDetail(v_val_187_);
if (v___x_201_ == 0)
{
v___y_189_ = v___x_200_;
goto v___jp_188_;
}
else
{
v_a_179_ = v_snd_173_;
goto v___jp_178_;
}
}
v___jp_188_:
{
if (v___y_189_ == 0)
{
v_a_179_ = v_snd_173_;
goto v___jp_178_;
}
else
{
lean_object* v___x_190_; lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = lean_array_get_size(v_snd_173_);
v___x_192_ = lean_nat_dec_lt(v___x_190_, v___x_191_);
if (v___x_192_ == 0)
{
v_a_179_ = v_snd_173_;
goto v___jp_178_;
}
else
{
lean_object* v_v_193_; lean_object* v___x_194_; lean_object* v_xs_x27_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v_v_193_ = lean_array_fget(v_snd_173_, v___x_190_);
v___x_194_ = lean_box(0);
v_xs_x27_195_ = lean_array_fset(v_snd_173_, v___x_190_, v___x_194_);
v___x_196_ = l_Lean_LocalDecl_fvarId(v_val_187_);
v___x_197_ = l_Lean_Meta_SimpTheorems_addLetDeclToUnfold(v_v_193_, v___x_196_);
v___x_198_ = lean_array_fset(v_xs_x27_195_, v___x_190_, v___x_197_);
v_a_179_ = v___x_198_;
goto v___jp_178_;
}
}
}
}
v___jp_178_:
{
lean_object* v___x_181_; 
if (v_isShared_176_ == 0)
{
lean_ctor_set(v___x_175_, 1, v_a_179_);
lean_ctor_set(v___x_175_, 0, v___x_177_);
v___x_181_ = v___x_175_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_177_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_a_179_);
v___x_181_ = v_reuseFailAlloc_185_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
size_t v___x_182_; size_t v___x_183_; 
v___x_182_ = ((size_t)1ULL);
v___x_183_ = lean_usize_add(v_i_168_, v___x_182_);
v_i_168_ = v___x_183_;
v_b_169_ = v___x_181_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg___boxed(lean_object* v_as_204_, lean_object* v_sz_205_, lean_object* v_i_206_, lean_object* v_b_207_, lean_object* v___y_208_){
_start:
{
size_t v_sz_boxed_209_; size_t v_i_boxed_210_; lean_object* v_res_211_; 
v_sz_boxed_209_ = lean_unbox_usize(v_sz_205_);
lean_dec(v_sz_205_);
v_i_boxed_210_ = lean_unbox_usize(v_i_206_);
lean_dec(v_i_206_);
v_res_211_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg(v_as_204_, v_sz_boxed_209_, v_i_boxed_210_, v_b_207_);
lean_dec_ref(v_as_204_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4(lean_object* v_as_212_, size_t v_sz_213_, size_t v_i_214_, lean_object* v_b_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_){
_start:
{
uint8_t v___x_221_; 
v___x_221_ = lean_usize_dec_lt(v_i_214_, v_sz_213_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; 
v___x_222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_222_, 0, v_b_215_);
return v___x_222_;
}
else
{
lean_object* v_snd_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_252_; 
v_snd_223_ = lean_ctor_get(v_b_215_, 1);
v_isSharedCheck_252_ = !lean_is_exclusive(v_b_215_);
if (v_isSharedCheck_252_ == 0)
{
lean_object* v_unused_253_; 
v_unused_253_ = lean_ctor_get(v_b_215_, 0);
lean_dec(v_unused_253_);
v___x_225_ = v_b_215_;
v_isShared_226_ = v_isSharedCheck_252_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_snd_223_);
lean_dec(v_b_215_);
v___x_225_ = lean_box(0);
v_isShared_226_ = v_isSharedCheck_252_;
goto v_resetjp_224_;
}
v_resetjp_224_:
{
lean_object* v___x_227_; lean_object* v_a_229_; lean_object* v_a_236_; 
v___x_227_ = lean_box(0);
v_a_236_ = lean_array_uget_borrowed(v_as_212_, v_i_214_);
if (lean_obj_tag(v_a_236_) == 0)
{
v_a_229_ = v_snd_223_;
goto v___jp_228_;
}
else
{
lean_object* v_val_237_; uint8_t v___y_239_; uint8_t v___x_249_; uint8_t v___x_250_; 
v_val_237_ = lean_ctor_get(v_a_236_, 0);
v___x_249_ = 0;
v___x_250_ = l_Lean_LocalDecl_hasValue(v_val_237_, v___x_249_);
if (v___x_250_ == 0)
{
v___y_239_ = v___x_250_;
goto v___jp_238_;
}
else
{
uint8_t v___x_251_; 
v___x_251_ = l_Lean_LocalDecl_isImplementationDetail(v_val_237_);
if (v___x_251_ == 0)
{
v___y_239_ = v___x_250_;
goto v___jp_238_;
}
else
{
v_a_229_ = v_snd_223_;
goto v___jp_228_;
}
}
v___jp_238_:
{
if (v___y_239_ == 0)
{
v_a_229_ = v_snd_223_;
goto v___jp_228_;
}
else
{
lean_object* v___x_240_; lean_object* v___x_241_; uint8_t v___x_242_; 
v___x_240_ = lean_unsigned_to_nat(0u);
v___x_241_ = lean_array_get_size(v_snd_223_);
v___x_242_ = lean_nat_dec_lt(v___x_240_, v___x_241_);
if (v___x_242_ == 0)
{
v_a_229_ = v_snd_223_;
goto v___jp_228_;
}
else
{
lean_object* v_v_243_; lean_object* v___x_244_; lean_object* v_xs_x27_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v_v_243_ = lean_array_fget(v_snd_223_, v___x_240_);
v___x_244_ = lean_box(0);
v_xs_x27_245_ = lean_array_fset(v_snd_223_, v___x_240_, v___x_244_);
v___x_246_ = l_Lean_LocalDecl_fvarId(v_val_237_);
v___x_247_ = l_Lean_Meta_SimpTheorems_addLetDeclToUnfold(v_v_243_, v___x_246_);
v___x_248_ = lean_array_fset(v_xs_x27_245_, v___x_240_, v___x_247_);
v_a_229_ = v___x_248_;
goto v___jp_228_;
}
}
}
}
v___jp_228_:
{
lean_object* v___x_231_; 
if (v_isShared_226_ == 0)
{
lean_ctor_set(v___x_225_, 1, v_a_229_);
lean_ctor_set(v___x_225_, 0, v___x_227_);
v___x_231_ = v___x_225_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_227_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v_a_229_);
v___x_231_ = v_reuseFailAlloc_235_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
size_t v___x_232_; size_t v___x_233_; lean_object* v___x_234_; 
v___x_232_ = ((size_t)1ULL);
v___x_233_ = lean_usize_add(v_i_214_, v___x_232_);
v___x_234_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg(v_as_212_, v_sz_213_, v___x_233_, v___x_231_);
return v___x_234_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4___boxed(lean_object* v_as_254_, lean_object* v_sz_255_, lean_object* v_i_256_, lean_object* v_b_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
size_t v_sz_boxed_263_; size_t v_i_boxed_264_; lean_object* v_res_265_; 
v_sz_boxed_263_ = lean_unbox_usize(v_sz_255_);
lean_dec(v_sz_255_);
v_i_boxed_264_ = lean_unbox_usize(v_i_256_);
lean_dec(v_i_256_);
v_res_265_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4(v_as_254_, v_sz_boxed_263_, v_i_boxed_264_, v_b_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec_ref(v_as_254_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0(lean_object* v_init_266_, lean_object* v_n_267_, lean_object* v_b_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
if (lean_obj_tag(v_n_267_) == 0)
{
lean_object* v_cs_274_; lean_object* v___x_275_; lean_object* v___x_276_; size_t v_sz_277_; size_t v___x_278_; lean_object* v___x_279_; 
v_cs_274_ = lean_ctor_get(v_n_267_, 0);
v___x_275_ = lean_box(0);
v___x_276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
lean_ctor_set(v___x_276_, 1, v_b_268_);
v_sz_277_ = lean_array_size(v_cs_274_);
v___x_278_ = ((size_t)0ULL);
v___x_279_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3(v_init_266_, v_cs_274_, v_sz_277_, v___x_278_, v___x_276_, v___y_269_, v___y_270_, v___y_271_, v___y_272_);
if (lean_obj_tag(v___x_279_) == 0)
{
lean_object* v_a_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_294_; 
v_a_280_ = lean_ctor_get(v___x_279_, 0);
v_isSharedCheck_294_ = !lean_is_exclusive(v___x_279_);
if (v_isSharedCheck_294_ == 0)
{
v___x_282_ = v___x_279_;
v_isShared_283_ = v_isSharedCheck_294_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_a_280_);
lean_dec(v___x_279_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_294_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v_fst_284_; 
v_fst_284_ = lean_ctor_get(v_a_280_, 0);
if (lean_obj_tag(v_fst_284_) == 0)
{
lean_object* v_snd_285_; lean_object* v___x_286_; lean_object* v___x_288_; 
v_snd_285_ = lean_ctor_get(v_a_280_, 1);
lean_inc(v_snd_285_);
lean_dec(v_a_280_);
v___x_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_286_, 0, v_snd_285_);
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 0, v___x_286_);
v___x_288_ = v___x_282_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_286_);
v___x_288_ = v_reuseFailAlloc_289_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
return v___x_288_;
}
}
else
{
lean_object* v_val_290_; lean_object* v___x_292_; 
lean_inc_ref(v_fst_284_);
lean_dec(v_a_280_);
v_val_290_ = lean_ctor_get(v_fst_284_, 0);
lean_inc(v_val_290_);
lean_dec_ref_known(v_fst_284_, 1);
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 0, v_val_290_);
v___x_292_ = v___x_282_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v_val_290_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
return v___x_292_;
}
}
}
}
else
{
lean_object* v_a_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_302_; 
v_a_295_ = lean_ctor_get(v___x_279_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_279_);
if (v_isSharedCheck_302_ == 0)
{
v___x_297_ = v___x_279_;
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
else
{
lean_inc(v_a_295_);
lean_dec(v___x_279_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v___x_300_; 
if (v_isShared_298_ == 0)
{
v___x_300_ = v___x_297_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_a_295_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
else
{
lean_object* v_vs_303_; lean_object* v___x_304_; lean_object* v___x_305_; size_t v_sz_306_; size_t v___x_307_; lean_object* v___x_308_; 
v_vs_303_ = lean_ctor_get(v_n_267_, 0);
v___x_304_ = lean_box(0);
v___x_305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_b_268_);
v_sz_306_ = lean_array_size(v_vs_303_);
v___x_307_ = ((size_t)0ULL);
v___x_308_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4(v_vs_303_, v_sz_306_, v___x_307_, v___x_305_, v___y_269_, v___y_270_, v___y_271_, v___y_272_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_323_; 
v_a_309_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_323_ == 0)
{
v___x_311_ = v___x_308_;
v_isShared_312_ = v_isSharedCheck_323_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_308_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_323_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v_fst_313_; 
v_fst_313_ = lean_ctor_get(v_a_309_, 0);
if (lean_obj_tag(v_fst_313_) == 0)
{
lean_object* v_snd_314_; lean_object* v___x_315_; lean_object* v___x_317_; 
v_snd_314_ = lean_ctor_get(v_a_309_, 1);
lean_inc(v_snd_314_);
lean_dec(v_a_309_);
v___x_315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_315_, 0, v_snd_314_);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 0, v___x_315_);
v___x_317_ = v___x_311_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v___x_315_);
v___x_317_ = v_reuseFailAlloc_318_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
return v___x_317_;
}
}
else
{
lean_object* v_val_319_; lean_object* v___x_321_; 
lean_inc_ref(v_fst_313_);
lean_dec(v_a_309_);
v_val_319_ = lean_ctor_get(v_fst_313_, 0);
lean_inc(v_val_319_);
lean_dec_ref_known(v_fst_313_, 1);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 0, v_val_319_);
v___x_321_ = v___x_311_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_val_319_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
else
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_331_; 
v_a_324_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_331_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_331_ == 0)
{
v___x_326_ = v___x_308_;
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_308_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_329_; 
if (v_isShared_327_ == 0)
{
v___x_329_ = v___x_326_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v_a_324_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3(lean_object* v_init_332_, lean_object* v_as_333_, size_t v_sz_334_, size_t v_i_335_, lean_object* v_b_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
uint8_t v___x_342_; 
v___x_342_ = lean_usize_dec_lt(v_i_335_, v_sz_334_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; 
v___x_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_343_, 0, v_b_336_);
return v___x_343_;
}
else
{
lean_object* v_snd_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_378_; 
v_snd_344_ = lean_ctor_get(v_b_336_, 1);
v_isSharedCheck_378_ = !lean_is_exclusive(v_b_336_);
if (v_isSharedCheck_378_ == 0)
{
lean_object* v_unused_379_; 
v_unused_379_ = lean_ctor_get(v_b_336_, 0);
lean_dec(v_unused_379_);
v___x_346_ = v_b_336_;
v_isShared_347_ = v_isSharedCheck_378_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_snd_344_);
lean_dec(v_b_336_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_378_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v_a_348_; lean_object* v___x_349_; 
v_a_348_ = lean_array_uget_borrowed(v_as_333_, v_i_335_);
lean_inc(v_snd_344_);
v___x_349_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0(v_init_332_, v_a_348_, v_snd_344_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
if (lean_obj_tag(v___x_349_) == 0)
{
lean_object* v_a_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_369_; 
v_a_350_ = lean_ctor_get(v___x_349_, 0);
v_isSharedCheck_369_ = !lean_is_exclusive(v___x_349_);
if (v_isSharedCheck_369_ == 0)
{
v___x_352_ = v___x_349_;
v_isShared_353_ = v_isSharedCheck_369_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_a_350_);
lean_dec(v___x_349_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_369_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
if (lean_obj_tag(v_a_350_) == 0)
{
lean_object* v___x_354_; lean_object* v___x_356_; 
v___x_354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_354_, 0, v_a_350_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 0, v___x_354_);
v___x_356_ = v___x_346_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_354_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_snd_344_);
v___x_356_ = v_reuseFailAlloc_360_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
lean_object* v___x_358_; 
if (v_isShared_353_ == 0)
{
lean_ctor_set(v___x_352_, 0, v___x_356_);
v___x_358_ = v___x_352_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_356_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
}
else
{
lean_object* v_a_361_; lean_object* v___x_362_; lean_object* v___x_364_; 
lean_del_object(v___x_352_);
lean_dec(v_snd_344_);
v_a_361_ = lean_ctor_get(v_a_350_, 0);
lean_inc(v_a_361_);
lean_dec_ref_known(v_a_350_, 1);
v___x_362_ = lean_box(0);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 1, v_a_361_);
lean_ctor_set(v___x_346_, 0, v___x_362_);
v___x_364_ = v___x_346_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_368_; 
v_reuseFailAlloc_368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_368_, 0, v___x_362_);
lean_ctor_set(v_reuseFailAlloc_368_, 1, v_a_361_);
v___x_364_ = v_reuseFailAlloc_368_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
size_t v___x_365_; size_t v___x_366_; 
v___x_365_ = ((size_t)1ULL);
v___x_366_ = lean_usize_add(v_i_335_, v___x_365_);
v_i_335_ = v___x_366_;
v_b_336_ = v___x_364_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_377_; 
lean_del_object(v___x_346_);
lean_dec(v_snd_344_);
v_a_370_ = lean_ctor_get(v___x_349_, 0);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_349_);
if (v_isSharedCheck_377_ == 0)
{
v___x_372_ = v___x_349_;
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_a_370_);
lean_dec(v___x_349_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_375_; 
if (v_isShared_373_ == 0)
{
v___x_375_ = v___x_372_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_a_370_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3___boxed(lean_object* v_init_380_, lean_object* v_as_381_, lean_object* v_sz_382_, lean_object* v_i_383_, lean_object* v_b_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_){
_start:
{
size_t v_sz_boxed_390_; size_t v_i_boxed_391_; lean_object* v_res_392_; 
v_sz_boxed_390_ = lean_unbox_usize(v_sz_382_);
lean_dec(v_sz_382_);
v_i_boxed_391_ = lean_unbox_usize(v_i_383_);
lean_dec(v_i_383_);
v_res_392_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__3(v_init_380_, v_as_381_, v_sz_boxed_390_, v_i_boxed_391_, v_b_384_, v___y_385_, v___y_386_, v___y_387_, v___y_388_);
lean_dec(v___y_388_);
lean_dec_ref(v___y_387_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec_ref(v_as_381_);
lean_dec_ref(v_init_380_);
return v_res_392_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0___boxed(lean_object* v_init_393_, lean_object* v_n_394_, lean_object* v_b_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0(v_init_393_, v_n_394_, v_b_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v___y_397_);
lean_dec_ref(v___y_396_);
lean_dec_ref(v_n_394_);
lean_dec_ref(v_init_393_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0(lean_object* v_t_402_, lean_object* v_init_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_){
_start:
{
lean_object* v_root_409_; lean_object* v_tail_410_; lean_object* v___x_411_; 
v_root_409_ = lean_ctor_get(v_t_402_, 0);
v_tail_410_ = lean_ctor_get(v_t_402_, 1);
lean_inc_ref(v_init_403_);
v___x_411_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0(v_init_403_, v_root_409_, v_init_403_, v___y_404_, v___y_405_, v___y_406_, v___y_407_);
lean_dec_ref(v_init_403_);
if (lean_obj_tag(v___x_411_) == 0)
{
lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_448_; 
v_a_412_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_448_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_448_ == 0)
{
v___x_414_ = v___x_411_;
v_isShared_415_ = v_isSharedCheck_448_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_411_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_448_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
if (lean_obj_tag(v_a_412_) == 0)
{
lean_object* v_a_416_; lean_object* v___x_418_; 
v_a_416_ = lean_ctor_get(v_a_412_, 0);
lean_inc(v_a_416_);
lean_dec_ref_known(v_a_412_, 1);
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 0, v_a_416_);
v___x_418_ = v___x_414_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_a_416_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
else
{
lean_object* v_a_420_; lean_object* v___x_421_; lean_object* v___x_422_; size_t v_sz_423_; size_t v___x_424_; lean_object* v___x_425_; 
lean_del_object(v___x_414_);
v_a_420_ = lean_ctor_get(v_a_412_, 0);
lean_inc(v_a_420_);
lean_dec_ref_known(v_a_412_, 1);
v___x_421_ = lean_box(0);
v___x_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v_a_420_);
v_sz_423_ = lean_array_size(v_tail_410_);
v___x_424_ = ((size_t)0ULL);
v___x_425_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1(v_tail_410_, v_sz_423_, v___x_424_, v___x_422_, v___y_404_, v___y_405_, v___y_406_, v___y_407_);
if (lean_obj_tag(v___x_425_) == 0)
{
lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_439_; 
v_a_426_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_439_ == 0)
{
v___x_428_ = v___x_425_;
v_isShared_429_ = v_isSharedCheck_439_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_425_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_439_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v_fst_430_; 
v_fst_430_ = lean_ctor_get(v_a_426_, 0);
if (lean_obj_tag(v_fst_430_) == 0)
{
lean_object* v_snd_431_; lean_object* v___x_433_; 
v_snd_431_ = lean_ctor_get(v_a_426_, 1);
lean_inc(v_snd_431_);
lean_dec(v_a_426_);
if (v_isShared_429_ == 0)
{
lean_ctor_set(v___x_428_, 0, v_snd_431_);
v___x_433_ = v___x_428_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_snd_431_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
else
{
lean_object* v_val_435_; lean_object* v___x_437_; 
lean_inc_ref(v_fst_430_);
lean_dec(v_a_426_);
v_val_435_ = lean_ctor_get(v_fst_430_, 0);
lean_inc(v_val_435_);
lean_dec_ref_known(v_fst_430_, 1);
if (v_isShared_429_ == 0)
{
lean_ctor_set(v___x_428_, 0, v_val_435_);
v___x_437_ = v___x_428_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_val_435_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
else
{
lean_object* v_a_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_447_; 
v_a_440_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_447_ == 0)
{
v___x_442_ = v___x_425_;
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_a_440_);
lean_dec(v___x_425_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_a_440_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
}
}
else
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_456_; 
v_a_449_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_456_ == 0)
{
v___x_451_ = v___x_411_;
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_411_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_456_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_454_; 
if (v_isShared_452_ == 0)
{
v___x_454_ = v___x_451_;
goto v_reusejp_453_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v_a_449_);
v___x_454_ = v_reuseFailAlloc_455_;
goto v_reusejp_453_;
}
v_reusejp_453_:
{
return v___x_454_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0___boxed(lean_object* v_t_457_, lean_object* v_init_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_){
_start:
{
lean_object* v_res_464_; 
v_res_464_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0(v_t_457_, v_init_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
lean_dec(v___y_460_);
lean_dec_ref(v___y_459_);
lean_dec_ref(v_t_457_);
return v_res_464_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0(void){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_465_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1(void){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__1(lean_box(0));
return v___x_466_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2(void){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_addLetDeclsToSimpTheorems_spec__2(lean_box(0));
return v___x_467_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3(void){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_468_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4(void){
_start:
{
lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_469_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__3);
v___x_470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_470_, 0, v___x_469_);
return v___x_470_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5(void){
_start:
{
lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_471_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__4);
v___x_472_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__2);
v___x_473_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__1);
v___x_474_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__0);
v___x_475_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v___x_474_);
lean_ctor_set(v___x_475_, 2, v___x_473_);
lean_ctor_set(v___x_475_, 3, v___x_472_);
lean_ctor_set(v___x_475_, 4, v___x_473_);
lean_ctor_set(v___x_475_, 5, v___x_471_);
return v___x_475_;
}
}
static lean_object* _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v_simpTheoremsArray_479_; 
v___x_476_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__5);
v___x_477_ = lean_unsigned_to_nat(1u);
v___x_478_ = lean_mk_empty_array_with_capacity(v___x_477_);
v_simpTheoremsArray_479_ = lean_array_push(v___x_478_, v___x_476_);
return v_simpTheoremsArray_479_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems(lean_object* v_ctx_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_){
_start:
{
lean_object* v_simpTheoremsArray_487_; lean_object* v___y_488_; lean_object* v___y_489_; lean_object* v___y_490_; lean_object* v___y_491_; lean_object* v_simpTheorems_512_; lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; 
v_simpTheorems_512_ = lean_ctor_get(v_ctx_480_, 6);
v___x_513_ = lean_array_get_size(v_simpTheorems_512_);
v___x_514_ = lean_unsigned_to_nat(0u);
v___x_515_ = lean_nat_dec_eq(v___x_513_, v___x_514_);
if (v___x_515_ == 0)
{
lean_inc_ref(v_simpTheorems_512_);
v_simpTheoremsArray_487_ = v_simpTheorems_512_;
v___y_488_ = v_a_481_;
v___y_489_ = v_a_482_;
v___y_490_ = v_a_483_;
v___y_491_ = v_a_484_;
goto v___jp_486_;
}
else
{
lean_object* v_simpTheoremsArray_516_; 
v_simpTheoremsArray_516_ = lean_obj_once(&lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6, &lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6_once, _init_lp_aesop_Aesop_addLetDeclsToSimpTheorems___closed__6);
v_simpTheoremsArray_487_ = v_simpTheoremsArray_516_;
v___y_488_ = v_a_481_;
v___y_489_ = v_a_482_;
v___y_490_ = v_a_483_;
v___y_491_ = v_a_484_;
goto v___jp_486_;
}
v___jp_486_:
{
lean_object* v_lctx_492_; lean_object* v_decls_493_; lean_object* v___x_494_; 
v_lctx_492_ = lean_ctor_get(v___y_488_, 2);
v_decls_493_ = lean_ctor_get(v_lctx_492_, 1);
v___x_494_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0(v_decls_493_, v_simpTheoremsArray_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_503_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_503_ == 0)
{
v___x_497_ = v___x_494_;
v_isShared_498_ = v_isSharedCheck_503_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_494_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_503_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_499_; lean_object* v___x_501_; 
v___x_499_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v_ctx_480_, v_a_495_);
if (v_isShared_498_ == 0)
{
lean_ctor_set(v___x_497_, 0, v___x_499_);
v___x_501_ = v___x_497_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_499_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
else
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
lean_dec_ref(v_ctx_480_);
v_a_504_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_494_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_494_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheorems___boxed(lean_object* v_ctx_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_aesop_Aesop_addLetDeclsToSimpTheorems(v_ctx_517_, v_a_518_, v_a_519_, v_a_520_, v_a_521_);
lean_dec(v_a_521_);
lean_dec_ref(v_a_520_);
lean_dec(v_a_519_);
lean_dec_ref(v_a_518_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6(lean_object* v_as_524_, size_t v_sz_525_, size_t v_i_526_, lean_object* v_b_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___redArg(v_as_524_, v_sz_525_, v_i_526_, v_b_527_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6___boxed(lean_object* v_as_534_, lean_object* v_sz_535_, lean_object* v_i_536_, lean_object* v_b_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
size_t v_sz_boxed_543_; size_t v_i_boxed_544_; lean_object* v_res_545_; 
v_sz_boxed_543_ = lean_unbox_usize(v_sz_535_);
lean_dec(v_sz_535_);
v_i_boxed_544_ = lean_unbox_usize(v_i_536_);
lean_dec(v_i_536_);
v_res_545_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__1_spec__6(v_as_534_, v_sz_boxed_543_, v_i_boxed_544_, v_b_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec(v___y_539_);
lean_dec_ref(v___y_538_);
lean_dec_ref(v_as_534_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5(lean_object* v_as_546_, size_t v_sz_547_, size_t v_i_548_, lean_object* v_b_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___redArg(v_as_546_, v_sz_547_, v_i_548_, v_b_549_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5___boxed(lean_object* v_as_556_, lean_object* v_sz_557_, lean_object* v_i_558_, lean_object* v_b_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_){
_start:
{
size_t v_sz_boxed_565_; size_t v_i_boxed_566_; lean_object* v_res_567_; 
v_sz_boxed_565_ = lean_unbox_usize(v_sz_557_);
lean_dec(v_sz_557_);
v_i_boxed_566_ = lean_unbox_usize(v_i_558_);
lean_dec(v_i_558_);
v_res_567_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_addLetDeclsToSimpTheorems_spec__0_spec__0_spec__4_spec__5(v_as_556_, v_sz_boxed_565_, v_i_boxed_566_, v_b_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
lean_dec(v___y_563_);
lean_dec_ref(v___y_562_);
lean_dec(v___y_561_);
lean_dec_ref(v___y_560_);
lean_dec_ref(v_as_556_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta(lean_object* v_ctx_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_){
_start:
{
lean_object* v_config_574_; uint8_t v_zetaDelta_575_; 
v_config_574_ = lean_ctor_get(v_ctx_568_, 0);
v_zetaDelta_575_ = lean_ctor_get_uint8(v_config_574_, sizeof(void*)*3 + 16);
if (v_zetaDelta_575_ == 0)
{
lean_object* v___x_576_; 
v___x_576_ = lp_aesop_Aesop_addLetDeclsToSimpTheorems(v_ctx_568_, v_a_569_, v_a_570_, v_a_571_, v_a_572_);
return v___x_576_;
}
else
{
lean_object* v___x_577_; 
v___x_577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_577_, 0, v_ctx_568_);
return v___x_577_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta___boxed(lean_object* v_ctx_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta(v_ctx_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
lean_dec(v_a_582_);
lean_dec_ref(v_a_581_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoal(lean_object* v_mvarId_585_, lean_object* v_ctx_586_, lean_object* v_simprocs_587_, lean_object* v_discharge_x3f_588_, uint8_t v_simplifyTarget_589_, lean_object* v_fvarIdsToSimp_590_, lean_object* v_stats_591_, lean_object* v_a_592_, lean_object* v_a_593_, lean_object* v_a_594_, lean_object* v_a_595_){
_start:
{
uint8_t v___x_597_; lean_object* v_ctx_598_; lean_object* v___x_599_; 
v___x_597_ = 0;
v_ctx_598_ = l_Lean_Meta_Simp_Context_setFailIfUnchanged(v_ctx_586_, v___x_597_);
lean_inc(v_mvarId_585_);
v___x_599_ = l_Lean_Meta_simpGoal(v_mvarId_585_, v_ctx_598_, v_simprocs_587_, v_discharge_x3f_588_, v_simplifyTarget_589_, v_fvarIdsToSimp_590_, v_stats_591_, v_a_592_, v_a_593_, v_a_594_, v_a_595_);
if (lean_obj_tag(v___x_599_) == 0)
{
lean_object* v_a_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_630_; 
v_a_600_ = lean_ctor_get(v___x_599_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_599_);
if (v_isSharedCheck_630_ == 0)
{
v___x_602_ = v___x_599_;
v_isShared_603_ = v_isSharedCheck_630_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_a_600_);
lean_dec(v___x_599_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_630_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v_snd_604_; lean_object* v_fst_605_; 
v_snd_604_ = lean_ctor_get(v_a_600_, 1);
lean_inc(v_snd_604_);
v_fst_605_ = lean_ctor_get(v_a_600_, 0);
lean_inc(v_fst_605_);
lean_dec(v_a_600_);
if (lean_obj_tag(v_fst_605_) == 1)
{
lean_object* v_val_606_; lean_object* v_usedTheorems_607_; lean_object* v_snd_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_623_; 
v_val_606_ = lean_ctor_get(v_fst_605_, 0);
lean_inc(v_val_606_);
lean_dec_ref_known(v_fst_605_, 1);
v_usedTheorems_607_ = lean_ctor_get(v_snd_604_, 0);
lean_inc_ref(v_usedTheorems_607_);
lean_dec(v_snd_604_);
v_snd_608_ = lean_ctor_get(v_val_606_, 1);
v_isSharedCheck_623_ = !lean_is_exclusive(v_val_606_);
if (v_isSharedCheck_623_ == 0)
{
lean_object* v_unused_624_; 
v_unused_624_ = lean_ctor_get(v_val_606_, 0);
lean_dec(v_unused_624_);
v___x_610_ = v_val_606_;
v_isShared_611_ = v_isSharedCheck_623_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_snd_608_);
lean_dec(v_val_606_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_623_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
uint8_t v___x_612_; 
v___x_612_ = l_Lean_instBEqMVarId_beq(v_snd_608_, v_mvarId_585_);
lean_dec(v_mvarId_585_);
if (v___x_612_ == 0)
{
lean_object* v___x_614_; 
if (v_isShared_611_ == 0)
{
lean_ctor_set_tag(v___x_610_, 2);
lean_ctor_set(v___x_610_, 1, v_usedTheorems_607_);
lean_ctor_set(v___x_610_, 0, v_snd_608_);
v___x_614_ = v___x_610_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v_snd_608_);
lean_ctor_set(v_reuseFailAlloc_618_, 1, v_usedTheorems_607_);
v___x_614_ = v_reuseFailAlloc_618_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
lean_object* v___x_616_; 
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_614_);
v___x_616_ = v___x_602_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v___x_614_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
}
else
{
lean_object* v___x_619_; lean_object* v___x_621_; 
lean_del_object(v___x_610_);
lean_dec(v_snd_608_);
lean_dec_ref(v_usedTheorems_607_);
v___x_619_ = lean_box(1);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_619_);
v___x_621_ = v___x_602_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v___x_619_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
}
else
{
lean_object* v_usedTheorems_625_; lean_object* v___x_626_; lean_object* v___x_628_; 
lean_dec(v_fst_605_);
lean_dec(v_mvarId_585_);
v_usedTheorems_625_ = lean_ctor_get(v_snd_604_, 0);
lean_inc_ref(v_usedTheorems_625_);
lean_dec(v_snd_604_);
v___x_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_626_, 0, v_usedTheorems_625_);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_626_);
v___x_628_ = v___x_602_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v___x_626_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
else
{
lean_object* v_a_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_638_; 
lean_dec(v_mvarId_585_);
v_a_631_ = lean_ctor_get(v___x_599_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v___x_599_);
if (v_isSharedCheck_638_ == 0)
{
v___x_633_ = v___x_599_;
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_a_631_);
lean_dec(v___x_599_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoal___boxed(lean_object* v_mvarId_639_, lean_object* v_ctx_640_, lean_object* v_simprocs_641_, lean_object* v_discharge_x3f_642_, lean_object* v_simplifyTarget_643_, lean_object* v_fvarIdsToSimp_644_, lean_object* v_stats_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
uint8_t v_simplifyTarget_boxed_651_; lean_object* v_res_652_; 
v_simplifyTarget_boxed_651_ = lean_unbox(v_simplifyTarget_643_);
v_res_652_ = lp_aesop_Aesop_simpGoal(v_mvarId_639_, v_ctx_640_, v_simprocs_641_, v_discharge_x3f_642_, v_simplifyTarget_boxed_651_, v_fvarIdsToSimp_644_, v_stats_645_, v_a_646_, v_a_647_, v_a_648_, v_a_649_);
lean_dec(v_a_649_);
lean_dec_ref(v_a_648_);
lean_dec(v_a_647_);
lean_dec_ref(v_a_646_);
return v_res_652_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(lean_object* v_mvarId_653_, lean_object* v_x_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_653_, v_x_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
if (lean_obj_tag(v___x_660_) == 0)
{
lean_object* v_a_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_668_; 
v_a_661_ = lean_ctor_get(v___x_660_, 0);
v_isSharedCheck_668_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_668_ == 0)
{
v___x_663_ = v___x_660_;
v_isShared_664_ = v_isSharedCheck_668_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_a_661_);
lean_dec(v___x_660_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_668_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___x_666_; 
if (v_isShared_664_ == 0)
{
v___x_666_ = v___x_663_;
goto v_reusejp_665_;
}
else
{
lean_object* v_reuseFailAlloc_667_; 
v_reuseFailAlloc_667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_667_, 0, v_a_661_);
v___x_666_ = v_reuseFailAlloc_667_;
goto v_reusejp_665_;
}
v_reusejp_665_:
{
return v___x_666_;
}
}
}
else
{
lean_object* v_a_669_; lean_object* v___x_671_; uint8_t v_isShared_672_; uint8_t v_isSharedCheck_676_; 
v_a_669_ = lean_ctor_get(v___x_660_, 0);
v_isSharedCheck_676_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_676_ == 0)
{
v___x_671_ = v___x_660_;
v_isShared_672_ = v_isSharedCheck_676_;
goto v_resetjp_670_;
}
else
{
lean_inc(v_a_669_);
lean_dec(v___x_660_);
v___x_671_ = lean_box(0);
v_isShared_672_ = v_isSharedCheck_676_;
goto v_resetjp_670_;
}
v_resetjp_670_:
{
lean_object* v___x_674_; 
if (v_isShared_672_ == 0)
{
v___x_674_ = v___x_671_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_675_; 
v_reuseFailAlloc_675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_675_, 0, v_a_669_);
v___x_674_ = v_reuseFailAlloc_675_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
return v___x_674_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg___boxed(lean_object* v_mvarId_677_, lean_object* v_x_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(v_mvarId_677_, v_x_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1(lean_object* v_00_u03b1_685_, lean_object* v_mvarId_686_, lean_object* v_x_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(v_mvarId_686_, v_x_687_, v___y_688_, v___y_689_, v___y_690_, v___y_691_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___boxed(lean_object* v_00_u03b1_694_, lean_object* v_mvarId_695_, lean_object* v_x_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1(v_00_u03b1_694_, v_mvarId_695_, v_x_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg(lean_object* v_as_703_, size_t v_sz_704_, size_t v_i_705_, lean_object* v_b_706_){
_start:
{
uint8_t v___x_708_; 
v___x_708_ = lean_usize_dec_lt(v_i_705_, v_sz_704_);
if (v___x_708_ == 0)
{
lean_object* v___x_709_; 
v___x_709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_709_, 0, v_b_706_);
return v___x_709_;
}
else
{
lean_object* v_snd_710_; lean_object* v___x_712_; uint8_t v_isShared_713_; uint8_t v_isSharedCheck_728_; 
v_snd_710_ = lean_ctor_get(v_b_706_, 1);
v_isSharedCheck_728_ = !lean_is_exclusive(v_b_706_);
if (v_isSharedCheck_728_ == 0)
{
lean_object* v_unused_729_; 
v_unused_729_ = lean_ctor_get(v_b_706_, 0);
lean_dec(v_unused_729_);
v___x_712_ = v_b_706_;
v_isShared_713_ = v_isSharedCheck_728_;
goto v_resetjp_711_;
}
else
{
lean_inc(v_snd_710_);
lean_dec(v_b_706_);
v___x_712_ = lean_box(0);
v_isShared_713_ = v_isSharedCheck_728_;
goto v_resetjp_711_;
}
v_resetjp_711_:
{
lean_object* v___x_714_; lean_object* v_a_716_; lean_object* v_a_723_; 
v___x_714_ = lean_box(0);
v_a_723_ = lean_array_uget_borrowed(v_as_703_, v_i_705_);
if (lean_obj_tag(v_a_723_) == 0)
{
v_a_716_ = v_snd_710_;
goto v___jp_715_;
}
else
{
lean_object* v_val_724_; uint8_t v___x_725_; 
v_val_724_ = lean_ctor_get(v_a_723_, 0);
v___x_725_ = l_Lean_LocalDecl_isImplementationDetail(v_val_724_);
if (v___x_725_ == 0)
{
lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_726_ = l_Lean_LocalDecl_fvarId(v_val_724_);
v___x_727_ = lean_array_push(v_snd_710_, v___x_726_);
v_a_716_ = v___x_727_;
goto v___jp_715_;
}
else
{
v_a_716_ = v_snd_710_;
goto v___jp_715_;
}
}
v___jp_715_:
{
lean_object* v___x_718_; 
if (v_isShared_713_ == 0)
{
lean_ctor_set(v___x_712_, 1, v_a_716_);
lean_ctor_set(v___x_712_, 0, v___x_714_);
v___x_718_ = v___x_712_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_722_; 
v_reuseFailAlloc_722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_722_, 0, v___x_714_);
lean_ctor_set(v_reuseFailAlloc_722_, 1, v_a_716_);
v___x_718_ = v_reuseFailAlloc_722_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
size_t v___x_719_; size_t v___x_720_; 
v___x_719_ = ((size_t)1ULL);
v___x_720_ = lean_usize_add(v_i_705_, v___x_719_);
v_i_705_ = v___x_720_;
v_b_706_ = v___x_718_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object* v_as_730_, lean_object* v_sz_731_, lean_object* v_i_732_, lean_object* v_b_733_, lean_object* v___y_734_){
_start:
{
size_t v_sz_boxed_735_; size_t v_i_boxed_736_; lean_object* v_res_737_; 
v_sz_boxed_735_ = lean_unbox_usize(v_sz_731_);
lean_dec(v_sz_731_);
v_i_boxed_736_ = lean_unbox_usize(v_i_732_);
lean_dec(v_i_732_);
v_res_737_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg(v_as_730_, v_sz_boxed_735_, v_i_boxed_736_, v_b_733_);
lean_dec_ref(v_as_730_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3(lean_object* v_as_738_, size_t v_sz_739_, size_t v_i_740_, lean_object* v_b_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
uint8_t v___x_747_; 
v___x_747_ = lean_usize_dec_lt(v_i_740_, v_sz_739_);
if (v___x_747_ == 0)
{
lean_object* v___x_748_; 
v___x_748_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_748_, 0, v_b_741_);
return v___x_748_;
}
else
{
lean_object* v_snd_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_767_; 
v_snd_749_ = lean_ctor_get(v_b_741_, 1);
v_isSharedCheck_767_ = !lean_is_exclusive(v_b_741_);
if (v_isSharedCheck_767_ == 0)
{
lean_object* v_unused_768_; 
v_unused_768_ = lean_ctor_get(v_b_741_, 0);
lean_dec(v_unused_768_);
v___x_751_ = v_b_741_;
v_isShared_752_ = v_isSharedCheck_767_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_snd_749_);
lean_dec(v_b_741_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_767_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
lean_object* v___x_753_; lean_object* v_a_755_; lean_object* v_a_762_; 
v___x_753_ = lean_box(0);
v_a_762_ = lean_array_uget_borrowed(v_as_738_, v_i_740_);
if (lean_obj_tag(v_a_762_) == 0)
{
v_a_755_ = v_snd_749_;
goto v___jp_754_;
}
else
{
lean_object* v_val_763_; uint8_t v___x_764_; 
v_val_763_ = lean_ctor_get(v_a_762_, 0);
v___x_764_ = l_Lean_LocalDecl_isImplementationDetail(v_val_763_);
if (v___x_764_ == 0)
{
lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_765_ = l_Lean_LocalDecl_fvarId(v_val_763_);
v___x_766_ = lean_array_push(v_snd_749_, v___x_765_);
v_a_755_ = v___x_766_;
goto v___jp_754_;
}
else
{
v_a_755_ = v_snd_749_;
goto v___jp_754_;
}
}
v___jp_754_:
{
lean_object* v___x_757_; 
if (v_isShared_752_ == 0)
{
lean_ctor_set(v___x_751_, 1, v_a_755_);
lean_ctor_set(v___x_751_, 0, v___x_753_);
v___x_757_ = v___x_751_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_761_; 
v_reuseFailAlloc_761_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_761_, 0, v___x_753_);
lean_ctor_set(v_reuseFailAlloc_761_, 1, v_a_755_);
v___x_757_ = v_reuseFailAlloc_761_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
size_t v___x_758_; size_t v___x_759_; lean_object* v___x_760_; 
v___x_758_ = ((size_t)1ULL);
v___x_759_ = lean_usize_add(v_i_740_, v___x_758_);
v___x_760_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg(v_as_738_, v_sz_739_, v___x_759_, v___x_757_);
return v___x_760_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3___boxed(lean_object* v_as_769_, lean_object* v_sz_770_, lean_object* v_i_771_, lean_object* v_b_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
size_t v_sz_boxed_778_; size_t v_i_boxed_779_; lean_object* v_res_780_; 
v_sz_boxed_778_ = lean_unbox_usize(v_sz_770_);
lean_dec(v_sz_770_);
v_i_boxed_779_ = lean_unbox_usize(v_i_771_);
lean_dec(v_i_771_);
v_res_780_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3(v_as_769_, v_sz_boxed_778_, v_i_boxed_779_, v_b_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
lean_dec_ref(v_as_769_);
return v_res_780_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0(lean_object* v_init_781_, lean_object* v_n_782_, lean_object* v_b_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
if (lean_obj_tag(v_n_782_) == 0)
{
lean_object* v_cs_789_; lean_object* v___x_790_; lean_object* v___x_791_; size_t v_sz_792_; size_t v___x_793_; lean_object* v___x_794_; 
v_cs_789_ = lean_ctor_get(v_n_782_, 0);
v___x_790_ = lean_box(0);
v___x_791_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_791_, 0, v___x_790_);
lean_ctor_set(v___x_791_, 1, v_b_783_);
v_sz_792_ = lean_array_size(v_cs_789_);
v___x_793_ = ((size_t)0ULL);
v___x_794_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2(v_init_781_, v_cs_789_, v_sz_792_, v___x_793_, v___x_791_, v___y_784_, v___y_785_, v___y_786_, v___y_787_);
if (lean_obj_tag(v___x_794_) == 0)
{
lean_object* v_a_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_809_; 
v_a_795_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_809_ == 0)
{
v___x_797_ = v___x_794_;
v_isShared_798_ = v_isSharedCheck_809_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_a_795_);
lean_dec(v___x_794_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_809_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v_fst_799_; 
v_fst_799_ = lean_ctor_get(v_a_795_, 0);
if (lean_obj_tag(v_fst_799_) == 0)
{
lean_object* v_snd_800_; lean_object* v___x_801_; lean_object* v___x_803_; 
v_snd_800_ = lean_ctor_get(v_a_795_, 1);
lean_inc(v_snd_800_);
lean_dec(v_a_795_);
v___x_801_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_801_, 0, v_snd_800_);
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v___x_801_);
v___x_803_ = v___x_797_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v___x_801_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
else
{
lean_object* v_val_805_; lean_object* v___x_807_; 
lean_inc_ref(v_fst_799_);
lean_dec(v_a_795_);
v_val_805_ = lean_ctor_get(v_fst_799_, 0);
lean_inc(v_val_805_);
lean_dec_ref_known(v_fst_799_, 1);
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v_val_805_);
v___x_807_ = v___x_797_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v_val_805_);
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
else
{
lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_817_; 
v_a_810_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_817_ == 0)
{
v___x_812_ = v___x_794_;
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_794_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_815_; 
if (v_isShared_813_ == 0)
{
v___x_815_ = v___x_812_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v_a_810_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
}
}
else
{
lean_object* v_vs_818_; lean_object* v___x_819_; lean_object* v___x_820_; size_t v_sz_821_; size_t v___x_822_; lean_object* v___x_823_; 
v_vs_818_ = lean_ctor_get(v_n_782_, 0);
v___x_819_ = lean_box(0);
v___x_820_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
lean_ctor_set(v___x_820_, 1, v_b_783_);
v_sz_821_ = lean_array_size(v_vs_818_);
v___x_822_ = ((size_t)0ULL);
v___x_823_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3(v_vs_818_, v_sz_821_, v___x_822_, v___x_820_, v___y_784_, v___y_785_, v___y_786_, v___y_787_);
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v_a_824_; lean_object* v___x_826_; uint8_t v_isShared_827_; uint8_t v_isSharedCheck_838_; 
v_a_824_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_838_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_838_ == 0)
{
v___x_826_ = v___x_823_;
v_isShared_827_ = v_isSharedCheck_838_;
goto v_resetjp_825_;
}
else
{
lean_inc(v_a_824_);
lean_dec(v___x_823_);
v___x_826_ = lean_box(0);
v_isShared_827_ = v_isSharedCheck_838_;
goto v_resetjp_825_;
}
v_resetjp_825_:
{
lean_object* v_fst_828_; 
v_fst_828_ = lean_ctor_get(v_a_824_, 0);
if (lean_obj_tag(v_fst_828_) == 0)
{
lean_object* v_snd_829_; lean_object* v___x_830_; lean_object* v___x_832_; 
v_snd_829_ = lean_ctor_get(v_a_824_, 1);
lean_inc(v_snd_829_);
lean_dec(v_a_824_);
v___x_830_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_830_, 0, v_snd_829_);
if (v_isShared_827_ == 0)
{
lean_ctor_set(v___x_826_, 0, v___x_830_);
v___x_832_ = v___x_826_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v___x_830_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
else
{
lean_object* v_val_834_; lean_object* v___x_836_; 
lean_inc_ref(v_fst_828_);
lean_dec(v_a_824_);
v_val_834_ = lean_ctor_get(v_fst_828_, 0);
lean_inc(v_val_834_);
lean_dec_ref_known(v_fst_828_, 1);
if (v_isShared_827_ == 0)
{
lean_ctor_set(v___x_826_, 0, v_val_834_);
v___x_836_ = v___x_826_;
goto v_reusejp_835_;
}
else
{
lean_object* v_reuseFailAlloc_837_; 
v_reuseFailAlloc_837_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_837_, 0, v_val_834_);
v___x_836_ = v_reuseFailAlloc_837_;
goto v_reusejp_835_;
}
v_reusejp_835_:
{
return v___x_836_;
}
}
}
}
else
{
lean_object* v_a_839_; lean_object* v___x_841_; uint8_t v_isShared_842_; uint8_t v_isSharedCheck_846_; 
v_a_839_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_846_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_846_ == 0)
{
v___x_841_ = v___x_823_;
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
else
{
lean_inc(v_a_839_);
lean_dec(v___x_823_);
v___x_841_ = lean_box(0);
v_isShared_842_ = v_isSharedCheck_846_;
goto v_resetjp_840_;
}
v_resetjp_840_:
{
lean_object* v___x_844_; 
if (v_isShared_842_ == 0)
{
v___x_844_ = v___x_841_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_a_839_);
v___x_844_ = v_reuseFailAlloc_845_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
return v___x_844_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2(lean_object* v_init_847_, lean_object* v_as_848_, size_t v_sz_849_, size_t v_i_850_, lean_object* v_b_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_){
_start:
{
uint8_t v___x_857_; 
v___x_857_ = lean_usize_dec_lt(v_i_850_, v_sz_849_);
if (v___x_857_ == 0)
{
lean_object* v___x_858_; 
v___x_858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_858_, 0, v_b_851_);
return v___x_858_;
}
else
{
lean_object* v_snd_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_893_; 
v_snd_859_ = lean_ctor_get(v_b_851_, 1);
v_isSharedCheck_893_ = !lean_is_exclusive(v_b_851_);
if (v_isSharedCheck_893_ == 0)
{
lean_object* v_unused_894_; 
v_unused_894_ = lean_ctor_get(v_b_851_, 0);
lean_dec(v_unused_894_);
v___x_861_ = v_b_851_;
v_isShared_862_ = v_isSharedCheck_893_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_snd_859_);
lean_dec(v_b_851_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_893_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v_a_863_; lean_object* v___x_864_; 
v_a_863_ = lean_array_uget_borrowed(v_as_848_, v_i_850_);
lean_inc(v_snd_859_);
v___x_864_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0(v_init_847_, v_a_863_, v_snd_859_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
if (lean_obj_tag(v___x_864_) == 0)
{
lean_object* v_a_865_; lean_object* v___x_867_; uint8_t v_isShared_868_; uint8_t v_isSharedCheck_884_; 
v_a_865_ = lean_ctor_get(v___x_864_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_864_);
if (v_isSharedCheck_884_ == 0)
{
v___x_867_ = v___x_864_;
v_isShared_868_ = v_isSharedCheck_884_;
goto v_resetjp_866_;
}
else
{
lean_inc(v_a_865_);
lean_dec(v___x_864_);
v___x_867_ = lean_box(0);
v_isShared_868_ = v_isSharedCheck_884_;
goto v_resetjp_866_;
}
v_resetjp_866_:
{
if (lean_obj_tag(v_a_865_) == 0)
{
lean_object* v___x_869_; lean_object* v___x_871_; 
v___x_869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_869_, 0, v_a_865_);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 0, v___x_869_);
v___x_871_ = v___x_861_;
goto v_reusejp_870_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v___x_869_);
lean_ctor_set(v_reuseFailAlloc_875_, 1, v_snd_859_);
v___x_871_ = v_reuseFailAlloc_875_;
goto v_reusejp_870_;
}
v_reusejp_870_:
{
lean_object* v___x_873_; 
if (v_isShared_868_ == 0)
{
lean_ctor_set(v___x_867_, 0, v___x_871_);
v___x_873_ = v___x_867_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_871_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
else
{
lean_object* v_a_876_; lean_object* v___x_877_; lean_object* v___x_879_; 
lean_del_object(v___x_867_);
lean_dec(v_snd_859_);
v_a_876_ = lean_ctor_get(v_a_865_, 0);
lean_inc(v_a_876_);
lean_dec_ref_known(v_a_865_, 1);
v___x_877_ = lean_box(0);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 1, v_a_876_);
lean_ctor_set(v___x_861_, 0, v___x_877_);
v___x_879_ = v___x_861_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v___x_877_);
lean_ctor_set(v_reuseFailAlloc_883_, 1, v_a_876_);
v___x_879_ = v_reuseFailAlloc_883_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
size_t v___x_880_; size_t v___x_881_; 
v___x_880_ = ((size_t)1ULL);
v___x_881_ = lean_usize_add(v_i_850_, v___x_880_);
v_i_850_ = v___x_881_;
v_b_851_ = v___x_879_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_892_; 
lean_del_object(v___x_861_);
lean_dec(v_snd_859_);
v_a_885_ = lean_ctor_get(v___x_864_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_864_);
if (v_isSharedCheck_892_ == 0)
{
v___x_887_ = v___x_864_;
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_a_885_);
lean_dec(v___x_864_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_890_; 
if (v_isShared_888_ == 0)
{
v___x_890_ = v___x_887_;
goto v_reusejp_889_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v_a_885_);
v___x_890_ = v_reuseFailAlloc_891_;
goto v_reusejp_889_;
}
v_reusejp_889_:
{
return v___x_890_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2___boxed(lean_object* v_init_895_, lean_object* v_as_896_, lean_object* v_sz_897_, lean_object* v_i_898_, lean_object* v_b_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_){
_start:
{
size_t v_sz_boxed_905_; size_t v_i_boxed_906_; lean_object* v_res_907_; 
v_sz_boxed_905_ = lean_unbox_usize(v_sz_897_);
lean_dec(v_sz_897_);
v_i_boxed_906_ = lean_unbox_usize(v_i_898_);
lean_dec(v_i_898_);
v_res_907_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__2(v_init_895_, v_as_896_, v_sz_boxed_905_, v_i_boxed_906_, v_b_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
lean_dec(v___y_903_);
lean_dec_ref(v___y_902_);
lean_dec(v___y_901_);
lean_dec_ref(v___y_900_);
lean_dec_ref(v_as_896_);
lean_dec_ref(v_init_895_);
return v_res_907_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0___boxed(lean_object* v_init_908_, lean_object* v_n_909_, lean_object* v_b_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0(v_init_908_, v_n_909_, v_b_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_);
lean_dec(v___y_914_);
lean_dec_ref(v___y_913_);
lean_dec(v___y_912_);
lean_dec_ref(v___y_911_);
lean_dec_ref(v_n_909_);
lean_dec_ref(v_init_908_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg(lean_object* v_as_917_, size_t v_sz_918_, size_t v_i_919_, lean_object* v_b_920_){
_start:
{
uint8_t v___x_922_; 
v___x_922_ = lean_usize_dec_lt(v_i_919_, v_sz_918_);
if (v___x_922_ == 0)
{
lean_object* v___x_923_; 
v___x_923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_923_, 0, v_b_920_);
return v___x_923_;
}
else
{
lean_object* v_snd_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_942_; 
v_snd_924_ = lean_ctor_get(v_b_920_, 1);
v_isSharedCheck_942_ = !lean_is_exclusive(v_b_920_);
if (v_isSharedCheck_942_ == 0)
{
lean_object* v_unused_943_; 
v_unused_943_ = lean_ctor_get(v_b_920_, 0);
lean_dec(v_unused_943_);
v___x_926_ = v_b_920_;
v_isShared_927_ = v_isSharedCheck_942_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_snd_924_);
lean_dec(v_b_920_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_942_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___x_928_; lean_object* v_a_930_; lean_object* v_a_937_; 
v___x_928_ = lean_box(0);
v_a_937_ = lean_array_uget_borrowed(v_as_917_, v_i_919_);
if (lean_obj_tag(v_a_937_) == 0)
{
v_a_930_ = v_snd_924_;
goto v___jp_929_;
}
else
{
lean_object* v_val_938_; uint8_t v___x_939_; 
v_val_938_ = lean_ctor_get(v_a_937_, 0);
v___x_939_ = l_Lean_LocalDecl_isImplementationDetail(v_val_938_);
if (v___x_939_ == 0)
{
lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_940_ = l_Lean_LocalDecl_fvarId(v_val_938_);
v___x_941_ = lean_array_push(v_snd_924_, v___x_940_);
v_a_930_ = v___x_941_;
goto v___jp_929_;
}
else
{
v_a_930_ = v_snd_924_;
goto v___jp_929_;
}
}
v___jp_929_:
{
lean_object* v___x_932_; 
if (v_isShared_927_ == 0)
{
lean_ctor_set(v___x_926_, 1, v_a_930_);
lean_ctor_set(v___x_926_, 0, v___x_928_);
v___x_932_ = v___x_926_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v___x_928_);
lean_ctor_set(v_reuseFailAlloc_936_, 1, v_a_930_);
v___x_932_ = v_reuseFailAlloc_936_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
size_t v___x_933_; size_t v___x_934_; 
v___x_933_ = ((size_t)1ULL);
v___x_934_ = lean_usize_add(v_i_919_, v___x_933_);
v_i_919_ = v___x_934_;
v_b_920_ = v___x_932_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_as_944_, lean_object* v_sz_945_, lean_object* v_i_946_, lean_object* v_b_947_, lean_object* v___y_948_){
_start:
{
size_t v_sz_boxed_949_; size_t v_i_boxed_950_; lean_object* v_res_951_; 
v_sz_boxed_949_ = lean_unbox_usize(v_sz_945_);
lean_dec(v_sz_945_);
v_i_boxed_950_ = lean_unbox_usize(v_i_946_);
lean_dec(v_i_946_);
v_res_951_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg(v_as_944_, v_sz_boxed_949_, v_i_boxed_950_, v_b_947_);
lean_dec_ref(v_as_944_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1(lean_object* v_as_952_, size_t v_sz_953_, size_t v_i_954_, lean_object* v_b_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
uint8_t v___x_961_; 
v___x_961_ = lean_usize_dec_lt(v_i_954_, v_sz_953_);
if (v___x_961_ == 0)
{
lean_object* v___x_962_; 
v___x_962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_962_, 0, v_b_955_);
return v___x_962_;
}
else
{
lean_object* v_snd_963_; lean_object* v___x_965_; uint8_t v_isShared_966_; uint8_t v_isSharedCheck_981_; 
v_snd_963_ = lean_ctor_get(v_b_955_, 1);
v_isSharedCheck_981_ = !lean_is_exclusive(v_b_955_);
if (v_isSharedCheck_981_ == 0)
{
lean_object* v_unused_982_; 
v_unused_982_ = lean_ctor_get(v_b_955_, 0);
lean_dec(v_unused_982_);
v___x_965_ = v_b_955_;
v_isShared_966_ = v_isSharedCheck_981_;
goto v_resetjp_964_;
}
else
{
lean_inc(v_snd_963_);
lean_dec(v_b_955_);
v___x_965_ = lean_box(0);
v_isShared_966_ = v_isSharedCheck_981_;
goto v_resetjp_964_;
}
v_resetjp_964_:
{
lean_object* v___x_967_; lean_object* v_a_969_; lean_object* v_a_976_; 
v___x_967_ = lean_box(0);
v_a_976_ = lean_array_uget_borrowed(v_as_952_, v_i_954_);
if (lean_obj_tag(v_a_976_) == 0)
{
v_a_969_ = v_snd_963_;
goto v___jp_968_;
}
else
{
lean_object* v_val_977_; uint8_t v___x_978_; 
v_val_977_ = lean_ctor_get(v_a_976_, 0);
v___x_978_ = l_Lean_LocalDecl_isImplementationDetail(v_val_977_);
if (v___x_978_ == 0)
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = l_Lean_LocalDecl_fvarId(v_val_977_);
v___x_980_ = lean_array_push(v_snd_963_, v___x_979_);
v_a_969_ = v___x_980_;
goto v___jp_968_;
}
else
{
v_a_969_ = v_snd_963_;
goto v___jp_968_;
}
}
v___jp_968_:
{
lean_object* v___x_971_; 
if (v_isShared_966_ == 0)
{
lean_ctor_set(v___x_965_, 1, v_a_969_);
lean_ctor_set(v___x_965_, 0, v___x_967_);
v___x_971_ = v___x_965_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v___x_967_);
lean_ctor_set(v_reuseFailAlloc_975_, 1, v_a_969_);
v___x_971_ = v_reuseFailAlloc_975_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
size_t v___x_972_; size_t v___x_973_; lean_object* v___x_974_; 
v___x_972_ = ((size_t)1ULL);
v___x_973_ = lean_usize_add(v_i_954_, v___x_972_);
v___x_974_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg(v_as_952_, v_sz_953_, v___x_973_, v___x_971_);
return v___x_974_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1___boxed(lean_object* v_as_983_, lean_object* v_sz_984_, lean_object* v_i_985_, lean_object* v_b_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_){
_start:
{
size_t v_sz_boxed_992_; size_t v_i_boxed_993_; lean_object* v_res_994_; 
v_sz_boxed_992_ = lean_unbox_usize(v_sz_984_);
lean_dec(v_sz_984_);
v_i_boxed_993_ = lean_unbox_usize(v_i_985_);
lean_dec(v_i_985_);
v_res_994_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1(v_as_983_, v_sz_boxed_992_, v_i_boxed_993_, v_b_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_);
lean_dec(v___y_990_);
lean_dec_ref(v___y_989_);
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
lean_dec_ref(v_as_983_);
return v_res_994_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0(lean_object* v_t_995_, lean_object* v_init_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_){
_start:
{
lean_object* v_root_1002_; lean_object* v_tail_1003_; lean_object* v___x_1004_; 
v_root_1002_ = lean_ctor_get(v_t_995_, 0);
v_tail_1003_ = lean_ctor_get(v_t_995_, 1);
lean_inc_ref(v_init_996_);
v___x_1004_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0(v_init_996_, v_root_1002_, v_init_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_);
lean_dec_ref(v_init_996_);
if (lean_obj_tag(v___x_1004_) == 0)
{
lean_object* v_a_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1041_; 
v_a_1005_ = lean_ctor_get(v___x_1004_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_1004_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1007_ = v___x_1004_;
v_isShared_1008_ = v_isSharedCheck_1041_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_a_1005_);
lean_dec(v___x_1004_);
v___x_1007_ = lean_box(0);
v_isShared_1008_ = v_isSharedCheck_1041_;
goto v_resetjp_1006_;
}
v_resetjp_1006_:
{
if (lean_obj_tag(v_a_1005_) == 0)
{
lean_object* v_a_1009_; lean_object* v___x_1011_; 
v_a_1009_ = lean_ctor_get(v_a_1005_, 0);
lean_inc(v_a_1009_);
lean_dec_ref_known(v_a_1005_, 1);
if (v_isShared_1008_ == 0)
{
lean_ctor_set(v___x_1007_, 0, v_a_1009_);
v___x_1011_ = v___x_1007_;
goto v_reusejp_1010_;
}
else
{
lean_object* v_reuseFailAlloc_1012_; 
v_reuseFailAlloc_1012_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1012_, 0, v_a_1009_);
v___x_1011_ = v_reuseFailAlloc_1012_;
goto v_reusejp_1010_;
}
v_reusejp_1010_:
{
return v___x_1011_;
}
}
else
{
lean_object* v_a_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; size_t v_sz_1016_; size_t v___x_1017_; lean_object* v___x_1018_; 
lean_del_object(v___x_1007_);
v_a_1013_ = lean_ctor_get(v_a_1005_, 0);
lean_inc(v_a_1013_);
lean_dec_ref_known(v_a_1005_, 1);
v___x_1014_ = lean_box(0);
v___x_1015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
lean_ctor_set(v___x_1015_, 1, v_a_1013_);
v_sz_1016_ = lean_array_size(v_tail_1003_);
v___x_1017_ = ((size_t)0ULL);
v___x_1018_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1(v_tail_1003_, v_sz_1016_, v___x_1017_, v___x_1015_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_);
if (lean_obj_tag(v___x_1018_) == 0)
{
lean_object* v_a_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1032_; 
v_a_1019_ = lean_ctor_get(v___x_1018_, 0);
v_isSharedCheck_1032_ = !lean_is_exclusive(v___x_1018_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1021_ = v___x_1018_;
v_isShared_1022_ = v_isSharedCheck_1032_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_a_1019_);
lean_dec(v___x_1018_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1032_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v_fst_1023_; 
v_fst_1023_ = lean_ctor_get(v_a_1019_, 0);
if (lean_obj_tag(v_fst_1023_) == 0)
{
lean_object* v_snd_1024_; lean_object* v___x_1026_; 
v_snd_1024_ = lean_ctor_get(v_a_1019_, 1);
lean_inc(v_snd_1024_);
lean_dec(v_a_1019_);
if (v_isShared_1022_ == 0)
{
lean_ctor_set(v___x_1021_, 0, v_snd_1024_);
v___x_1026_ = v___x_1021_;
goto v_reusejp_1025_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1027_, 0, v_snd_1024_);
v___x_1026_ = v_reuseFailAlloc_1027_;
goto v_reusejp_1025_;
}
v_reusejp_1025_:
{
return v___x_1026_;
}
}
else
{
lean_object* v_val_1028_; lean_object* v___x_1030_; 
lean_inc_ref(v_fst_1023_);
lean_dec(v_a_1019_);
v_val_1028_ = lean_ctor_get(v_fst_1023_, 0);
lean_inc(v_val_1028_);
lean_dec_ref_known(v_fst_1023_, 1);
if (v_isShared_1022_ == 0)
{
lean_ctor_set(v___x_1021_, 0, v_val_1028_);
v___x_1030_ = v___x_1021_;
goto v_reusejp_1029_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_val_1028_);
v___x_1030_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1029_;
}
v_reusejp_1029_:
{
return v___x_1030_;
}
}
}
}
else
{
lean_object* v_a_1033_; lean_object* v___x_1035_; uint8_t v_isShared_1036_; uint8_t v_isSharedCheck_1040_; 
v_a_1033_ = lean_ctor_get(v___x_1018_, 0);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_1018_);
if (v_isSharedCheck_1040_ == 0)
{
v___x_1035_ = v___x_1018_;
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
else
{
lean_inc(v_a_1033_);
lean_dec(v___x_1018_);
v___x_1035_ = lean_box(0);
v_isShared_1036_ = v_isSharedCheck_1040_;
goto v_resetjp_1034_;
}
v_resetjp_1034_:
{
lean_object* v___x_1038_; 
if (v_isShared_1036_ == 0)
{
v___x_1038_ = v___x_1035_;
goto v_reusejp_1037_;
}
else
{
lean_object* v_reuseFailAlloc_1039_; 
v_reuseFailAlloc_1039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1039_, 0, v_a_1033_);
v___x_1038_ = v_reuseFailAlloc_1039_;
goto v_reusejp_1037_;
}
v_reusejp_1037_:
{
return v___x_1038_;
}
}
}
}
}
}
else
{
lean_object* v_a_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1049_; 
v_a_1042_ = lean_ctor_get(v___x_1004_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_1004_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1044_ = v___x_1004_;
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_a_1042_);
lean_dec(v___x_1004_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v___x_1047_; 
if (v_isShared_1045_ == 0)
{
v___x_1047_ = v___x_1044_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v_a_1042_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0___boxed(lean_object* v_t_1050_, lean_object* v_init_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_){
_start:
{
lean_object* v_res_1057_; 
v_res_1057_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0(v_t_1050_, v_init_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_);
lean_dec(v___y_1055_);
lean_dec_ref(v___y_1054_);
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec_ref(v_t_1050_);
return v_res_1057_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0(lean_object* v_ctx_1058_, lean_object* v_mvarId_1059_, lean_object* v_simprocs_1060_, lean_object* v_discharge_x3f_1061_, uint8_t v_simplifyTarget_1062_, lean_object* v_stats_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_){
_start:
{
lean_object* v_lctx_1069_; lean_object* v_decls_1070_; lean_object* v_size_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; 
v_lctx_1069_ = lean_ctor_get(v___y_1064_, 2);
v_decls_1070_ = lean_ctor_get(v_lctx_1069_, 1);
v_size_1071_ = lean_ctor_get(v_decls_1070_, 2);
v___x_1072_ = lean_mk_empty_array_with_capacity(v_size_1071_);
v___x_1073_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0(v_decls_1070_, v___x_1072_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
if (lean_obj_tag(v___x_1073_) == 0)
{
lean_object* v_a_1074_; lean_object* v___x_1075_; 
v_a_1074_ = lean_ctor_get(v___x_1073_, 0);
lean_inc(v_a_1074_);
lean_dec_ref_known(v___x_1073_, 1);
v___x_1075_ = lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta(v_ctx_1058_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
if (lean_obj_tag(v___x_1075_) == 0)
{
lean_object* v_a_1076_; lean_object* v___x_1077_; 
v_a_1076_ = lean_ctor_get(v___x_1075_, 0);
lean_inc(v_a_1076_);
lean_dec_ref_known(v___x_1075_, 1);
v___x_1077_ = lp_aesop_Aesop_simpGoal(v_mvarId_1059_, v_a_1076_, v_simprocs_1060_, v_discharge_x3f_1061_, v_simplifyTarget_1062_, v_a_1074_, v_stats_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
return v___x_1077_;
}
else
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec(v_a_1074_);
lean_dec_ref(v_stats_1063_);
lean_dec(v_discharge_x3f_1061_);
lean_dec_ref(v_simprocs_1060_);
lean_dec(v_mvarId_1059_);
v_a_1078_ = lean_ctor_get(v___x_1075_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1075_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1075_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
}
}
}
}
else
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1093_; 
lean_dec_ref(v_stats_1063_);
lean_dec(v_discharge_x3f_1061_);
lean_dec_ref(v_simprocs_1060_);
lean_dec(v_mvarId_1059_);
lean_dec_ref(v_ctx_1058_);
v_a_1086_ = lean_ctor_get(v___x_1073_, 0);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1073_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1088_ = v___x_1073_;
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_1073_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v___x_1091_; 
if (v_isShared_1089_ == 0)
{
v___x_1091_ = v___x_1088_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_a_1086_);
v___x_1091_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
return v___x_1091_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0___boxed(lean_object* v_ctx_1094_, lean_object* v_mvarId_1095_, lean_object* v_simprocs_1096_, lean_object* v_discharge_x3f_1097_, lean_object* v_simplifyTarget_1098_, lean_object* v_stats_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_){
_start:
{
uint8_t v_simplifyTarget_boxed_1105_; lean_object* v_res_1106_; 
v_simplifyTarget_boxed_1105_ = lean_unbox(v_simplifyTarget_1098_);
v_res_1106_ = lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0(v_ctx_1094_, v_mvarId_1095_, v_simprocs_1096_, v_discharge_x3f_1097_, v_simplifyTarget_boxed_1105_, v_stats_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_);
lean_dec(v___y_1103_);
lean_dec_ref(v___y_1102_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
return v_res_1106_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses(lean_object* v_mvarId_1107_, lean_object* v_ctx_1108_, lean_object* v_simprocs_1109_, lean_object* v_discharge_x3f_1110_, uint8_t v_simplifyTarget_1111_, lean_object* v_stats_1112_, lean_object* v_a_1113_, lean_object* v_a_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_){
_start:
{
lean_object* v___x_1118_; lean_object* v___f_1119_; lean_object* v___x_1120_; 
v___x_1118_ = lean_box(v_simplifyTarget_1111_);
lean_inc(v_mvarId_1107_);
v___f_1119_ = lean_alloc_closure((void*)(lp_aesop_Aesop_simpGoalWithAllHypotheses___lam__0___boxed), 11, 6);
lean_closure_set(v___f_1119_, 0, v_ctx_1108_);
lean_closure_set(v___f_1119_, 1, v_mvarId_1107_);
lean_closure_set(v___f_1119_, 2, v_simprocs_1109_);
lean_closure_set(v___f_1119_, 3, v_discharge_x3f_1110_);
lean_closure_set(v___f_1119_, 4, v___x_1118_);
lean_closure_set(v___f_1119_, 5, v_stats_1112_);
v___x_1120_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(v_mvarId_1107_, v___f_1119_, v_a_1113_, v_a_1114_, v_a_1115_, v_a_1116_);
return v___x_1120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpGoalWithAllHypotheses___boxed(lean_object* v_mvarId_1121_, lean_object* v_ctx_1122_, lean_object* v_simprocs_1123_, lean_object* v_discharge_x3f_1124_, lean_object* v_simplifyTarget_1125_, lean_object* v_stats_1126_, lean_object* v_a_1127_, lean_object* v_a_1128_, lean_object* v_a_1129_, lean_object* v_a_1130_, lean_object* v_a_1131_){
_start:
{
uint8_t v_simplifyTarget_boxed_1132_; lean_object* v_res_1133_; 
v_simplifyTarget_boxed_1132_ = lean_unbox(v_simplifyTarget_1125_);
v_res_1133_ = lp_aesop_Aesop_simpGoalWithAllHypotheses(v_mvarId_1121_, v_ctx_1122_, v_simprocs_1123_, v_discharge_x3f_1124_, v_simplifyTarget_boxed_1132_, v_stats_1126_, v_a_1127_, v_a_1128_, v_a_1129_, v_a_1130_);
lean_dec(v_a_1130_);
lean_dec_ref(v_a_1129_);
lean_dec(v_a_1128_);
lean_dec_ref(v_a_1127_);
return v_res_1133_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5(lean_object* v_as_1134_, size_t v_sz_1135_, size_t v_i_1136_, lean_object* v_b_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
lean_object* v___x_1143_; 
v___x_1143_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___redArg(v_as_1134_, v_sz_1135_, v_i_1136_, v_b_1137_);
return v___x_1143_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5___boxed(lean_object* v_as_1144_, lean_object* v_sz_1145_, lean_object* v_i_1146_, lean_object* v_b_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_){
_start:
{
size_t v_sz_boxed_1153_; size_t v_i_boxed_1154_; lean_object* v_res_1155_; 
v_sz_boxed_1153_ = lean_unbox_usize(v_sz_1145_);
lean_dec(v_sz_1145_);
v_i_boxed_1154_ = lean_unbox_usize(v_i_1146_);
lean_dec(v_i_1146_);
v_res_1155_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__1_spec__5(v_as_1144_, v_sz_boxed_1153_, v_i_boxed_1154_, v_b_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1149_);
lean_dec_ref(v___y_1148_);
lean_dec_ref(v_as_1144_);
return v_res_1155_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4(lean_object* v_as_1156_, size_t v_sz_1157_, size_t v_i_1158_, lean_object* v_b_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_){
_start:
{
lean_object* v___x_1165_; 
v___x_1165_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___redArg(v_as_1156_, v_sz_1157_, v_i_1158_, v_b_1159_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v_as_1166_, lean_object* v_sz_1167_, lean_object* v_i_1168_, lean_object* v_b_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_){
_start:
{
size_t v_sz_boxed_1175_; size_t v_i_boxed_1176_; lean_object* v_res_1177_; 
v_sz_boxed_1175_ = lean_unbox_usize(v_sz_1167_);
lean_dec(v_sz_1167_);
v_i_boxed_1176_ = lean_unbox_usize(v_i_1168_);
lean_dec(v_i_1168_);
v_res_1177_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_simpGoalWithAllHypotheses_spec__0_spec__0_spec__3_spec__4(v_as_1166_, v_sz_boxed_1175_, v_i_boxed_1176_, v_b_1169_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_);
lean_dec(v___y_1173_);
lean_dec_ref(v___y_1172_);
lean_dec(v___y_1171_);
lean_dec_ref(v___y_1170_);
lean_dec_ref(v_as_1166_);
return v_res_1177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___lam__0(lean_object* v_ctx_1178_, lean_object* v_mvarId_1179_, lean_object* v_simprocs_1180_, lean_object* v_stats_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lp_aesop_Aesop_addLetDeclsToSimpTheoremsUnlessZetaDelta(v_ctx_1178_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v_a_1188_; lean_object* v___x_1189_; 
v_a_1188_ = lean_ctor_get(v___x_1187_, 0);
lean_inc(v_a_1188_);
lean_dec_ref_known(v___x_1187_, 1);
lean_inc(v_mvarId_1179_);
v___x_1189_ = l_Lean_Meta_simpAll(v_mvarId_1179_, v_a_1188_, v_simprocs_1180_, v_stats_1181_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
if (lean_obj_tag(v___x_1189_) == 0)
{
lean_object* v_a_1190_; lean_object* v___x_1192_; uint8_t v_isShared_1193_; uint8_t v_isSharedCheck_1220_; 
v_a_1190_ = lean_ctor_get(v___x_1189_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v___x_1189_);
if (v_isSharedCheck_1220_ == 0)
{
v___x_1192_ = v___x_1189_;
v_isShared_1193_ = v_isSharedCheck_1220_;
goto v_resetjp_1191_;
}
else
{
lean_inc(v_a_1190_);
lean_dec(v___x_1189_);
v___x_1192_ = lean_box(0);
v_isShared_1193_ = v_isSharedCheck_1220_;
goto v_resetjp_1191_;
}
v_resetjp_1191_:
{
lean_object* v_fst_1194_; 
v_fst_1194_ = lean_ctor_get(v_a_1190_, 0);
if (lean_obj_tag(v_fst_1194_) == 0)
{
lean_object* v_snd_1195_; lean_object* v_usedTheorems_1196_; lean_object* v___x_1197_; lean_object* v___x_1199_; 
lean_dec(v_mvarId_1179_);
v_snd_1195_ = lean_ctor_get(v_a_1190_, 1);
lean_inc(v_snd_1195_);
lean_dec(v_a_1190_);
v_usedTheorems_1196_ = lean_ctor_get(v_snd_1195_, 0);
lean_inc_ref(v_usedTheorems_1196_);
lean_dec(v_snd_1195_);
v___x_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1197_, 0, v_usedTheorems_1196_);
if (v_isShared_1193_ == 0)
{
lean_ctor_set(v___x_1192_, 0, v___x_1197_);
v___x_1199_ = v___x_1192_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v___x_1197_);
v___x_1199_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
return v___x_1199_;
}
}
else
{
lean_object* v_snd_1201_; lean_object* v_val_1202_; uint8_t v___x_1203_; 
lean_inc_ref(v_fst_1194_);
v_snd_1201_ = lean_ctor_get(v_a_1190_, 1);
lean_inc(v_snd_1201_);
lean_dec(v_a_1190_);
v_val_1202_ = lean_ctor_get(v_fst_1194_, 0);
lean_inc(v_val_1202_);
lean_dec_ref_known(v_fst_1194_, 1);
v___x_1203_ = l_Lean_instBEqMVarId_beq(v_val_1202_, v_mvarId_1179_);
lean_dec(v_mvarId_1179_);
if (v___x_1203_ == 0)
{
lean_object* v_usedTheorems_1204_; lean_object* v___x_1206_; uint8_t v_isShared_1207_; uint8_t v_isSharedCheck_1214_; 
v_usedTheorems_1204_ = lean_ctor_get(v_snd_1201_, 0);
v_isSharedCheck_1214_ = !lean_is_exclusive(v_snd_1201_);
if (v_isSharedCheck_1214_ == 0)
{
lean_object* v_unused_1215_; 
v_unused_1215_ = lean_ctor_get(v_snd_1201_, 1);
lean_dec(v_unused_1215_);
v___x_1206_ = v_snd_1201_;
v_isShared_1207_ = v_isSharedCheck_1214_;
goto v_resetjp_1205_;
}
else
{
lean_inc(v_usedTheorems_1204_);
lean_dec(v_snd_1201_);
v___x_1206_ = lean_box(0);
v_isShared_1207_ = v_isSharedCheck_1214_;
goto v_resetjp_1205_;
}
v_resetjp_1205_:
{
lean_object* v___x_1209_; 
if (v_isShared_1207_ == 0)
{
lean_ctor_set_tag(v___x_1206_, 2);
lean_ctor_set(v___x_1206_, 1, v_usedTheorems_1204_);
lean_ctor_set(v___x_1206_, 0, v_val_1202_);
v___x_1209_ = v___x_1206_;
goto v_reusejp_1208_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v_val_1202_);
lean_ctor_set(v_reuseFailAlloc_1213_, 1, v_usedTheorems_1204_);
v___x_1209_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1208_;
}
v_reusejp_1208_:
{
lean_object* v___x_1211_; 
if (v_isShared_1193_ == 0)
{
lean_ctor_set(v___x_1192_, 0, v___x_1209_);
v___x_1211_ = v___x_1192_;
goto v_reusejp_1210_;
}
else
{
lean_object* v_reuseFailAlloc_1212_; 
v_reuseFailAlloc_1212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1212_, 0, v___x_1209_);
v___x_1211_ = v_reuseFailAlloc_1212_;
goto v_reusejp_1210_;
}
v_reusejp_1210_:
{
return v___x_1211_;
}
}
}
}
else
{
lean_object* v___x_1216_; lean_object* v___x_1218_; 
lean_dec(v_val_1202_);
lean_dec(v_snd_1201_);
v___x_1216_ = lean_box(1);
if (v_isShared_1193_ == 0)
{
lean_ctor_set(v___x_1192_, 0, v___x_1216_);
v___x_1218_ = v___x_1192_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v___x_1216_);
v___x_1218_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
return v___x_1218_;
}
}
}
}
}
else
{
lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1228_; 
lean_dec(v_mvarId_1179_);
v_a_1221_ = lean_ctor_get(v___x_1189_, 0);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1189_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1223_ = v___x_1189_;
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1189_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v___x_1226_; 
if (v_isShared_1224_ == 0)
{
v___x_1226_ = v___x_1223_;
goto v_reusejp_1225_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_a_1221_);
v___x_1226_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1225_;
}
v_reusejp_1225_:
{
return v___x_1226_;
}
}
}
}
else
{
lean_object* v_a_1229_; lean_object* v___x_1231_; uint8_t v_isShared_1232_; uint8_t v_isSharedCheck_1236_; 
lean_dec_ref(v_simprocs_1180_);
lean_dec(v_mvarId_1179_);
v_a_1229_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1236_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1236_ == 0)
{
v___x_1231_ = v___x_1187_;
v_isShared_1232_ = v_isSharedCheck_1236_;
goto v_resetjp_1230_;
}
else
{
lean_inc(v_a_1229_);
lean_dec(v___x_1187_);
v___x_1231_ = lean_box(0);
v_isShared_1232_ = v_isSharedCheck_1236_;
goto v_resetjp_1230_;
}
v_resetjp_1230_:
{
lean_object* v___x_1234_; 
if (v_isShared_1232_ == 0)
{
v___x_1234_ = v___x_1231_;
goto v_reusejp_1233_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v_a_1229_);
v___x_1234_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1233_;
}
v_reusejp_1233_:
{
return v___x_1234_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___lam__0___boxed(lean_object* v_ctx_1237_, lean_object* v_mvarId_1238_, lean_object* v_simprocs_1239_, lean_object* v_stats_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_){
_start:
{
lean_object* v_res_1246_; 
v_res_1246_ = lp_aesop_Aesop_simpAll___lam__0(v_ctx_1237_, v_mvarId_1238_, v_simprocs_1239_, v_stats_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec(v___y_1242_);
lean_dec_ref(v___y_1241_);
lean_dec_ref(v_stats_1240_);
return v_res_1246_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll(lean_object* v_mvarId_1247_, lean_object* v_ctx_1248_, lean_object* v_simprocs_1249_, lean_object* v_stats_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_){
_start:
{
uint8_t v___x_1256_; lean_object* v_ctx_1257_; lean_object* v___f_1258_; lean_object* v___x_1259_; 
v___x_1256_ = 0;
v_ctx_1257_ = l_Lean_Meta_Simp_Context_setFailIfUnchanged(v_ctx_1248_, v___x_1256_);
lean_inc(v_mvarId_1247_);
v___f_1258_ = lean_alloc_closure((void*)(lp_aesop_Aesop_simpAll___lam__0___boxed), 9, 4);
lean_closure_set(v___f_1258_, 0, v_ctx_1257_);
lean_closure_set(v___f_1258_, 1, v_mvarId_1247_);
lean_closure_set(v___f_1258_, 2, v_simprocs_1249_);
lean_closure_set(v___f_1258_, 3, v_stats_1250_);
v___x_1259_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_simpGoalWithAllHypotheses_spec__1___redArg(v_mvarId_1247_, v___f_1258_, v_a_1251_, v_a_1252_, v_a_1253_, v_a_1254_);
return v___x_1259_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_simpAll___boxed(lean_object* v_mvarId_1260_, lean_object* v_ctx_1261_, lean_object* v_simprocs_1262_, lean_object* v_stats_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_, lean_object* v_a_1268_){
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_aesop_Aesop_simpAll(v_mvarId_1260_, v_ctx_1261_, v_simprocs_1262_, v_stats_1263_, v_a_1264_, v_a_1265_, v_a_1266_, v_a_1267_);
lean_dec(v_a_1267_);
lean_dec_ref(v_a_1266_);
lean_dec(v_a_1265_);
lean_dec_ref(v_a_1264_);
return v_res_1269_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_SimpAll(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Expansion_Simp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_SimpAll(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Expansion_Simp(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Simp_SimpAll(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Expansion_Simp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_SimpAll(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Expansion_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Expansion_Simp(builtin);
}
#ifdef __cplusplus
}
#endif
