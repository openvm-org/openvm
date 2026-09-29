// Lean compiler output
// Module: Mathlib.Tactic.RenameBVar
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Location public meta import Mathlib.Lean.Expr.Basic public import Mathlib.Util.Tactic
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkAuxDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_instMonadTacticM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_instMonadTacticM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedLocalContext_default;
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_LocalContext_mkLetDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_sharecommon_quick(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* lp_mathlib_Lean_Expr_renameBVar(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_LocalDecl_setType(lean_object*, lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_set___redArg(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_expandLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 21, .m_data = "tacticRename_bvar_→__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(92, 179, 110, 231, 179, 40, 109, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rename_bvar "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " → "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192____;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unexpected location syntax"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__4_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__5 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__5_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Term_instMonadTermElabM___lam__1___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__6 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__6_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Tactic_instMonadTacticM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__7 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__7_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Tactic_instMonadTacticM___lam__1___boxed, .m_arity = 13, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__8 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Lean.MetavarContext"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.instantiateLCtxMVars"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Invalid auxiliary declaration found in local context: "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = " does not have an associated full name."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0(lean_object* v_old_1_, lean_object* v_new_2_, lean_object* v_ldecl_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = l_Lean_LocalDecl_type(v_ldecl_3_);
v___x_5_ = lp_mathlib_Lean_Expr_renameBVar(v___x_4_, v_old_1_, v_new_2_);
v___x_6_ = l_Lean_LocalDecl_setType(v_ldecl_3_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0___boxed(lean_object* v_old_7_, lean_object* v_new_8_, lean_object* v_ldecl_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0(v_old_7_, v_new_8_, v_ldecl_9_);
lean_dec(v_old_7_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1___lam__0(lean_object* v_f_11_, lean_object* v_mdecl_12_){
_start:
{
lean_object* v_userName_13_; lean_object* v_lctx_14_; lean_object* v_type_15_; lean_object* v_depth_16_; lean_object* v_localInstances_17_; uint8_t v_kind_18_; lean_object* v_numScopeArgs_19_; lean_object* v_index_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_28_; 
v_userName_13_ = lean_ctor_get(v_mdecl_12_, 0);
v_lctx_14_ = lean_ctor_get(v_mdecl_12_, 1);
v_type_15_ = lean_ctor_get(v_mdecl_12_, 2);
v_depth_16_ = lean_ctor_get(v_mdecl_12_, 3);
v_localInstances_17_ = lean_ctor_get(v_mdecl_12_, 4);
v_kind_18_ = lean_ctor_get_uint8(v_mdecl_12_, sizeof(void*)*7);
v_numScopeArgs_19_ = lean_ctor_get(v_mdecl_12_, 5);
v_index_20_ = lean_ctor_get(v_mdecl_12_, 6);
v_isSharedCheck_28_ = !lean_is_exclusive(v_mdecl_12_);
if (v_isSharedCheck_28_ == 0)
{
v___x_22_ = v_mdecl_12_;
v_isShared_23_ = v_isSharedCheck_28_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_index_20_);
lean_inc(v_numScopeArgs_19_);
lean_inc(v_localInstances_17_);
lean_inc(v_depth_16_);
lean_inc(v_type_15_);
lean_inc(v_lctx_14_);
lean_inc(v_userName_13_);
lean_dec(v_mdecl_12_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_28_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_24_; lean_object* v___x_26_; 
v___x_24_ = lean_apply_1(v_f_11_, v_lctx_14_);
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 1, v___x_24_);
v___x_26_ = v___x_22_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_userName_13_);
lean_ctor_set(v_reuseFailAlloc_27_, 1, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_27_, 2, v_type_15_);
lean_ctor_set(v_reuseFailAlloc_27_, 3, v_depth_16_);
lean_ctor_set(v_reuseFailAlloc_27_, 4, v_localInstances_17_);
lean_ctor_set(v_reuseFailAlloc_27_, 5, v_numScopeArgs_19_);
lean_ctor_set(v_reuseFailAlloc_27_, 6, v_index_20_);
lean_ctor_set_uint8(v_reuseFailAlloc_27_, sizeof(void*)*7, v_kind_18_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13___redArg(lean_object* v_x_29_, lean_object* v_x_30_, lean_object* v_x_31_, lean_object* v_x_32_){
_start:
{
lean_object* v_ks_33_; lean_object* v_vs_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_58_; 
v_ks_33_ = lean_ctor_get(v_x_29_, 0);
v_vs_34_ = lean_ctor_get(v_x_29_, 1);
v_isSharedCheck_58_ = !lean_is_exclusive(v_x_29_);
if (v_isSharedCheck_58_ == 0)
{
v___x_36_ = v_x_29_;
v_isShared_37_ = v_isSharedCheck_58_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_vs_34_);
lean_inc(v_ks_33_);
lean_dec(v_x_29_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_58_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = lean_array_get_size(v_ks_33_);
v___x_39_ = lean_nat_dec_lt(v_x_30_, v___x_38_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_43_; 
lean_dec(v_x_30_);
v___x_40_ = lean_array_push(v_ks_33_, v_x_31_);
v___x_41_ = lean_array_push(v_vs_34_, v_x_32_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 1, v___x_41_);
lean_ctor_set(v___x_36_, 0, v___x_40_);
v___x_43_ = v___x_36_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v___x_40_);
lean_ctor_set(v_reuseFailAlloc_44_, 1, v___x_41_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
else
{
lean_object* v_k_x27_45_; uint8_t v___x_46_; 
v_k_x27_45_ = lean_array_fget_borrowed(v_ks_33_, v_x_30_);
v___x_46_ = l_Lean_instBEqMVarId_beq(v_x_31_, v_k_x27_45_);
if (v___x_46_ == 0)
{
lean_object* v___x_48_; 
if (v_isShared_37_ == 0)
{
v___x_48_ = v___x_36_;
goto v_reusejp_47_;
}
else
{
lean_object* v_reuseFailAlloc_52_; 
v_reuseFailAlloc_52_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_52_, 0, v_ks_33_);
lean_ctor_set(v_reuseFailAlloc_52_, 1, v_vs_34_);
v___x_48_ = v_reuseFailAlloc_52_;
goto v_reusejp_47_;
}
v_reusejp_47_:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = lean_nat_add(v_x_30_, v___x_49_);
lean_dec(v_x_30_);
v_x_29_ = v___x_48_;
v_x_30_ = v___x_50_;
goto _start;
}
}
else
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_56_; 
v___x_53_ = lean_array_fset(v_ks_33_, v_x_30_, v_x_31_);
v___x_54_ = lean_array_fset(v_vs_34_, v_x_30_, v_x_32_);
lean_dec(v_x_30_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 1, v___x_54_);
lean_ctor_set(v___x_36_, 0, v___x_53_);
v___x_56_ = v___x_36_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v___x_53_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v___x_54_);
v___x_56_ = v_reuseFailAlloc_57_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
return v___x_56_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12___redArg(lean_object* v_n_59_, lean_object* v_k_60_, lean_object* v_v_61_){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_unsigned_to_nat(0u);
v___x_63_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13___redArg(v_n_59_, v___x_62_, v_k_60_, v_v_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0(void){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(lean_object* v_x_65_, size_t v_x_66_, size_t v_x_67_, lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
if (lean_obj_tag(v_x_65_) == 0)
{
lean_object* v_es_70_; size_t v___x_71_; size_t v___x_72_; lean_object* v_j_73_; lean_object* v___x_74_; uint8_t v___x_75_; 
v_es_70_ = lean_ctor_get(v_x_65_, 0);
v___x_71_ = ((size_t)31ULL);
v___x_72_ = lean_usize_land(v_x_66_, v___x_71_);
v_j_73_ = lean_usize_to_nat(v___x_72_);
v___x_74_ = lean_array_get_size(v_es_70_);
v___x_75_ = lean_nat_dec_lt(v_j_73_, v___x_74_);
if (v___x_75_ == 0)
{
lean_dec(v_j_73_);
lean_dec(v_x_69_);
lean_dec(v_x_68_);
return v_x_65_;
}
else
{
lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_114_; 
lean_inc_ref(v_es_70_);
v_isSharedCheck_114_ = !lean_is_exclusive(v_x_65_);
if (v_isSharedCheck_114_ == 0)
{
lean_object* v_unused_115_; 
v_unused_115_ = lean_ctor_get(v_x_65_, 0);
lean_dec(v_unused_115_);
v___x_77_ = v_x_65_;
v_isShared_78_ = v_isSharedCheck_114_;
goto v_resetjp_76_;
}
else
{
lean_dec(v_x_65_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_114_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v_v_79_; lean_object* v___x_80_; lean_object* v_xs_x27_81_; lean_object* v___y_83_; 
v_v_79_ = lean_array_fget(v_es_70_, v_j_73_);
v___x_80_ = lean_box(0);
v_xs_x27_81_ = lean_array_fset(v_es_70_, v_j_73_, v___x_80_);
switch(lean_obj_tag(v_v_79_))
{
case 0:
{
lean_object* v_key_88_; lean_object* v_val_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_99_; 
v_key_88_ = lean_ctor_get(v_v_79_, 0);
v_val_89_ = lean_ctor_get(v_v_79_, 1);
v_isSharedCheck_99_ = !lean_is_exclusive(v_v_79_);
if (v_isSharedCheck_99_ == 0)
{
v___x_91_ = v_v_79_;
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_val_89_);
lean_inc(v_key_88_);
lean_dec(v_v_79_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
uint8_t v___x_93_; 
v___x_93_ = l_Lean_instBEqMVarId_beq(v_x_68_, v_key_88_);
if (v___x_93_ == 0)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
lean_del_object(v___x_91_);
v___x_94_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_88_, v_val_89_, v_x_68_, v_x_69_);
v___x_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
v___y_83_ = v___x_95_;
goto v___jp_82_;
}
else
{
lean_object* v___x_97_; 
lean_dec(v_val_89_);
lean_dec(v_key_88_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 1, v_x_69_);
lean_ctor_set(v___x_91_, 0, v_x_68_);
v___x_97_ = v___x_91_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_x_68_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_x_69_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
v___y_83_ = v___x_97_;
goto v___jp_82_;
}
}
}
}
case 1:
{
lean_object* v_node_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_112_; 
v_node_100_ = lean_ctor_get(v_v_79_, 0);
v_isSharedCheck_112_ = !lean_is_exclusive(v_v_79_);
if (v_isSharedCheck_112_ == 0)
{
v___x_102_ = v_v_79_;
v_isShared_103_ = v_isSharedCheck_112_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_node_100_);
lean_dec(v_v_79_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_112_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
size_t v___x_104_; size_t v___x_105_; size_t v___x_106_; size_t v___x_107_; lean_object* v___x_108_; lean_object* v___x_110_; 
v___x_104_ = ((size_t)5ULL);
v___x_105_ = lean_usize_shift_right(v_x_66_, v___x_104_);
v___x_106_ = ((size_t)1ULL);
v___x_107_ = lean_usize_add(v_x_67_, v___x_106_);
v___x_108_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(v_node_100_, v___x_105_, v___x_107_, v_x_68_, v_x_69_);
if (v_isShared_103_ == 0)
{
lean_ctor_set(v___x_102_, 0, v___x_108_);
v___x_110_ = v___x_102_;
goto v_reusejp_109_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v___x_108_);
v___x_110_ = v_reuseFailAlloc_111_;
goto v_reusejp_109_;
}
v_reusejp_109_:
{
v___y_83_ = v___x_110_;
goto v___jp_82_;
}
}
}
default: 
{
lean_object* v___x_113_; 
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v_x_68_);
lean_ctor_set(v___x_113_, 1, v_x_69_);
v___y_83_ = v___x_113_;
goto v___jp_82_;
}
}
v___jp_82_:
{
lean_object* v___x_84_; lean_object* v___x_86_; 
v___x_84_ = lean_array_fset(v_xs_x27_81_, v_j_73_, v___y_83_);
lean_dec(v_j_73_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 0, v___x_84_);
v___x_86_ = v___x_77_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
}
}
else
{
lean_object* v_ks_116_; lean_object* v_vs_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_137_; 
v_ks_116_ = lean_ctor_get(v_x_65_, 0);
v_vs_117_ = lean_ctor_get(v_x_65_, 1);
v_isSharedCheck_137_ = !lean_is_exclusive(v_x_65_);
if (v_isSharedCheck_137_ == 0)
{
v___x_119_ = v_x_65_;
v_isShared_120_ = v_isSharedCheck_137_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_vs_117_);
lean_inc(v_ks_116_);
lean_dec(v_x_65_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_137_;
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
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_ks_116_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v_vs_117_);
v___x_122_ = v_reuseFailAlloc_136_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
lean_object* v_newNode_123_; uint8_t v___y_125_; size_t v___x_131_; uint8_t v___x_132_; 
v_newNode_123_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12___redArg(v___x_122_, v_x_68_, v_x_69_);
v___x_131_ = ((size_t)7ULL);
v___x_132_ = lean_usize_dec_le(v___x_131_, v_x_67_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; lean_object* v___x_134_; uint8_t v___x_135_; 
v___x_133_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_123_);
v___x_134_ = lean_unsigned_to_nat(4u);
v___x_135_ = lean_nat_dec_lt(v___x_133_, v___x_134_);
lean_dec(v___x_133_);
v___y_125_ = v___x_135_;
goto v___jp_124_;
}
else
{
v___y_125_ = v___x_132_;
goto v___jp_124_;
}
v___jp_124_:
{
if (v___y_125_ == 0)
{
lean_object* v_ks_126_; lean_object* v_vs_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_ks_126_ = lean_ctor_get(v_newNode_123_, 0);
lean_inc_ref(v_ks_126_);
v_vs_127_ = lean_ctor_get(v_newNode_123_, 1);
lean_inc_ref(v_vs_127_);
lean_dec_ref(v_newNode_123_);
v___x_128_ = lean_unsigned_to_nat(0u);
v___x_129_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0);
v___x_130_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg(v_x_67_, v_ks_126_, v_vs_127_, v___x_128_, v___x_129_);
lean_dec_ref(v_vs_127_);
lean_dec_ref(v_ks_126_);
return v___x_130_;
}
else
{
return v_newNode_123_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg(size_t v_depth_138_, lean_object* v_keys_139_, lean_object* v_vals_140_, lean_object* v_i_141_, lean_object* v_entries_142_){
_start:
{
lean_object* v___x_143_; uint8_t v___x_144_; 
v___x_143_ = lean_array_get_size(v_keys_139_);
v___x_144_ = lean_nat_dec_lt(v_i_141_, v___x_143_);
if (v___x_144_ == 0)
{
lean_dec(v_i_141_);
return v_entries_142_;
}
else
{
lean_object* v_k_145_; lean_object* v_v_146_; uint64_t v___x_147_; size_t v_h_148_; size_t v___x_149_; lean_object* v___x_150_; size_t v___x_151_; size_t v___x_152_; size_t v___x_153_; size_t v_h_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v_k_145_ = lean_array_fget_borrowed(v_keys_139_, v_i_141_);
v_v_146_ = lean_array_fget_borrowed(v_vals_140_, v_i_141_);
v___x_147_ = l_Lean_instHashableMVarId_hash(v_k_145_);
v_h_148_ = lean_uint64_to_usize(v___x_147_);
v___x_149_ = ((size_t)5ULL);
v___x_150_ = lean_unsigned_to_nat(1u);
v___x_151_ = ((size_t)1ULL);
v___x_152_ = lean_usize_sub(v_depth_138_, v___x_151_);
v___x_153_ = lean_usize_mul(v___x_149_, v___x_152_);
v_h_154_ = lean_usize_shift_right(v_h_148_, v___x_153_);
v___x_155_ = lean_nat_add(v_i_141_, v___x_150_);
lean_dec(v_i_141_);
lean_inc(v_v_146_);
lean_inc(v_k_145_);
v___x_156_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(v_entries_142_, v_h_154_, v_depth_138_, v_k_145_, v_v_146_);
v_i_141_ = v___x_155_;
v_entries_142_ = v___x_156_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg___boxed(lean_object* v_depth_158_, lean_object* v_keys_159_, lean_object* v_vals_160_, lean_object* v_i_161_, lean_object* v_entries_162_){
_start:
{
size_t v_depth_boxed_163_; lean_object* v_res_164_; 
v_depth_boxed_163_ = lean_unbox_usize(v_depth_158_);
lean_dec(v_depth_158_);
v_res_164_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg(v_depth_boxed_163_, v_keys_159_, v_vals_160_, v_i_161_, v_entries_162_);
lean_dec_ref(v_vals_160_);
lean_dec_ref(v_keys_159_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___boxed(lean_object* v_x_165_, lean_object* v_x_166_, lean_object* v_x_167_, lean_object* v_x_168_, lean_object* v_x_169_){
_start:
{
size_t v_x_1063__boxed_170_; size_t v_x_1064__boxed_171_; lean_object* v_res_172_; 
v_x_1063__boxed_170_ = lean_unbox_usize(v_x_166_);
lean_dec(v_x_166_);
v_x_1064__boxed_171_ = lean_unbox_usize(v_x_167_);
lean_dec(v_x_167_);
v_res_172_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(v_x_165_, v_x_1063__boxed_170_, v_x_1064__boxed_171_, v_x_168_, v_x_169_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7___redArg(lean_object* v_x_173_, lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
uint64_t v___x_176_; size_t v___x_177_; size_t v___x_178_; lean_object* v___x_179_; 
v___x_176_ = l_Lean_instHashableMVarId_hash(v_x_174_);
v___x_177_ = lean_uint64_to_usize(v___x_176_);
v___x_178_ = ((size_t)1ULL);
v___x_179_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(v_x_173_, v___x_177_, v___x_178_, v_x_174_, v_x_175_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg(lean_object* v_keys_180_, lean_object* v_vals_181_, lean_object* v_i_182_, lean_object* v_k_183_){
_start:
{
lean_object* v___x_184_; uint8_t v___x_185_; 
v___x_184_ = lean_array_get_size(v_keys_180_);
v___x_185_ = lean_nat_dec_lt(v_i_182_, v___x_184_);
if (v___x_185_ == 0)
{
lean_object* v___x_186_; 
lean_dec(v_i_182_);
v___x_186_ = lean_box(0);
return v___x_186_;
}
else
{
lean_object* v_k_x27_187_; uint8_t v___x_188_; 
v_k_x27_187_ = lean_array_fget_borrowed(v_keys_180_, v_i_182_);
v___x_188_ = l_Lean_instBEqMVarId_beq(v_k_183_, v_k_x27_187_);
if (v___x_188_ == 0)
{
lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_189_ = lean_unsigned_to_nat(1u);
v___x_190_ = lean_nat_add(v_i_182_, v___x_189_);
lean_dec(v_i_182_);
v_i_182_ = v___x_190_;
goto _start;
}
else
{
lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_192_ = lean_array_fget_borrowed(v_vals_181_, v_i_182_);
lean_dec(v_i_182_);
lean_inc(v___x_192_);
v___x_193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_193_, 0, v___x_192_);
return v___x_193_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg___boxed(lean_object* v_keys_194_, lean_object* v_vals_195_, lean_object* v_i_196_, lean_object* v_k_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg(v_keys_194_, v_vals_195_, v_i_196_, v_k_197_);
lean_dec(v_k_197_);
lean_dec_ref(v_vals_195_);
lean_dec_ref(v_keys_194_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg(lean_object* v_x_199_, size_t v_x_200_, lean_object* v_x_201_){
_start:
{
if (lean_obj_tag(v_x_199_) == 0)
{
lean_object* v_es_202_; lean_object* v___x_203_; size_t v___x_204_; size_t v___x_205_; lean_object* v_j_206_; lean_object* v___x_207_; 
v_es_202_ = lean_ctor_get(v_x_199_, 0);
v___x_203_ = lean_box(2);
v___x_204_ = ((size_t)31ULL);
v___x_205_ = lean_usize_land(v_x_200_, v___x_204_);
v_j_206_ = lean_usize_to_nat(v___x_205_);
v___x_207_ = lean_array_get_borrowed(v___x_203_, v_es_202_, v_j_206_);
lean_dec(v_j_206_);
switch(lean_obj_tag(v___x_207_))
{
case 0:
{
lean_object* v_key_208_; lean_object* v_val_209_; uint8_t v___x_210_; 
v_key_208_ = lean_ctor_get(v___x_207_, 0);
v_val_209_ = lean_ctor_get(v___x_207_, 1);
v___x_210_ = l_Lean_instBEqMVarId_beq(v_x_201_, v_key_208_);
if (v___x_210_ == 0)
{
lean_object* v___x_211_; 
v___x_211_ = lean_box(0);
return v___x_211_;
}
else
{
lean_object* v___x_212_; 
lean_inc(v_val_209_);
v___x_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_212_, 0, v_val_209_);
return v___x_212_;
}
}
case 1:
{
lean_object* v_node_213_; size_t v___x_214_; size_t v___x_215_; 
v_node_213_ = lean_ctor_get(v___x_207_, 0);
v___x_214_ = ((size_t)5ULL);
v___x_215_ = lean_usize_shift_right(v_x_200_, v___x_214_);
v_x_199_ = v_node_213_;
v_x_200_ = v___x_215_;
goto _start;
}
default: 
{
lean_object* v___x_217_; 
v___x_217_ = lean_box(0);
return v___x_217_;
}
}
}
else
{
lean_object* v_ks_218_; lean_object* v_vs_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v_ks_218_ = lean_ctor_get(v_x_199_, 0);
v_vs_219_ = lean_ctor_get(v_x_199_, 1);
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg(v_ks_218_, v_vs_219_, v___x_220_, v_x_201_);
return v___x_221_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg___boxed(lean_object* v_x_222_, lean_object* v_x_223_, lean_object* v_x_224_){
_start:
{
size_t v_x_1251__boxed_225_; lean_object* v_res_226_; 
v_x_1251__boxed_225_ = lean_unbox_usize(v_x_223_);
lean_dec(v_x_223_);
v_res_226_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg(v_x_222_, v_x_1251__boxed_225_, v_x_224_);
lean_dec(v_x_224_);
lean_dec_ref(v_x_222_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg(lean_object* v_x_227_, lean_object* v_x_228_){
_start:
{
uint64_t v___x_229_; size_t v___x_230_; lean_object* v___x_231_; 
v___x_229_ = l_Lean_instHashableMVarId_hash(v_x_228_);
v___x_230_ = lean_uint64_to_usize(v___x_229_);
v___x_231_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg(v_x_227_, v___x_230_, v_x_228_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg___boxed(lean_object* v_x_232_, lean_object* v_x_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg(v_x_232_, v_x_233_);
lean_dec(v_x_233_);
lean_dec_ref(v_x_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(lean_object* v_mvarId_235_, lean_object* v_f_236_, lean_object* v___y_237_){
_start:
{
lean_object* v___x_239_; lean_object* v_mctx_240_; lean_object* v_cache_241_; lean_object* v_zetaDeltaFVarIds_242_; lean_object* v_postponed_243_; lean_object* v_diag_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_287_; 
v___x_239_ = lean_st_ref_take(v___y_237_);
v_mctx_240_ = lean_ctor_get(v___x_239_, 0);
v_cache_241_ = lean_ctor_get(v___x_239_, 1);
v_zetaDeltaFVarIds_242_ = lean_ctor_get(v___x_239_, 2);
v_postponed_243_ = lean_ctor_get(v___x_239_, 3);
v_diag_244_ = lean_ctor_get(v___x_239_, 4);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_239_);
if (v_isSharedCheck_287_ == 0)
{
v___x_246_ = v___x_239_;
v_isShared_247_ = v_isSharedCheck_287_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_diag_244_);
lean_inc(v_postponed_243_);
lean_inc(v_zetaDeltaFVarIds_242_);
lean_inc(v_cache_241_);
lean_inc(v_mctx_240_);
lean_dec(v___x_239_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_287_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___y_249_; lean_object* v_depth_256_; lean_object* v_levelAssignDepth_257_; lean_object* v_lmvarCounter_258_; lean_object* v_mvarCounter_259_; lean_object* v_lDecls_260_; lean_object* v_decls_261_; lean_object* v_userNames_262_; lean_object* v_lAssignment_263_; lean_object* v_eAssignment_264_; lean_object* v_dAssignment_265_; lean_object* v___x_266_; 
v_depth_256_ = lean_ctor_get(v_mctx_240_, 0);
v_levelAssignDepth_257_ = lean_ctor_get(v_mctx_240_, 1);
v_lmvarCounter_258_ = lean_ctor_get(v_mctx_240_, 2);
v_mvarCounter_259_ = lean_ctor_get(v_mctx_240_, 3);
v_lDecls_260_ = lean_ctor_get(v_mctx_240_, 4);
v_decls_261_ = lean_ctor_get(v_mctx_240_, 5);
v_userNames_262_ = lean_ctor_get(v_mctx_240_, 6);
v_lAssignment_263_ = lean_ctor_get(v_mctx_240_, 7);
v_eAssignment_264_ = lean_ctor_get(v_mctx_240_, 8);
v_dAssignment_265_ = lean_ctor_get(v_mctx_240_, 9);
v___x_266_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg(v_decls_261_, v_mvarId_235_);
if (lean_obj_tag(v___x_266_) == 0)
{
lean_dec_ref(v_f_236_);
lean_dec(v_mvarId_235_);
v___y_249_ = v_mctx_240_;
goto v___jp_248_;
}
else
{
lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_276_; 
lean_inc_ref(v_dAssignment_265_);
lean_inc_ref(v_eAssignment_264_);
lean_inc_ref(v_lAssignment_263_);
lean_inc_ref(v_userNames_262_);
lean_inc_ref(v_decls_261_);
lean_inc_ref(v_lDecls_260_);
lean_inc(v_mvarCounter_259_);
lean_inc(v_lmvarCounter_258_);
lean_inc(v_levelAssignDepth_257_);
lean_inc(v_depth_256_);
v_isSharedCheck_276_ = !lean_is_exclusive(v_mctx_240_);
if (v_isSharedCheck_276_ == 0)
{
lean_object* v_unused_277_; lean_object* v_unused_278_; lean_object* v_unused_279_; lean_object* v_unused_280_; lean_object* v_unused_281_; lean_object* v_unused_282_; lean_object* v_unused_283_; lean_object* v_unused_284_; lean_object* v_unused_285_; lean_object* v_unused_286_; 
v_unused_277_ = lean_ctor_get(v_mctx_240_, 9);
lean_dec(v_unused_277_);
v_unused_278_ = lean_ctor_get(v_mctx_240_, 8);
lean_dec(v_unused_278_);
v_unused_279_ = lean_ctor_get(v_mctx_240_, 7);
lean_dec(v_unused_279_);
v_unused_280_ = lean_ctor_get(v_mctx_240_, 6);
lean_dec(v_unused_280_);
v_unused_281_ = lean_ctor_get(v_mctx_240_, 5);
lean_dec(v_unused_281_);
v_unused_282_ = lean_ctor_get(v_mctx_240_, 4);
lean_dec(v_unused_282_);
v_unused_283_ = lean_ctor_get(v_mctx_240_, 3);
lean_dec(v_unused_283_);
v_unused_284_ = lean_ctor_get(v_mctx_240_, 2);
lean_dec(v_unused_284_);
v_unused_285_ = lean_ctor_get(v_mctx_240_, 1);
lean_dec(v_unused_285_);
v_unused_286_ = lean_ctor_get(v_mctx_240_, 0);
lean_dec(v_unused_286_);
v___x_268_ = v_mctx_240_;
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
else
{
lean_dec(v_mctx_240_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v_val_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_274_; 
v_val_270_ = lean_ctor_get(v___x_266_, 0);
lean_inc(v_val_270_);
lean_dec_ref_known(v___x_266_, 1);
v___x_271_ = lean_apply_1(v_f_236_, v_val_270_);
v___x_272_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7___redArg(v_decls_261_, v_mvarId_235_, v___x_271_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 5, v___x_272_);
v___x_274_ = v___x_268_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_depth_256_);
lean_ctor_set(v_reuseFailAlloc_275_, 1, v_levelAssignDepth_257_);
lean_ctor_set(v_reuseFailAlloc_275_, 2, v_lmvarCounter_258_);
lean_ctor_set(v_reuseFailAlloc_275_, 3, v_mvarCounter_259_);
lean_ctor_set(v_reuseFailAlloc_275_, 4, v_lDecls_260_);
lean_ctor_set(v_reuseFailAlloc_275_, 5, v___x_272_);
lean_ctor_set(v_reuseFailAlloc_275_, 6, v_userNames_262_);
lean_ctor_set(v_reuseFailAlloc_275_, 7, v_lAssignment_263_);
lean_ctor_set(v_reuseFailAlloc_275_, 8, v_eAssignment_264_);
lean_ctor_set(v_reuseFailAlloc_275_, 9, v_dAssignment_265_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
v___y_249_ = v___x_274_;
goto v___jp_248_;
}
}
}
v___jp_248_:
{
lean_object* v___x_251_; 
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 0, v___y_249_);
v___x_251_ = v___x_246_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v___y_249_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v_cache_241_);
lean_ctor_set(v_reuseFailAlloc_255_, 2, v_zetaDeltaFVarIds_242_);
lean_ctor_set(v_reuseFailAlloc_255_, 3, v_postponed_243_);
lean_ctor_set(v_reuseFailAlloc_255_, 4, v_diag_244_);
v___x_251_ = v_reuseFailAlloc_255_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_252_ = lean_st_ref_set(v___y_237_, v___x_251_);
v___x_253_ = lean_box(0);
v___x_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
return v___x_254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_mvarId_288_, lean_object* v_f_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(v_mvarId_288_, v_f_289_, v___y_290_);
lean_dec(v___y_290_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1(lean_object* v_mvarId_293_, lean_object* v_f_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v___f_300_; lean_object* v___x_301_; 
v___f_300_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1___lam__0), 2, 1);
lean_closure_set(v___f_300_, 0, v_f_294_);
v___x_301_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(v_mvarId_293_, v___f_300_, v___y_296_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1___boxed(lean_object* v_mvarId_302_, lean_object* v_f_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1(v_mvarId_302_, v_f_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_310_, lean_object* v_x_311_, lean_object* v_x_312_, lean_object* v_x_313_){
_start:
{
lean_object* v_ks_314_; lean_object* v_vs_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_339_; 
v_ks_314_ = lean_ctor_get(v_x_310_, 0);
v_vs_315_ = lean_ctor_get(v_x_310_, 1);
v_isSharedCheck_339_ = !lean_is_exclusive(v_x_310_);
if (v_isSharedCheck_339_ == 0)
{
v___x_317_ = v_x_310_;
v_isShared_318_ = v_isSharedCheck_339_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_vs_315_);
lean_inc(v_ks_314_);
lean_dec(v_x_310_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_339_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_319_ = lean_array_get_size(v_ks_314_);
v___x_320_ = lean_nat_dec_lt(v_x_311_, v___x_319_);
if (v___x_320_ == 0)
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_324_; 
lean_dec(v_x_311_);
v___x_321_ = lean_array_push(v_ks_314_, v_x_312_);
v___x_322_ = lean_array_push(v_vs_315_, v_x_313_);
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 1, v___x_322_);
lean_ctor_set(v___x_317_, 0, v___x_321_);
v___x_324_ = v___x_317_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_321_);
lean_ctor_set(v_reuseFailAlloc_325_, 1, v___x_322_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
else
{
lean_object* v_k_x27_326_; uint8_t v___x_327_; 
v_k_x27_326_ = lean_array_fget_borrowed(v_ks_314_, v_x_311_);
v___x_327_ = l_Lean_instBEqFVarId_beq(v_x_312_, v_k_x27_326_);
if (v___x_327_ == 0)
{
lean_object* v___x_329_; 
if (v_isShared_318_ == 0)
{
v___x_329_ = v___x_317_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_ks_314_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_vs_315_);
v___x_329_ = v_reuseFailAlloc_333_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = lean_unsigned_to_nat(1u);
v___x_331_ = lean_nat_add(v_x_311_, v___x_330_);
lean_dec(v_x_311_);
v_x_310_ = v___x_329_;
v_x_311_ = v___x_331_;
goto _start;
}
}
else
{
lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_337_; 
v___x_334_ = lean_array_fset(v_ks_314_, v_x_311_, v_x_312_);
v___x_335_ = lean_array_fset(v_vs_315_, v_x_311_, v_x_313_);
lean_dec(v_x_311_);
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 1, v___x_335_);
lean_ctor_set(v___x_317_, 0, v___x_334_);
v___x_337_ = v___x_317_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_334_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v___x_335_);
v___x_337_ = v_reuseFailAlloc_338_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
return v___x_337_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_n_340_, lean_object* v_k_341_, lean_object* v_v_342_){
_start:
{
lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_n_340_, v___x_343_, v_k_341_, v_v_342_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(lean_object* v_x_345_, size_t v_x_346_, size_t v_x_347_, lean_object* v_x_348_, lean_object* v_x_349_){
_start:
{
if (lean_obj_tag(v_x_345_) == 0)
{
lean_object* v_es_350_; size_t v___x_351_; size_t v___x_352_; lean_object* v_j_353_; lean_object* v___x_354_; uint8_t v___x_355_; 
v_es_350_ = lean_ctor_get(v_x_345_, 0);
v___x_351_ = ((size_t)31ULL);
v___x_352_ = lean_usize_land(v_x_346_, v___x_351_);
v_j_353_ = lean_usize_to_nat(v___x_352_);
v___x_354_ = lean_array_get_size(v_es_350_);
v___x_355_ = lean_nat_dec_lt(v_j_353_, v___x_354_);
if (v___x_355_ == 0)
{
lean_dec(v_j_353_);
lean_dec(v_x_349_);
lean_dec(v_x_348_);
return v_x_345_;
}
else
{
lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_394_; 
lean_inc_ref(v_es_350_);
v_isSharedCheck_394_ = !lean_is_exclusive(v_x_345_);
if (v_isSharedCheck_394_ == 0)
{
lean_object* v_unused_395_; 
v_unused_395_ = lean_ctor_get(v_x_345_, 0);
lean_dec(v_unused_395_);
v___x_357_ = v_x_345_;
v_isShared_358_ = v_isSharedCheck_394_;
goto v_resetjp_356_;
}
else
{
lean_dec(v_x_345_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_394_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v_v_359_; lean_object* v___x_360_; lean_object* v_xs_x27_361_; lean_object* v___y_363_; 
v_v_359_ = lean_array_fget(v_es_350_, v_j_353_);
v___x_360_ = lean_box(0);
v_xs_x27_361_ = lean_array_fset(v_es_350_, v_j_353_, v___x_360_);
switch(lean_obj_tag(v_v_359_))
{
case 0:
{
lean_object* v_key_368_; lean_object* v_val_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_379_; 
v_key_368_ = lean_ctor_get(v_v_359_, 0);
v_val_369_ = lean_ctor_get(v_v_359_, 1);
v_isSharedCheck_379_ = !lean_is_exclusive(v_v_359_);
if (v_isSharedCheck_379_ == 0)
{
v___x_371_ = v_v_359_;
v_isShared_372_ = v_isSharedCheck_379_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_val_369_);
lean_inc(v_key_368_);
lean_dec(v_v_359_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_379_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
uint8_t v___x_373_; 
v___x_373_ = l_Lean_instBEqFVarId_beq(v_x_348_, v_key_368_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; lean_object* v___x_375_; 
lean_del_object(v___x_371_);
v___x_374_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_368_, v_val_369_, v_x_348_, v_x_349_);
v___x_375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
v___y_363_ = v___x_375_;
goto v___jp_362_;
}
else
{
lean_object* v___x_377_; 
lean_dec(v_val_369_);
lean_dec(v_key_368_);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 1, v_x_349_);
lean_ctor_set(v___x_371_, 0, v_x_348_);
v___x_377_ = v___x_371_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_x_348_);
lean_ctor_set(v_reuseFailAlloc_378_, 1, v_x_349_);
v___x_377_ = v_reuseFailAlloc_378_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
v___y_363_ = v___x_377_;
goto v___jp_362_;
}
}
}
}
case 1:
{
lean_object* v_node_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_392_; 
v_node_380_ = lean_ctor_get(v_v_359_, 0);
v_isSharedCheck_392_ = !lean_is_exclusive(v_v_359_);
if (v_isSharedCheck_392_ == 0)
{
v___x_382_ = v_v_359_;
v_isShared_383_ = v_isSharedCheck_392_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_node_380_);
lean_dec(v_v_359_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_392_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
size_t v___x_384_; size_t v___x_385_; size_t v___x_386_; size_t v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
v___x_384_ = ((size_t)5ULL);
v___x_385_ = lean_usize_shift_right(v_x_346_, v___x_384_);
v___x_386_ = ((size_t)1ULL);
v___x_387_ = lean_usize_add(v_x_347_, v___x_386_);
v___x_388_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(v_node_380_, v___x_385_, v___x_387_, v_x_348_, v_x_349_);
if (v_isShared_383_ == 0)
{
lean_ctor_set(v___x_382_, 0, v___x_388_);
v___x_390_ = v___x_382_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_388_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
v___y_363_ = v___x_390_;
goto v___jp_362_;
}
}
}
default: 
{
lean_object* v___x_393_; 
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_x_348_);
lean_ctor_set(v___x_393_, 1, v_x_349_);
v___y_363_ = v___x_393_;
goto v___jp_362_;
}
}
v___jp_362_:
{
lean_object* v___x_364_; lean_object* v___x_366_; 
v___x_364_ = lean_array_fset(v_xs_x27_361_, v_j_353_, v___y_363_);
lean_dec(v_j_353_);
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 0, v___x_364_);
v___x_366_ = v___x_357_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v___x_364_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
}
}
else
{
lean_object* v_ks_396_; lean_object* v_vs_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_417_; 
v_ks_396_ = lean_ctor_get(v_x_345_, 0);
v_vs_397_ = lean_ctor_get(v_x_345_, 1);
v_isSharedCheck_417_ = !lean_is_exclusive(v_x_345_);
if (v_isSharedCheck_417_ == 0)
{
v___x_399_ = v_x_345_;
v_isShared_400_ = v_isSharedCheck_417_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_vs_397_);
lean_inc(v_ks_396_);
lean_dec(v_x_345_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_417_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_ks_396_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v_vs_397_);
v___x_402_ = v_reuseFailAlloc_416_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
lean_object* v_newNode_403_; uint8_t v___y_405_; size_t v___x_411_; uint8_t v___x_412_; 
v_newNode_403_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2___redArg(v___x_402_, v_x_348_, v_x_349_);
v___x_411_ = ((size_t)7ULL);
v___x_412_ = lean_usize_dec_le(v___x_411_, v_x_347_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_413_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_403_);
v___x_414_ = lean_unsigned_to_nat(4u);
v___x_415_ = lean_nat_dec_lt(v___x_413_, v___x_414_);
lean_dec(v___x_413_);
v___y_405_ = v___x_415_;
goto v___jp_404_;
}
else
{
v___y_405_ = v___x_412_;
goto v___jp_404_;
}
v___jp_404_:
{
if (v___y_405_ == 0)
{
lean_object* v_ks_406_; lean_object* v_vs_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v_ks_406_ = lean_ctor_get(v_newNode_403_, 0);
lean_inc_ref(v_ks_406_);
v_vs_407_ = lean_ctor_get(v_newNode_403_, 1);
lean_inc_ref(v_vs_407_);
lean_dec_ref(v_newNode_403_);
v___x_408_ = lean_unsigned_to_nat(0u);
v___x_409_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg___closed__0);
v___x_410_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg(v_x_347_, v_ks_406_, v_vs_407_, v___x_408_, v___x_409_);
lean_dec_ref(v_vs_407_);
lean_dec_ref(v_ks_406_);
return v___x_410_;
}
else
{
return v_newNode_403_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg(size_t v_depth_418_, lean_object* v_keys_419_, lean_object* v_vals_420_, lean_object* v_i_421_, lean_object* v_entries_422_){
_start:
{
lean_object* v___x_423_; uint8_t v___x_424_; 
v___x_423_ = lean_array_get_size(v_keys_419_);
v___x_424_ = lean_nat_dec_lt(v_i_421_, v___x_423_);
if (v___x_424_ == 0)
{
lean_dec(v_i_421_);
return v_entries_422_;
}
else
{
lean_object* v_k_425_; lean_object* v_v_426_; uint64_t v___x_427_; size_t v_h_428_; size_t v___x_429_; lean_object* v___x_430_; size_t v___x_431_; size_t v___x_432_; size_t v___x_433_; size_t v_h_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v_k_425_ = lean_array_fget_borrowed(v_keys_419_, v_i_421_);
v_v_426_ = lean_array_fget_borrowed(v_vals_420_, v_i_421_);
v___x_427_ = l_Lean_instHashableFVarId_hash(v_k_425_);
v_h_428_ = lean_uint64_to_usize(v___x_427_);
v___x_429_ = ((size_t)5ULL);
v___x_430_ = lean_unsigned_to_nat(1u);
v___x_431_ = ((size_t)1ULL);
v___x_432_ = lean_usize_sub(v_depth_418_, v___x_431_);
v___x_433_ = lean_usize_mul(v___x_429_, v___x_432_);
v_h_434_ = lean_usize_shift_right(v_h_428_, v___x_433_);
v___x_435_ = lean_nat_add(v_i_421_, v___x_430_);
lean_dec(v_i_421_);
lean_inc(v_v_426_);
lean_inc(v_k_425_);
v___x_436_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(v_entries_422_, v_h_434_, v_depth_418_, v_k_425_, v_v_426_);
v_i_421_ = v___x_435_;
v_entries_422_ = v___x_436_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_depth_438_, lean_object* v_keys_439_, lean_object* v_vals_440_, lean_object* v_i_441_, lean_object* v_entries_442_){
_start:
{
size_t v_depth_boxed_443_; lean_object* v_res_444_; 
v_depth_boxed_443_ = lean_unbox_usize(v_depth_438_);
lean_dec(v_depth_438_);
v_res_444_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_boxed_443_, v_keys_439_, v_vals_440_, v_i_441_, v_entries_442_);
lean_dec_ref(v_vals_440_);
lean_dec_ref(v_keys_439_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_445_, lean_object* v_x_446_, lean_object* v_x_447_, lean_object* v_x_448_, lean_object* v_x_449_){
_start:
{
size_t v_x_1478__boxed_450_; size_t v_x_1479__boxed_451_; lean_object* v_res_452_; 
v_x_1478__boxed_450_ = lean_unbox_usize(v_x_446_);
lean_dec(v_x_446_);
v_x_1479__boxed_451_ = lean_unbox_usize(v_x_447_);
lean_dec(v_x_447_);
v_res_452_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(v_x_445_, v_x_1478__boxed_450_, v_x_1479__boxed_451_, v_x_448_, v_x_449_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0___redArg(lean_object* v_x_453_, lean_object* v_x_454_, lean_object* v_x_455_){
_start:
{
uint64_t v___x_456_; size_t v___x_457_; size_t v___x_458_; lean_object* v___x_459_; 
v___x_456_ = l_Lean_instHashableFVarId_hash(v_x_454_);
v___x_457_ = lean_uint64_to_usize(v___x_456_);
v___x_458_ = ((size_t)1ULL);
v___x_459_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(v_x_453_, v___x_457_, v___x_458_, v_x_454_, v_x_455_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0___lam__0(lean_object* v_fvarId_460_, lean_object* v_f_461_, lean_object* v_lctx_462_){
_start:
{
lean_object* v_fvarIdToDecl_463_; lean_object* v_decls_464_; lean_object* v_auxDeclToFullName_465_; lean_object* v___x_466_; 
v_fvarIdToDecl_463_ = lean_ctor_get(v_lctx_462_, 0);
v_decls_464_ = lean_ctor_get(v_lctx_462_, 1);
v_auxDeclToFullName_465_ = lean_ctor_get(v_lctx_462_, 2);
lean_inc_ref(v_lctx_462_);
v___x_466_ = lean_local_ctx_find(v_lctx_462_, v_fvarId_460_);
if (lean_obj_tag(v___x_466_) == 0)
{
lean_dec_ref(v_f_461_);
return v_lctx_462_;
}
else
{
lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_491_; 
lean_inc(v_auxDeclToFullName_465_);
lean_inc_ref(v_decls_464_);
lean_inc_ref(v_fvarIdToDecl_463_);
v_isSharedCheck_491_ = !lean_is_exclusive(v_lctx_462_);
if (v_isSharedCheck_491_ == 0)
{
lean_object* v_unused_492_; lean_object* v_unused_493_; lean_object* v_unused_494_; 
v_unused_492_ = lean_ctor_get(v_lctx_462_, 2);
lean_dec(v_unused_492_);
v_unused_493_ = lean_ctor_get(v_lctx_462_, 1);
lean_dec(v_unused_493_);
v_unused_494_ = lean_ctor_get(v_lctx_462_, 0);
lean_dec(v_unused_494_);
v___x_468_ = v_lctx_462_;
v_isShared_469_ = v_isSharedCheck_491_;
goto v_resetjp_467_;
}
else
{
lean_dec(v_lctx_462_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_491_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v_val_470_; lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_490_; 
v_val_470_ = lean_ctor_get(v___x_466_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_466_);
if (v_isSharedCheck_490_ == 0)
{
v___x_472_ = v___x_466_;
v_isShared_473_ = v_isSharedCheck_490_;
goto v_resetjp_471_;
}
else
{
lean_inc(v_val_470_);
lean_dec(v___x_466_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_490_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
lean_object* v_decl_474_; lean_object* v___y_476_; lean_object* v___y_477_; lean_object* v___y_486_; lean_object* v_fvarId_489_; 
v_decl_474_ = lean_apply_1(v_f_461_, v_val_470_);
v_fvarId_489_ = lean_ctor_get(v_decl_474_, 1);
lean_inc(v_fvarId_489_);
v___y_486_ = v_fvarId_489_;
goto v___jp_485_;
v___jp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 0, v_decl_474_);
v___x_479_ = v___x_472_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_decl_474_);
v___x_479_ = v_reuseFailAlloc_484_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
lean_object* v___x_480_; lean_object* v___x_482_; 
v___x_480_ = l_Lean_PersistentArray_set___redArg(v_decls_464_, v___y_477_, v___x_479_);
lean_dec(v___y_477_);
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 1, v___x_480_);
lean_ctor_set(v___x_468_, 0, v___y_476_);
v___x_482_ = v___x_468_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v___y_476_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v___x_480_);
lean_ctor_set(v_reuseFailAlloc_483_, 2, v_auxDeclToFullName_465_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
return v___x_482_;
}
}
}
v___jp_485_:
{
lean_object* v___x_487_; lean_object* v_index_488_; 
lean_inc_ref(v_decl_474_);
v___x_487_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0___redArg(v_fvarIdToDecl_463_, v___y_486_, v_decl_474_);
v_index_488_ = lean_ctor_get(v_decl_474_, 0);
lean_inc(v_index_488_);
v___y_476_ = v___x_487_;
v___y_477_ = v_index_488_;
goto v___jp_475_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0(lean_object* v_mvarId_495_, lean_object* v_fvarId_496_, lean_object* v_f_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_){
_start:
{
lean_object* v___f_503_; lean_object* v___x_504_; 
v___f_503_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0___lam__0), 3, 2);
lean_closure_set(v___f_503_, 0, v_fvarId_496_);
lean_closure_set(v___f_503_, 1, v_f_497_);
v___x_504_ = lp_mathlib_Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1(v_mvarId_495_, v___f_503_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0___boxed(lean_object* v_mvarId_505_, lean_object* v_fvarId_506_, lean_object* v_f_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
lean_object* v_res_513_; 
v_res_513_ = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0(v_mvarId_505_, v_fvarId_506_, v_f_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
lean_dec(v___y_511_);
lean_dec_ref(v___y_510_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
return v_res_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp(lean_object* v_mvarId_514_, lean_object* v_fvarId_515_, lean_object* v_old_516_, lean_object* v_new_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_){
_start:
{
lean_object* v___f_523_; lean_object* v___x_524_; 
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_renameBVarHyp___lam__0___boxed), 3, 2);
lean_closure_set(v___f_523_, 0, v_old_516_);
lean_closure_set(v___f_523_, 1, v_new_517_);
v___x_524_ = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0(v_mvarId_514_, v_fvarId_515_, v___f_523_, v_a_518_, v_a_519_, v_a_520_, v_a_521_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarHyp___boxed(lean_object* v_mvarId_525_, lean_object* v_fvarId_526_, lean_object* v_old_527_, lean_object* v_new_528_, lean_object* v_a_529_, lean_object* v_a_530_, lean_object* v_a_531_, lean_object* v_a_532_, lean_object* v_a_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_Mathlib_Tactic_renameBVarHyp(v_mvarId_525_, v_fvarId_526_, v_old_527_, v_new_528_, v_a_529_, v_a_530_, v_a_531_, v_a_532_);
lean_dec(v_a_532_);
lean_dec_ref(v_a_531_);
lean_dec(v_a_530_);
lean_dec_ref(v_a_529_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0(lean_object* v_00_u03b2_535_, lean_object* v_x_536_, lean_object* v_x_537_, lean_object* v_x_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0___redArg(v_x_536_, v_x_537_, v_x_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_540_, lean_object* v_x_541_, size_t v_x_542_, size_t v_x_543_, lean_object* v_x_544_, lean_object* v_x_545_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___redArg(v_x_541_, v_x_542_, v_x_543_, v_x_544_, v_x_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_547_, lean_object* v_x_548_, lean_object* v_x_549_, lean_object* v_x_550_, lean_object* v_x_551_, lean_object* v_x_552_){
_start:
{
size_t v_x_1767__boxed_553_; size_t v_x_1768__boxed_554_; lean_object* v_res_555_; 
v_x_1767__boxed_553_ = lean_unbox_usize(v_x_549_);
lean_dec(v_x_549_);
v_x_1768__boxed_554_ = lean_unbox_usize(v_x_550_);
lean_dec(v_x_550_);
v_res_555_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1(v_00_u03b2_547_, v_x_548_, v_x_1767__boxed_553_, v_x_1768__boxed_554_, v_x_551_, v_x_552_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3(lean_object* v_mvarId_556_, lean_object* v_f_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(v_mvarId_556_, v_f_557_, v___y_559_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___boxed(lean_object* v_mvarId_564_, lean_object* v_f_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3(v_mvarId_564_, v_f_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
lean_dec(v___y_567_);
lean_dec_ref(v___y_566_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_572_, lean_object* v_n_573_, lean_object* v_k_574_, lean_object* v_v_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2___redArg(v_n_573_, v_k_574_, v_v_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_577_, size_t v_depth_578_, lean_object* v_keys_579_, lean_object* v_vals_580_, lean_object* v_heq_581_, lean_object* v_i_582_, lean_object* v_entries_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_578_, v_keys_579_, v_vals_580_, v_i_582_, v_entries_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b2_585_, lean_object* v_depth_586_, lean_object* v_keys_587_, lean_object* v_vals_588_, lean_object* v_heq_589_, lean_object* v_i_590_, lean_object* v_entries_591_){
_start:
{
size_t v_depth_boxed_592_; lean_object* v_res_593_; 
v_depth_boxed_592_ = lean_unbox_usize(v_depth_586_);
lean_dec(v_depth_586_);
v_res_593_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__3(v_00_u03b2_585_, v_depth_boxed_592_, v_keys_587_, v_vals_588_, v_heq_589_, v_i_590_, v_entries_591_);
lean_dec_ref(v_vals_588_);
lean_dec_ref(v_keys_587_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6(lean_object* v_00_u03b2_594_, lean_object* v_x_595_, lean_object* v_x_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___redArg(v_x_595_, v_x_596_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6___boxed(lean_object* v_00_u03b2_598_, lean_object* v_x_599_, lean_object* v_x_600_){
_start:
{
lean_object* v_res_601_; 
v_res_601_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6(v_00_u03b2_598_, v_x_599_, v_x_600_);
lean_dec(v_x_600_);
lean_dec_ref(v_x_599_);
return v_res_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7(lean_object* v_00_u03b2_602_, lean_object* v_x_603_, lean_object* v_x_604_, lean_object* v_x_605_){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7___redArg(v_x_603_, v_x_604_, v_x_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_607_, lean_object* v_x_608_, lean_object* v_x_609_, lean_object* v_x_610_, lean_object* v_x_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_x_608_, v_x_609_, v_x_610_, v_x_611_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8(lean_object* v_00_u03b2_613_, lean_object* v_x_614_, size_t v_x_615_, lean_object* v_x_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___redArg(v_x_614_, v_x_615_, v_x_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8___boxed(lean_object* v_00_u03b2_618_, lean_object* v_x_619_, lean_object* v_x_620_, lean_object* v_x_621_){
_start:
{
size_t v_x_1831__boxed_622_; lean_object* v_res_623_; 
v_x_1831__boxed_622_ = lean_unbox_usize(v_x_620_);
lean_dec(v_x_620_);
v_res_623_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8(v_00_u03b2_618_, v_x_619_, v_x_1831__boxed_622_, v_x_621_);
lean_dec(v_x_621_);
lean_dec_ref(v_x_619_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10(lean_object* v_00_u03b2_624_, lean_object* v_x_625_, size_t v_x_626_, size_t v_x_627_, lean_object* v_x_628_, lean_object* v_x_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___redArg(v_x_625_, v_x_626_, v_x_627_, v_x_628_, v_x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10___boxed(lean_object* v_00_u03b2_631_, lean_object* v_x_632_, lean_object* v_x_633_, lean_object* v_x_634_, lean_object* v_x_635_, lean_object* v_x_636_){
_start:
{
size_t v_x_1842__boxed_637_; size_t v_x_1843__boxed_638_; lean_object* v_res_639_; 
v_x_1842__boxed_637_ = lean_unbox_usize(v_x_633_);
lean_dec(v_x_633_);
v_x_1843__boxed_638_ = lean_unbox_usize(v_x_634_);
lean_dec(v_x_634_);
v_res_639_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10(v_00_u03b2_631_, v_x_632_, v_x_1842__boxed_637_, v_x_1843__boxed_638_, v_x_635_, v_x_636_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9(lean_object* v_00_u03b2_640_, lean_object* v_keys_641_, lean_object* v_vals_642_, lean_object* v_heq_643_, lean_object* v_i_644_, lean_object* v_k_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___redArg(v_keys_641_, v_vals_642_, v_i_644_, v_k_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9___boxed(lean_object* v_00_u03b2_647_, lean_object* v_keys_648_, lean_object* v_vals_649_, lean_object* v_heq_650_, lean_object* v_i_651_, lean_object* v_k_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__6_spec__8_spec__9(v_00_u03b2_647_, v_keys_648_, v_vals_649_, v_heq_650_, v_i_651_, v_k_652_);
lean_dec(v_k_652_);
lean_dec_ref(v_vals_649_);
lean_dec_ref(v_keys_648_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12(lean_object* v_00_u03b2_654_, lean_object* v_n_655_, lean_object* v_k_656_, lean_object* v_v_657_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12___redArg(v_n_655_, v_k_656_, v_v_657_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13(lean_object* v_00_u03b2_659_, size_t v_depth_660_, lean_object* v_keys_661_, lean_object* v_vals_662_, lean_object* v_heq_663_, lean_object* v_i_664_, lean_object* v_entries_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___redArg(v_depth_660_, v_keys_661_, v_vals_662_, v_i_664_, v_entries_665_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13___boxed(lean_object* v_00_u03b2_667_, lean_object* v_depth_668_, lean_object* v_keys_669_, lean_object* v_vals_670_, lean_object* v_heq_671_, lean_object* v_i_672_, lean_object* v_entries_673_){
_start:
{
size_t v_depth_boxed_674_; lean_object* v_res_675_; 
v_depth_boxed_674_ = lean_unbox_usize(v_depth_668_);
lean_dec(v_depth_668_);
v_res_675_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__13(v_00_u03b2_667_, v_depth_boxed_674_, v_keys_669_, v_vals_670_, v_heq_671_, v_i_672_, v_entries_673_);
lean_dec_ref(v_vals_670_);
lean_dec_ref(v_keys_669_);
return v_res_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13(lean_object* v_00_u03b2_676_, lean_object* v_x_677_, lean_object* v_x_678_, lean_object* v_x_679_, lean_object* v_x_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7_spec__10_spec__12_spec__13___redArg(v_x_677_, v_x_678_, v_x_679_, v_x_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0(lean_object* v_old_682_, lean_object* v_new_683_, lean_object* v_e_684_){
_start:
{
lean_object* v___x_685_; 
v___x_685_ = lp_mathlib_Lean_Expr_renameBVar(v_e_684_, v_old_682_, v_new_683_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0___boxed(lean_object* v_old_686_, lean_object* v_new_687_, lean_object* v_e_688_){
_start:
{
lean_object* v_res_689_; 
v_res_689_ = lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0(v_old_686_, v_new_687_, v_e_688_);
lean_dec(v_old_686_);
return v_res_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg___lam__0(lean_object* v_f_690_, lean_object* v_mdecl_691_){
_start:
{
lean_object* v_userName_692_; lean_object* v_lctx_693_; lean_object* v_type_694_; lean_object* v_depth_695_; lean_object* v_localInstances_696_; uint8_t v_kind_697_; lean_object* v_numScopeArgs_698_; lean_object* v_index_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_707_; 
v_userName_692_ = lean_ctor_get(v_mdecl_691_, 0);
v_lctx_693_ = lean_ctor_get(v_mdecl_691_, 1);
v_type_694_ = lean_ctor_get(v_mdecl_691_, 2);
v_depth_695_ = lean_ctor_get(v_mdecl_691_, 3);
v_localInstances_696_ = lean_ctor_get(v_mdecl_691_, 4);
v_kind_697_ = lean_ctor_get_uint8(v_mdecl_691_, sizeof(void*)*7);
v_numScopeArgs_698_ = lean_ctor_get(v_mdecl_691_, 5);
v_index_699_ = lean_ctor_get(v_mdecl_691_, 6);
v_isSharedCheck_707_ = !lean_is_exclusive(v_mdecl_691_);
if (v_isSharedCheck_707_ == 0)
{
v___x_701_ = v_mdecl_691_;
v_isShared_702_ = v_isSharedCheck_707_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_index_699_);
lean_inc(v_numScopeArgs_698_);
lean_inc(v_localInstances_696_);
lean_inc(v_depth_695_);
lean_inc(v_type_694_);
lean_inc(v_lctx_693_);
lean_inc(v_userName_692_);
lean_dec(v_mdecl_691_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_707_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_703_; lean_object* v___x_705_; 
v___x_703_ = lean_apply_1(v_f_690_, v_type_694_);
if (v_isShared_702_ == 0)
{
lean_ctor_set(v___x_701_, 2, v___x_703_);
v___x_705_ = v___x_701_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v_userName_692_);
lean_ctor_set(v_reuseFailAlloc_706_, 1, v_lctx_693_);
lean_ctor_set(v_reuseFailAlloc_706_, 2, v___x_703_);
lean_ctor_set(v_reuseFailAlloc_706_, 3, v_depth_695_);
lean_ctor_set(v_reuseFailAlloc_706_, 4, v_localInstances_696_);
lean_ctor_set(v_reuseFailAlloc_706_, 5, v_numScopeArgs_698_);
lean_ctor_set(v_reuseFailAlloc_706_, 6, v_index_699_);
lean_ctor_set_uint8(v_reuseFailAlloc_706_, sizeof(void*)*7, v_kind_697_);
v___x_705_ = v_reuseFailAlloc_706_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
return v___x_705_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg(lean_object* v_mvarId_708_, lean_object* v_f_709_, lean_object* v___y_710_){
_start:
{
lean_object* v___f_712_; lean_object* v___x_713_; 
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_712_, 0, v_f_709_);
v___x_713_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3___redArg(v_mvarId_708_, v___f_712_, v___y_710_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg___boxed(lean_object* v_mvarId_714_, lean_object* v_f_715_, lean_object* v___y_716_, lean_object* v___y_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg(v_mvarId_714_, v_f_715_, v___y_716_);
lean_dec(v___y_716_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget(lean_object* v_mvarId_719_, lean_object* v_old_720_, lean_object* v_new_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_){
_start:
{
lean_object* v___f_727_; lean_object* v___x_728_; 
v___f_727_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_renameBVarTarget___lam__0___boxed), 3, 2);
lean_closure_set(v___f_727_, 0, v_old_720_);
lean_closure_set(v___f_727_, 1, v_new_721_);
v___x_728_ = lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg(v_mvarId_719_, v___f_727_, v_a_723_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_renameBVarTarget___boxed(lean_object* v_mvarId_729_, lean_object* v_old_730_, lean_object* v_new_731_, lean_object* v_a_732_, lean_object* v_a_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_mathlib_Mathlib_Tactic_renameBVarTarget(v_mvarId_729_, v_old_730_, v_new_731_, v_a_732_, v_a_733_, v_a_734_, v_a_735_);
lean_dec(v_a_735_);
lean_dec_ref(v_a_734_);
lean_dec(v_a_733_);
lean_dec_ref(v_a_732_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0(lean_object* v_mvarId_738_, lean_object* v_f_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___redArg(v_mvarId_738_, v_f_739_, v___y_741_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0___boxed(lean_object* v_mvarId_746_, lean_object* v_f_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_){
_start:
{
lean_object* v_res_753_; 
v_res_753_ = lp_mathlib_Mathlib_Tactic_modifyTarget___at___00Mathlib_Tactic_renameBVarTarget_spec__0(v_mvarId_746_, v_f_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
lean_dec(v___y_749_);
lean_dec_ref(v___y_748_);
return v_res_753_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18(void){
_start:
{
lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; 
v___x_791_ = l_Lean_Parser_Tactic_location;
v___x_792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__17));
v___x_793_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_793_, 0, v___x_792_);
lean_ctor_set(v___x_793_, 1, v___x_791_);
return v___x_793_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19(void){
_start:
{
lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; 
v___x_794_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18, &lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18_once, _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__18);
v___x_795_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__15));
v___x_796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__5));
v___x_797_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_797_, 0, v___x_796_);
lean_ctor_set(v___x_797_, 1, v___x_795_);
lean_ctor_set(v___x_797_, 2, v___x_794_);
return v___x_797_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20(void){
_start:
{
lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; 
v___x_798_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19, &lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19_once, _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__19);
v___x_799_ = lean_unsigned_to_nat(1022u);
v___x_800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3));
v___x_801_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
lean_ctor_set(v___x_801_, 1, v___x_799_);
lean_ctor_set(v___x_801_, 2, v___x_798_);
return v___x_801_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192____(void){
_start:
{
lean_object* v___x_802_; 
v___x_802_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20, &lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20_once, _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__20);
return v___x_802_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; 
v___x_803_ = lean_box(0);
v___x_804_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_805_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_805_, 0, v___x_804_);
lean_ctor_set(v___x_805_, 1, v___x_803_);
return v___x_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg(){
_start:
{
lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_807_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___closed__0);
v___x_808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_808_, 0, v___x_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg___boxed(lean_object* v___y_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg();
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0(lean_object* v_00_u03b1_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_){
_start:
{
lean_object* v___x_821_; 
v___x_821_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg();
return v___x_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___boxed(lean_object* v_00_u03b1_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0(v_00_u03b1_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_);
lean_dec(v___y_830_);
lean_dec_ref(v___y_829_);
lean_dec(v___y_828_);
lean_dec_ref(v___y_827_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
lean_dec(v___y_824_);
lean_dec_ref(v___y_823_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1(lean_object* v_msgData_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v___x_839_; lean_object* v_env_840_; lean_object* v___x_841_; lean_object* v_mctx_842_; lean_object* v_lctx_843_; lean_object* v_options_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v___x_839_ = lean_st_ref_get(v___y_837_);
v_env_840_ = lean_ctor_get(v___x_839_, 0);
lean_inc_ref(v_env_840_);
lean_dec(v___x_839_);
v___x_841_ = lean_st_ref_get(v___y_835_);
v_mctx_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc_ref(v_mctx_842_);
lean_dec(v___x_841_);
v_lctx_843_ = lean_ctor_get(v___y_834_, 2);
v_options_844_ = lean_ctor_get(v___y_836_, 2);
lean_inc_ref(v_options_844_);
lean_inc_ref(v_lctx_843_);
v___x_845_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_845_, 0, v_env_840_);
lean_ctor_set(v___x_845_, 1, v_mctx_842_);
lean_ctor_set(v___x_845_, 2, v_lctx_843_);
lean_ctor_set(v___x_845_, 3, v_options_844_);
v___x_846_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_846_, 0, v___x_845_);
lean_ctor_set(v___x_846_, 1, v_msgData_833_);
v___x_847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_847_, 0, v___x_846_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1___boxed(lean_object* v_msgData_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1(v_msgData_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
return v_res_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg(lean_object* v_msg_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
lean_object* v_ref_861_; lean_object* v___x_862_; lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_871_; 
v_ref_861_ = lean_ctor_get(v___y_858_, 5);
v___x_862_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1_spec__1(v_msg_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
v_a_863_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_871_ == 0)
{
v___x_865_ = v___x_862_;
v_isShared_866_ = v_isSharedCheck_871_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___x_862_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_871_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v___x_867_; lean_object* v___x_869_; 
lean_inc(v_ref_861_);
v___x_867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_867_, 0, v_ref_861_);
lean_ctor_set(v___x_867_, 1, v_a_863_);
if (v_isShared_866_ == 0)
{
lean_ctor_set_tag(v___x_865_, 1);
lean_ctor_set(v___x_865_, 0, v___x_867_);
v___x_869_ = v___x_865_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v___x_867_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg___boxed(lean_object* v_msg_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg(v_msg_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
lean_dec(v___y_876_);
lean_dec_ref(v___y_875_);
lean_dec(v___y_874_);
lean_dec_ref(v___y_873_);
return v_res_878_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_880_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__0));
v___x_881_ = l_Lean_stringToMessageData(v___x_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0(lean_object* v_x_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; 
v___x_892_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___closed__1);
v___x_893_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg(v___x_892_, v___y_887_, v___y_888_, v___y_889_, v___y_890_);
return v___x_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0___boxed(lean_object* v_x_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__0(v_x_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, v___y_902_);
lean_dec(v___y_902_);
lean_dec_ref(v___y_901_);
lean_dec(v___y_900_);
lean_dec_ref(v___y_899_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
lean_dec(v_x_894_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1(lean_object* v_old_905_, lean_object* v_new_906_, lean_object* v_a_907_, lean_object* v_fvarId_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_){
_start:
{
lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_918_ = l_Lean_TSyntax_getId(v_old_905_);
v___x_919_ = l_Lean_TSyntax_getId(v_new_906_);
v___x_920_ = lp_mathlib_Mathlib_Tactic_renameBVarHyp(v_a_907_, v_fvarId_908_, v___x_918_, v___x_919_, v___y_913_, v___y_914_, v___y_915_, v___y_916_);
return v___x_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1___boxed(lean_object* v_old_921_, lean_object* v_new_922_, lean_object* v_a_923_, lean_object* v_fvarId_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1(v_old_921_, v_new_922_, v_a_923_, v_fvarId_924_, v___y_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec(v___y_930_);
lean_dec_ref(v___y_929_);
lean_dec(v___y_928_);
lean_dec_ref(v___y_927_);
lean_dec(v___y_926_);
lean_dec_ref(v___y_925_);
lean_dec(v_new_922_);
lean_dec(v_old_921_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2(lean_object* v_a_935_, lean_object* v___x_936_, lean_object* v___x_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_){
_start:
{
lean_object* v___x_947_; 
v___x_947_ = lp_mathlib_Mathlib_Tactic_renameBVarTarget(v_a_935_, v___x_936_, v___x_937_, v___y_942_, v___y_943_, v___y_944_, v___y_945_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2___boxed(lean_object* v_a_948_, lean_object* v___x_949_, lean_object* v___x_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2(v_a_948_, v___x_949_, v___x_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
lean_dec(v___y_954_);
lean_dec_ref(v___y_953_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg(lean_object* v_t_961_, lean_object* v_k_962_){
_start:
{
if (lean_obj_tag(v_t_961_) == 0)
{
lean_object* v_k_963_; lean_object* v_v_964_; lean_object* v_l_965_; lean_object* v_r_966_; uint8_t v___x_967_; 
v_k_963_ = lean_ctor_get(v_t_961_, 1);
v_v_964_ = lean_ctor_get(v_t_961_, 2);
v_l_965_ = lean_ctor_get(v_t_961_, 3);
v_r_966_ = lean_ctor_get(v_t_961_, 4);
v___x_967_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_962_, v_k_963_);
switch(v___x_967_)
{
case 0:
{
v_t_961_ = v_l_965_;
goto _start;
}
case 1:
{
lean_object* v___x_969_; 
lean_inc(v_v_964_);
v___x_969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_969_, 0, v_v_964_);
return v___x_969_;
}
default: 
{
v_t_961_ = v_r_966_;
goto _start;
}
}
}
else
{
lean_object* v___x_971_; 
v___x_971_ = lean_box(0);
return v___x_971_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_t_972_, lean_object* v_k_973_){
_start:
{
lean_object* v_res_974_; 
v_res_974_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg(v_t_972_, v_k_973_);
lean_dec(v_k_973_);
lean_dec(v_t_972_);
return v_res_974_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0(void){
_start:
{
lean_object* v___x_975_; 
v___x_975_ = l_instMonadEIO(lean_box(0));
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5(lean_object* v_msg_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_){
_start:
{
lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v_toApplicative_996_; lean_object* v___x_998_; uint8_t v_isShared_999_; uint8_t v_isSharedCheck_1117_; 
v___x_994_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0, &lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0_once, _init_lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__0);
v___x_995_ = l_StateRefT_x27_instMonad___redArg(v___x_994_);
v_toApplicative_996_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1117_ == 0)
{
lean_object* v_unused_1118_; 
v_unused_1118_ = lean_ctor_get(v___x_995_, 1);
lean_dec(v_unused_1118_);
v___x_998_ = v___x_995_;
v_isShared_999_ = v_isSharedCheck_1117_;
goto v_resetjp_997_;
}
else
{
lean_inc(v_toApplicative_996_);
lean_dec(v___x_995_);
v___x_998_ = lean_box(0);
v_isShared_999_ = v_isSharedCheck_1117_;
goto v_resetjp_997_;
}
v_resetjp_997_:
{
lean_object* v_toFunctor_1000_; lean_object* v_toSeq_1001_; lean_object* v_toSeqLeft_1002_; lean_object* v_toSeqRight_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1115_; 
v_toFunctor_1000_ = lean_ctor_get(v_toApplicative_996_, 0);
v_toSeq_1001_ = lean_ctor_get(v_toApplicative_996_, 2);
v_toSeqLeft_1002_ = lean_ctor_get(v_toApplicative_996_, 3);
v_toSeqRight_1003_ = lean_ctor_get(v_toApplicative_996_, 4);
v_isSharedCheck_1115_ = !lean_is_exclusive(v_toApplicative_996_);
if (v_isSharedCheck_1115_ == 0)
{
lean_object* v_unused_1116_; 
v_unused_1116_ = lean_ctor_get(v_toApplicative_996_, 1);
lean_dec(v_unused_1116_);
v___x_1005_ = v_toApplicative_996_;
v_isShared_1006_ = v_isSharedCheck_1115_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_toSeqRight_1003_);
lean_inc(v_toSeqLeft_1002_);
lean_inc(v_toSeq_1001_);
lean_inc(v_toFunctor_1000_);
lean_dec(v_toApplicative_996_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1115_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___f_1007_; lean_object* v___f_1008_; lean_object* v___f_1009_; lean_object* v___f_1010_; lean_object* v___x_1011_; lean_object* v___f_1012_; lean_object* v___f_1013_; lean_object* v___f_1014_; lean_object* v___x_1016_; 
v___f_1007_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__1));
v___f_1008_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__2));
lean_inc_ref(v_toFunctor_1000_);
v___f_1009_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1009_, 0, v_toFunctor_1000_);
v___f_1010_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1010_, 0, v_toFunctor_1000_);
v___x_1011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1011_, 0, v___f_1009_);
lean_ctor_set(v___x_1011_, 1, v___f_1010_);
v___f_1012_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1012_, 0, v_toSeqRight_1003_);
v___f_1013_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1013_, 0, v_toSeqLeft_1002_);
v___f_1014_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1014_, 0, v_toSeq_1001_);
if (v_isShared_1006_ == 0)
{
lean_ctor_set(v___x_1005_, 4, v___f_1012_);
lean_ctor_set(v___x_1005_, 3, v___f_1013_);
lean_ctor_set(v___x_1005_, 2, v___f_1014_);
lean_ctor_set(v___x_1005_, 1, v___f_1007_);
lean_ctor_set(v___x_1005_, 0, v___x_1011_);
v___x_1016_ = v___x_1005_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v___x_1011_);
lean_ctor_set(v_reuseFailAlloc_1114_, 1, v___f_1007_);
lean_ctor_set(v_reuseFailAlloc_1114_, 2, v___f_1014_);
lean_ctor_set(v_reuseFailAlloc_1114_, 3, v___f_1013_);
lean_ctor_set(v_reuseFailAlloc_1114_, 4, v___f_1012_);
v___x_1016_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
lean_object* v___x_1018_; 
if (v_isShared_999_ == 0)
{
lean_ctor_set(v___x_998_, 1, v___f_1008_);
lean_ctor_set(v___x_998_, 0, v___x_1016_);
v___x_1018_ = v___x_998_;
goto v_reusejp_1017_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v___x_1016_);
lean_ctor_set(v_reuseFailAlloc_1113_, 1, v___f_1008_);
v___x_1018_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1017_;
}
v_reusejp_1017_:
{
lean_object* v___x_1019_; lean_object* v_toApplicative_1020_; lean_object* v___x_1022_; uint8_t v_isShared_1023_; uint8_t v_isSharedCheck_1111_; 
v___x_1019_ = l_StateRefT_x27_instMonad___redArg(v___x_1018_);
v_toApplicative_1020_ = lean_ctor_get(v___x_1019_, 0);
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1019_);
if (v_isSharedCheck_1111_ == 0)
{
lean_object* v_unused_1112_; 
v_unused_1112_ = lean_ctor_get(v___x_1019_, 1);
lean_dec(v_unused_1112_);
v___x_1022_ = v___x_1019_;
v_isShared_1023_ = v_isSharedCheck_1111_;
goto v_resetjp_1021_;
}
else
{
lean_inc(v_toApplicative_1020_);
lean_dec(v___x_1019_);
v___x_1022_ = lean_box(0);
v_isShared_1023_ = v_isSharedCheck_1111_;
goto v_resetjp_1021_;
}
v_resetjp_1021_:
{
lean_object* v_toFunctor_1024_; lean_object* v_toSeq_1025_; lean_object* v_toSeqLeft_1026_; lean_object* v_toSeqRight_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1109_; 
v_toFunctor_1024_ = lean_ctor_get(v_toApplicative_1020_, 0);
v_toSeq_1025_ = lean_ctor_get(v_toApplicative_1020_, 2);
v_toSeqLeft_1026_ = lean_ctor_get(v_toApplicative_1020_, 3);
v_toSeqRight_1027_ = lean_ctor_get(v_toApplicative_1020_, 4);
v_isSharedCheck_1109_ = !lean_is_exclusive(v_toApplicative_1020_);
if (v_isSharedCheck_1109_ == 0)
{
lean_object* v_unused_1110_; 
v_unused_1110_ = lean_ctor_get(v_toApplicative_1020_, 1);
lean_dec(v_unused_1110_);
v___x_1029_ = v_toApplicative_1020_;
v_isShared_1030_ = v_isSharedCheck_1109_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_toSeqRight_1027_);
lean_inc(v_toSeqLeft_1026_);
lean_inc(v_toSeq_1025_);
lean_inc(v_toFunctor_1024_);
lean_dec(v_toApplicative_1020_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1109_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___f_1031_; lean_object* v___f_1032_; lean_object* v___f_1033_; lean_object* v___f_1034_; lean_object* v___x_1035_; lean_object* v___f_1036_; lean_object* v___f_1037_; lean_object* v___f_1038_; lean_object* v___x_1040_; 
v___f_1031_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__3));
v___f_1032_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__4));
lean_inc_ref(v_toFunctor_1024_);
v___f_1033_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1033_, 0, v_toFunctor_1024_);
v___f_1034_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1034_, 0, v_toFunctor_1024_);
v___x_1035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1035_, 0, v___f_1033_);
lean_ctor_set(v___x_1035_, 1, v___f_1034_);
v___f_1036_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1036_, 0, v_toSeqRight_1027_);
v___f_1037_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1037_, 0, v_toSeqLeft_1026_);
v___f_1038_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1038_, 0, v_toSeq_1025_);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 4, v___f_1036_);
lean_ctor_set(v___x_1029_, 3, v___f_1037_);
lean_ctor_set(v___x_1029_, 2, v___f_1038_);
lean_ctor_set(v___x_1029_, 1, v___f_1031_);
lean_ctor_set(v___x_1029_, 0, v___x_1035_);
v___x_1040_ = v___x_1029_;
goto v_reusejp_1039_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v___x_1035_);
lean_ctor_set(v_reuseFailAlloc_1108_, 1, v___f_1031_);
lean_ctor_set(v_reuseFailAlloc_1108_, 2, v___f_1038_);
lean_ctor_set(v_reuseFailAlloc_1108_, 3, v___f_1037_);
lean_ctor_set(v_reuseFailAlloc_1108_, 4, v___f_1036_);
v___x_1040_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1039_;
}
v_reusejp_1039_:
{
lean_object* v___x_1042_; 
if (v_isShared_1023_ == 0)
{
lean_ctor_set(v___x_1022_, 1, v___f_1032_);
lean_ctor_set(v___x_1022_, 0, v___x_1040_);
v___x_1042_ = v___x_1022_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1107_; 
v_reuseFailAlloc_1107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1107_, 0, v___x_1040_);
lean_ctor_set(v_reuseFailAlloc_1107_, 1, v___f_1032_);
v___x_1042_ = v_reuseFailAlloc_1107_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
lean_object* v___x_1043_; lean_object* v_toApplicative_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1105_; 
v___x_1043_ = l_StateRefT_x27_instMonad___redArg(v___x_1042_);
v_toApplicative_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1105_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1105_ == 0)
{
lean_object* v_unused_1106_; 
v_unused_1106_ = lean_ctor_get(v___x_1043_, 1);
lean_dec(v_unused_1106_);
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1105_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_toApplicative_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1105_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v_toFunctor_1048_; lean_object* v_toSeq_1049_; lean_object* v_toSeqLeft_1050_; lean_object* v_toSeqRight_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1103_; 
v_toFunctor_1048_ = lean_ctor_get(v_toApplicative_1044_, 0);
v_toSeq_1049_ = lean_ctor_get(v_toApplicative_1044_, 2);
v_toSeqLeft_1050_ = lean_ctor_get(v_toApplicative_1044_, 3);
v_toSeqRight_1051_ = lean_ctor_get(v_toApplicative_1044_, 4);
v_isSharedCheck_1103_ = !lean_is_exclusive(v_toApplicative_1044_);
if (v_isSharedCheck_1103_ == 0)
{
lean_object* v_unused_1104_; 
v_unused_1104_ = lean_ctor_get(v_toApplicative_1044_, 1);
lean_dec(v_unused_1104_);
v___x_1053_ = v_toApplicative_1044_;
v_isShared_1054_ = v_isSharedCheck_1103_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_toSeqRight_1051_);
lean_inc(v_toSeqLeft_1050_);
lean_inc(v_toSeq_1049_);
lean_inc(v_toFunctor_1048_);
lean_dec(v_toApplicative_1044_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1103_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___f_1055_; lean_object* v___f_1056_; lean_object* v___f_1057_; lean_object* v___f_1058_; lean_object* v___x_1059_; lean_object* v___f_1060_; lean_object* v___f_1061_; lean_object* v___f_1062_; lean_object* v___x_1064_; 
v___f_1055_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__5));
v___f_1056_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__6));
lean_inc_ref(v_toFunctor_1048_);
v___f_1057_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1057_, 0, v_toFunctor_1048_);
v___f_1058_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1058_, 0, v_toFunctor_1048_);
v___x_1059_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1059_, 0, v___f_1057_);
lean_ctor_set(v___x_1059_, 1, v___f_1058_);
v___f_1060_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1060_, 0, v_toSeqRight_1051_);
v___f_1061_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1061_, 0, v_toSeqLeft_1050_);
v___f_1062_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1062_, 0, v_toSeq_1049_);
if (v_isShared_1054_ == 0)
{
lean_ctor_set(v___x_1053_, 4, v___f_1060_);
lean_ctor_set(v___x_1053_, 3, v___f_1061_);
lean_ctor_set(v___x_1053_, 2, v___f_1062_);
lean_ctor_set(v___x_1053_, 1, v___f_1055_);
lean_ctor_set(v___x_1053_, 0, v___x_1059_);
v___x_1064_ = v___x_1053_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v___x_1059_);
lean_ctor_set(v_reuseFailAlloc_1102_, 1, v___f_1055_);
lean_ctor_set(v_reuseFailAlloc_1102_, 2, v___f_1062_);
lean_ctor_set(v_reuseFailAlloc_1102_, 3, v___f_1061_);
lean_ctor_set(v_reuseFailAlloc_1102_, 4, v___f_1060_);
v___x_1064_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
lean_object* v___x_1066_; 
if (v_isShared_1047_ == 0)
{
lean_ctor_set(v___x_1046_, 1, v___f_1056_);
lean_ctor_set(v___x_1046_, 0, v___x_1064_);
v___x_1066_ = v___x_1046_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1101_, 1, v___f_1056_);
v___x_1066_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
lean_object* v___x_1067_; lean_object* v_toApplicative_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1099_; 
v___x_1067_ = l_StateRefT_x27_instMonad___redArg(v___x_1066_);
v_toApplicative_1068_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1099_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1099_ == 0)
{
lean_object* v_unused_1100_; 
v_unused_1100_ = lean_ctor_get(v___x_1067_, 1);
lean_dec(v_unused_1100_);
v___x_1070_ = v___x_1067_;
v_isShared_1071_ = v_isSharedCheck_1099_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_toApplicative_1068_);
lean_dec(v___x_1067_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1099_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v_toFunctor_1072_; lean_object* v_toSeq_1073_; lean_object* v_toSeqLeft_1074_; lean_object* v_toSeqRight_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1097_; 
v_toFunctor_1072_ = lean_ctor_get(v_toApplicative_1068_, 0);
v_toSeq_1073_ = lean_ctor_get(v_toApplicative_1068_, 2);
v_toSeqLeft_1074_ = lean_ctor_get(v_toApplicative_1068_, 3);
v_toSeqRight_1075_ = lean_ctor_get(v_toApplicative_1068_, 4);
v_isSharedCheck_1097_ = !lean_is_exclusive(v_toApplicative_1068_);
if (v_isSharedCheck_1097_ == 0)
{
lean_object* v_unused_1098_; 
v_unused_1098_ = lean_ctor_get(v_toApplicative_1068_, 1);
lean_dec(v_unused_1098_);
v___x_1077_ = v_toApplicative_1068_;
v_isShared_1078_ = v_isSharedCheck_1097_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_toSeqRight_1075_);
lean_inc(v_toSeqLeft_1074_);
lean_inc(v_toSeq_1073_);
lean_inc(v_toFunctor_1072_);
lean_dec(v_toApplicative_1068_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1097_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v___f_1079_; lean_object* v___f_1080_; lean_object* v___f_1081_; lean_object* v___f_1082_; lean_object* v___x_1083_; lean_object* v___f_1084_; lean_object* v___f_1085_; lean_object* v___f_1086_; lean_object* v___x_1088_; 
v___f_1079_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__7));
v___f_1080_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___closed__8));
lean_inc_ref(v_toFunctor_1072_);
v___f_1081_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1081_, 0, v_toFunctor_1072_);
v___f_1082_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1082_, 0, v_toFunctor_1072_);
v___x_1083_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1083_, 0, v___f_1081_);
lean_ctor_set(v___x_1083_, 1, v___f_1082_);
v___f_1084_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1084_, 0, v_toSeqRight_1075_);
v___f_1085_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1085_, 0, v_toSeqLeft_1074_);
v___f_1086_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1086_, 0, v_toSeq_1073_);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 4, v___f_1084_);
lean_ctor_set(v___x_1077_, 3, v___f_1085_);
lean_ctor_set(v___x_1077_, 2, v___f_1086_);
lean_ctor_set(v___x_1077_, 1, v___f_1079_);
lean_ctor_set(v___x_1077_, 0, v___x_1083_);
v___x_1088_ = v___x_1077_;
goto v_reusejp_1087_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v___x_1083_);
lean_ctor_set(v_reuseFailAlloc_1096_, 1, v___f_1079_);
lean_ctor_set(v_reuseFailAlloc_1096_, 2, v___f_1086_);
lean_ctor_set(v_reuseFailAlloc_1096_, 3, v___f_1085_);
lean_ctor_set(v_reuseFailAlloc_1096_, 4, v___f_1084_);
v___x_1088_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1087_;
}
v_reusejp_1087_:
{
lean_object* v___x_1090_; 
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 1, v___f_1080_);
lean_ctor_set(v___x_1070_, 0, v___x_1088_);
v___x_1090_ = v___x_1070_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v___x_1088_);
lean_ctor_set(v_reuseFailAlloc_1095_, 1, v___f_1080_);
v___x_1090_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_5732__overap_1093_; lean_object* v___x_1094_; 
v___x_1091_ = l_Lean_instInhabitedLocalContext_default;
v___x_1092_ = l_instInhabitedOfMonad___redArg(v___x_1090_, v___x_1091_);
v___x_5732__overap_1093_ = lean_panic_fn_borrowed(v___x_1092_, v_msg_984_);
lean_dec(v___x_1092_);
lean_inc(v___y_992_);
lean_inc_ref(v___y_991_);
lean_inc(v___y_990_);
lean_inc_ref(v___y_989_);
lean_inc(v___y_988_);
lean_inc_ref(v___y_987_);
lean_inc(v___y_986_);
lean_inc_ref(v___y_985_);
v___x_1094_ = lean_apply_9(v___x_5732__overap_1093_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_, v___y_991_, v___y_992_, lean_box(0));
return v___x_1094_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5___boxed(lean_object* v_msg_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_){
_start:
{
lean_object* v_res_1129_; 
v_res_1129_ = lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5(v_msg_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_, v___y_1127_);
lean_dec(v___y_1127_);
lean_dec_ref(v___y_1126_);
lean_dec(v___y_1125_);
lean_dec_ref(v___y_1124_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
return v_res_1129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(lean_object* v_e_1130_, lean_object* v___y_1131_){
_start:
{
uint8_t v___x_1133_; 
v___x_1133_ = l_Lean_Expr_hasMVar(v_e_1130_);
if (v___x_1133_ == 0)
{
lean_object* v___x_1134_; 
v___x_1134_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1134_, 0, v_e_1130_);
return v___x_1134_;
}
else
{
lean_object* v___x_1135_; lean_object* v_mctx_1136_; lean_object* v___x_1137_; lean_object* v_fst_1138_; lean_object* v_snd_1139_; lean_object* v___x_1140_; lean_object* v_cache_1141_; lean_object* v_zetaDeltaFVarIds_1142_; lean_object* v_postponed_1143_; lean_object* v_diag_1144_; lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1153_; 
v___x_1135_ = lean_st_ref_get(v___y_1131_);
v_mctx_1136_ = lean_ctor_get(v___x_1135_, 0);
lean_inc_ref(v_mctx_1136_);
lean_dec(v___x_1135_);
v___x_1137_ = l_Lean_instantiateMVarsCore(v_mctx_1136_, v_e_1130_);
v_fst_1138_ = lean_ctor_get(v___x_1137_, 0);
lean_inc(v_fst_1138_);
v_snd_1139_ = lean_ctor_get(v___x_1137_, 1);
lean_inc(v_snd_1139_);
lean_dec_ref(v___x_1137_);
v___x_1140_ = lean_st_ref_take(v___y_1131_);
v_cache_1141_ = lean_ctor_get(v___x_1140_, 1);
v_zetaDeltaFVarIds_1142_ = lean_ctor_get(v___x_1140_, 2);
v_postponed_1143_ = lean_ctor_get(v___x_1140_, 3);
v_diag_1144_ = lean_ctor_get(v___x_1140_, 4);
v_isSharedCheck_1153_ = !lean_is_exclusive(v___x_1140_);
if (v_isSharedCheck_1153_ == 0)
{
lean_object* v_unused_1154_; 
v_unused_1154_ = lean_ctor_get(v___x_1140_, 0);
lean_dec(v_unused_1154_);
v___x_1146_ = v___x_1140_;
v_isShared_1147_ = v_isSharedCheck_1153_;
goto v_resetjp_1145_;
}
else
{
lean_inc(v_diag_1144_);
lean_inc(v_postponed_1143_);
lean_inc(v_zetaDeltaFVarIds_1142_);
lean_inc(v_cache_1141_);
lean_dec(v___x_1140_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1153_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v___x_1149_; 
if (v_isShared_1147_ == 0)
{
lean_ctor_set(v___x_1146_, 0, v_snd_1139_);
v___x_1149_ = v___x_1146_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v_snd_1139_);
lean_ctor_set(v_reuseFailAlloc_1152_, 1, v_cache_1141_);
lean_ctor_set(v_reuseFailAlloc_1152_, 2, v_zetaDeltaFVarIds_1142_);
lean_ctor_set(v_reuseFailAlloc_1152_, 3, v_postponed_1143_);
lean_ctor_set(v_reuseFailAlloc_1152_, 4, v_diag_1144_);
v___x_1149_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = lean_st_ref_set(v___y_1131_, v___x_1149_);
v___x_1151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1151_, 0, v_fst_1138_);
return v___x_1151_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg___boxed(lean_object* v_e_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_){
_start:
{
lean_object* v_res_1158_; 
v_res_1158_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_e_1155_, v___y_1156_);
lean_dec(v___y_1156_);
return v_res_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(lean_object* v_auxDeclToFullName_1163_, lean_object* v_as_1164_, size_t v_i_1165_, size_t v_stop_1166_, lean_object* v_b_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_){
_start:
{
lean_object* v_a_1178_; uint8_t v___x_1182_; 
v___x_1182_ = lean_usize_dec_eq(v_i_1165_, v_stop_1166_);
if (v___x_1182_ == 0)
{
lean_object* v___x_1183_; 
v___x_1183_ = lean_array_uget_borrowed(v_as_1164_, v_i_1165_);
if (lean_obj_tag(v___x_1183_) == 0)
{
v_a_1178_ = v_b_1167_;
goto v___jp_1177_;
}
else
{
lean_object* v_val_1184_; 
v_val_1184_ = lean_ctor_get(v___x_1183_, 0);
if (lean_obj_tag(v_val_1184_) == 0)
{
uint8_t v_kind_1185_; 
v_kind_1185_ = lean_ctor_get_uint8(v_val_1184_, sizeof(void*)*4 + 1);
if (v_kind_1185_ == 2)
{
lean_object* v_fvarId_1186_; lean_object* v_userName_1187_; lean_object* v_type_1188_; lean_object* v___x_1189_; 
v_fvarId_1186_ = lean_ctor_get(v_val_1184_, 1);
v_userName_1187_ = lean_ctor_get(v_val_1184_, 2);
v_type_1188_ = lean_ctor_get(v_val_1184_, 3);
lean_inc_ref(v_type_1188_);
v___x_1189_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_type_1188_, v___y_1173_);
if (lean_obj_tag(v___x_1189_) == 0)
{
lean_object* v_a_1190_; lean_object* v___x_1191_; 
v_a_1190_ = lean_ctor_get(v___x_1189_, 0);
lean_inc(v_a_1190_);
lean_dec_ref_known(v___x_1189_, 1);
v___x_1191_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg(v_auxDeclToFullName_1163_, v_fvarId_1186_);
if (lean_obj_tag(v___x_1191_) == 1)
{
lean_object* v_val_1192_; lean_object* v___x_1193_; 
v_val_1192_ = lean_ctor_get(v___x_1191_, 0);
lean_inc(v_val_1192_);
lean_dec_ref_known(v___x_1191_, 1);
lean_inc(v_userName_1187_);
lean_inc(v_fvarId_1186_);
v___x_1193_ = l_Lean_LocalContext_mkAuxDecl(v_b_1167_, v_fvarId_1186_, v_userName_1187_, v_a_1190_, v_val_1192_);
v_a_1178_ = v___x_1193_;
goto v___jp_1177_;
}
else
{
lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; uint8_t v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
lean_dec(v___x_1191_);
lean_dec(v_a_1190_);
lean_dec_ref(v_b_1167_);
v___x_1194_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__0));
v___x_1195_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__1));
v___x_1196_ = lean_unsigned_to_nat(635u);
v___x_1197_ = lean_unsigned_to_nat(12u);
v___x_1198_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__2));
v___x_1199_ = 1;
lean_inc(v_userName_1187_);
v___x_1200_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_userName_1187_, v___x_1199_);
v___x_1201_ = lean_string_append(v___x_1198_, v___x_1200_);
lean_dec_ref(v___x_1200_);
v___x_1202_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___closed__3));
v___x_1203_ = lean_string_append(v___x_1201_, v___x_1202_);
v___x_1204_ = l_mkPanicMessageWithDecl(v___x_1194_, v___x_1195_, v___x_1196_, v___x_1197_, v___x_1203_);
lean_dec_ref(v___x_1203_);
v___x_1205_ = lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__5(v___x_1204_, v___y_1168_, v___y_1169_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_);
if (lean_obj_tag(v___x_1205_) == 0)
{
lean_object* v_a_1206_; 
v_a_1206_ = lean_ctor_get(v___x_1205_, 0);
lean_inc(v_a_1206_);
lean_dec_ref_known(v___x_1205_, 1);
v_a_1178_ = v_a_1206_;
goto v___jp_1177_;
}
else
{
return v___x_1205_;
}
}
}
else
{
lean_object* v_a_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1214_; 
lean_dec_ref(v_b_1167_);
v_a_1207_ = lean_ctor_get(v___x_1189_, 0);
v_isSharedCheck_1214_ = !lean_is_exclusive(v___x_1189_);
if (v_isSharedCheck_1214_ == 0)
{
v___x_1209_ = v___x_1189_;
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_a_1207_);
lean_dec(v___x_1189_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1212_; 
if (v_isShared_1210_ == 0)
{
v___x_1212_ = v___x_1209_;
goto v_reusejp_1211_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v_a_1207_);
v___x_1212_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1211_;
}
v_reusejp_1211_:
{
return v___x_1212_;
}
}
}
}
else
{
lean_object* v_fvarId_1215_; lean_object* v_userName_1216_; lean_object* v_type_1217_; uint8_t v_bi_1218_; lean_object* v___x_1219_; 
v_fvarId_1215_ = lean_ctor_get(v_val_1184_, 1);
v_userName_1216_ = lean_ctor_get(v_val_1184_, 2);
v_type_1217_ = lean_ctor_get(v_val_1184_, 3);
v_bi_1218_ = lean_ctor_get_uint8(v_val_1184_, sizeof(void*)*4);
lean_inc_ref(v_type_1217_);
v___x_1219_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_type_1217_, v___y_1173_);
if (lean_obj_tag(v___x_1219_) == 0)
{
lean_object* v_a_1220_; lean_object* v___x_1221_; 
v_a_1220_ = lean_ctor_get(v___x_1219_, 0);
lean_inc(v_a_1220_);
lean_dec_ref_known(v___x_1219_, 1);
lean_inc(v_userName_1216_);
lean_inc(v_fvarId_1215_);
v___x_1221_ = l_Lean_LocalContext_mkLocalDecl(v_b_1167_, v_fvarId_1215_, v_userName_1216_, v_a_1220_, v_bi_1218_, v_kind_1185_);
v_a_1178_ = v___x_1221_;
goto v___jp_1177_;
}
else
{
lean_object* v_a_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1229_; 
lean_dec_ref(v_b_1167_);
v_a_1222_ = lean_ctor_get(v___x_1219_, 0);
v_isSharedCheck_1229_ = !lean_is_exclusive(v___x_1219_);
if (v_isSharedCheck_1229_ == 0)
{
v___x_1224_ = v___x_1219_;
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_a_1222_);
lean_dec(v___x_1219_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
lean_object* v___x_1227_; 
if (v_isShared_1225_ == 0)
{
v___x_1227_ = v___x_1224_;
goto v_reusejp_1226_;
}
else
{
lean_object* v_reuseFailAlloc_1228_; 
v_reuseFailAlloc_1228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1228_, 0, v_a_1222_);
v___x_1227_ = v_reuseFailAlloc_1228_;
goto v_reusejp_1226_;
}
v_reusejp_1226_:
{
return v___x_1227_;
}
}
}
}
}
else
{
lean_object* v_fvarId_1230_; lean_object* v_userName_1231_; lean_object* v_type_1232_; lean_object* v_value_1233_; uint8_t v_nondep_1234_; uint8_t v_kind_1235_; lean_object* v___x_1236_; 
v_fvarId_1230_ = lean_ctor_get(v_val_1184_, 1);
v_userName_1231_ = lean_ctor_get(v_val_1184_, 2);
v_type_1232_ = lean_ctor_get(v_val_1184_, 3);
v_value_1233_ = lean_ctor_get(v_val_1184_, 4);
v_nondep_1234_ = lean_ctor_get_uint8(v_val_1184_, sizeof(void*)*5);
v_kind_1235_ = lean_ctor_get_uint8(v_val_1184_, sizeof(void*)*5 + 1);
lean_inc_ref(v_type_1232_);
v___x_1236_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_type_1232_, v___y_1173_);
if (lean_obj_tag(v___x_1236_) == 0)
{
lean_object* v_a_1237_; lean_object* v___x_1238_; 
v_a_1237_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1237_);
lean_dec_ref_known(v___x_1236_, 1);
lean_inc_ref(v_value_1233_);
v___x_1238_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_value_1233_, v___y_1173_);
if (lean_obj_tag(v___x_1238_) == 0)
{
lean_object* v_a_1239_; lean_object* v___x_1240_; 
v_a_1239_ = lean_ctor_get(v___x_1238_, 0);
lean_inc(v_a_1239_);
lean_dec_ref_known(v___x_1238_, 1);
lean_inc(v_userName_1231_);
lean_inc(v_fvarId_1230_);
v___x_1240_ = l_Lean_LocalContext_mkLetDecl(v_b_1167_, v_fvarId_1230_, v_userName_1231_, v_a_1237_, v_a_1239_, v_nondep_1234_, v_kind_1235_);
v_a_1178_ = v___x_1240_;
goto v___jp_1177_;
}
else
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
lean_dec(v_a_1237_);
lean_dec_ref(v_b_1167_);
v_a_1241_ = lean_ctor_get(v___x_1238_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1238_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1238_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v___x_1238_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_a_1241_);
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
lean_dec_ref(v_b_1167_);
v_a_1249_ = lean_ctor_get(v___x_1236_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v___x_1236_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v___x_1236_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v___x_1236_);
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
lean_object* v___x_1257_; 
v___x_1257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1257_, 0, v_b_1167_);
return v___x_1257_;
}
v___jp_1177_:
{
size_t v___x_1179_; size_t v___x_1180_; 
v___x_1179_ = ((size_t)1ULL);
v___x_1180_ = lean_usize_add(v_i_1165_, v___x_1179_);
v_i_1165_ = v___x_1180_;
v_b_1167_ = v_a_1178_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10___boxed(lean_object* v_auxDeclToFullName_1258_, lean_object* v_as_1259_, lean_object* v_i_1260_, lean_object* v_stop_1261_, lean_object* v_b_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_){
_start:
{
size_t v_i_boxed_1272_; size_t v_stop_boxed_1273_; lean_object* v_res_1274_; 
v_i_boxed_1272_ = lean_unbox_usize(v_i_1260_);
lean_dec(v_i_1260_);
v_stop_boxed_1273_ = lean_unbox_usize(v_stop_1261_);
lean_dec(v_stop_1261_);
v_res_1274_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1258_, v_as_1259_, v_i_boxed_1272_, v_stop_boxed_1273_, v_b_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
lean_dec(v___y_1268_);
lean_dec_ref(v___y_1267_);
lean_dec(v___y_1266_);
lean_dec_ref(v___y_1265_);
lean_dec(v___y_1264_);
lean_dec_ref(v___y_1263_);
lean_dec_ref(v_as_1259_);
lean_dec(v_auxDeclToFullName_1258_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11(lean_object* v_auxDeclToFullName_1275_, lean_object* v_x_1276_, lean_object* v_x_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_){
_start:
{
if (lean_obj_tag(v_x_1276_) == 0)
{
lean_object* v_cs_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1307_; 
v_cs_1287_ = lean_ctor_get(v_x_1276_, 0);
v_isSharedCheck_1307_ = !lean_is_exclusive(v_x_1276_);
if (v_isSharedCheck_1307_ == 0)
{
v___x_1289_ = v_x_1276_;
v_isShared_1290_ = v_isSharedCheck_1307_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_cs_1287_);
lean_dec(v_x_1276_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1307_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
lean_object* v___x_1291_; lean_object* v___x_1292_; uint8_t v___x_1293_; 
v___x_1291_ = lean_unsigned_to_nat(0u);
v___x_1292_ = lean_array_get_size(v_cs_1287_);
v___x_1293_ = lean_nat_dec_lt(v___x_1291_, v___x_1292_);
if (v___x_1293_ == 0)
{
lean_object* v___x_1295_; 
lean_dec_ref(v_cs_1287_);
if (v_isShared_1290_ == 0)
{
lean_ctor_set(v___x_1289_, 0, v_x_1277_);
v___x_1295_ = v___x_1289_;
goto v_reusejp_1294_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v_x_1277_);
v___x_1295_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1294_;
}
v_reusejp_1294_:
{
return v___x_1295_;
}
}
else
{
uint8_t v___x_1297_; 
v___x_1297_ = lean_nat_dec_le(v___x_1292_, v___x_1292_);
if (v___x_1297_ == 0)
{
if (v___x_1293_ == 0)
{
lean_object* v___x_1299_; 
lean_dec_ref(v_cs_1287_);
if (v_isShared_1290_ == 0)
{
lean_ctor_set(v___x_1289_, 0, v_x_1277_);
v___x_1299_ = v___x_1289_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_x_1277_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
else
{
size_t v___x_1301_; size_t v___x_1302_; lean_object* v___x_1303_; 
lean_del_object(v___x_1289_);
v___x_1301_ = ((size_t)0ULL);
v___x_1302_ = lean_usize_of_nat(v___x_1292_);
v___x_1303_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(v_auxDeclToFullName_1275_, v_cs_1287_, v___x_1301_, v___x_1302_, v_x_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec_ref(v_cs_1287_);
return v___x_1303_;
}
}
else
{
size_t v___x_1304_; size_t v___x_1305_; lean_object* v___x_1306_; 
lean_del_object(v___x_1289_);
v___x_1304_ = ((size_t)0ULL);
v___x_1305_ = lean_usize_of_nat(v___x_1292_);
v___x_1306_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(v_auxDeclToFullName_1275_, v_cs_1287_, v___x_1304_, v___x_1305_, v_x_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec_ref(v_cs_1287_);
return v___x_1306_;
}
}
}
}
else
{
lean_object* v_vs_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1328_; 
v_vs_1308_ = lean_ctor_get(v_x_1276_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v_x_1276_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1310_ = v_x_1276_;
v_isShared_1311_ = v_isSharedCheck_1328_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_vs_1308_);
lean_dec(v_x_1276_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1328_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v___x_1312_; lean_object* v___x_1313_; uint8_t v___x_1314_; 
v___x_1312_ = lean_unsigned_to_nat(0u);
v___x_1313_ = lean_array_get_size(v_vs_1308_);
v___x_1314_ = lean_nat_dec_lt(v___x_1312_, v___x_1313_);
if (v___x_1314_ == 0)
{
lean_object* v___x_1316_; 
lean_dec_ref(v_vs_1308_);
if (v_isShared_1311_ == 0)
{
lean_ctor_set_tag(v___x_1310_, 0);
lean_ctor_set(v___x_1310_, 0, v_x_1277_);
v___x_1316_ = v___x_1310_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_x_1277_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
else
{
uint8_t v___x_1318_; 
v___x_1318_ = lean_nat_dec_le(v___x_1313_, v___x_1313_);
if (v___x_1318_ == 0)
{
if (v___x_1314_ == 0)
{
lean_object* v___x_1320_; 
lean_dec_ref(v_vs_1308_);
if (v_isShared_1311_ == 0)
{
lean_ctor_set_tag(v___x_1310_, 0);
lean_ctor_set(v___x_1310_, 0, v_x_1277_);
v___x_1320_ = v___x_1310_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1321_; 
v_reuseFailAlloc_1321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1321_, 0, v_x_1277_);
v___x_1320_ = v_reuseFailAlloc_1321_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
return v___x_1320_;
}
}
else
{
size_t v___x_1322_; size_t v___x_1323_; lean_object* v___x_1324_; 
lean_del_object(v___x_1310_);
v___x_1322_ = ((size_t)0ULL);
v___x_1323_ = lean_usize_of_nat(v___x_1313_);
v___x_1324_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1275_, v_vs_1308_, v___x_1322_, v___x_1323_, v_x_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec_ref(v_vs_1308_);
return v___x_1324_;
}
}
else
{
size_t v___x_1325_; size_t v___x_1326_; lean_object* v___x_1327_; 
lean_del_object(v___x_1310_);
v___x_1325_ = ((size_t)0ULL);
v___x_1326_ = lean_usize_of_nat(v___x_1313_);
v___x_1327_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1275_, v_vs_1308_, v___x_1325_, v___x_1326_, v_x_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec_ref(v_vs_1308_);
return v___x_1327_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(lean_object* v_auxDeclToFullName_1329_, lean_object* v_as_1330_, size_t v_i_1331_, size_t v_stop_1332_, lean_object* v_b_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
uint8_t v___x_1343_; 
v___x_1343_ = lean_usize_dec_eq(v_i_1331_, v_stop_1332_);
if (v___x_1343_ == 0)
{
lean_object* v___x_1344_; lean_object* v___x_1345_; 
v___x_1344_ = lean_array_uget_borrowed(v_as_1330_, v_i_1331_);
lean_inc(v___x_1344_);
v___x_1345_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11(v_auxDeclToFullName_1329_, v___x_1344_, v_b_1333_, v___y_1334_, v___y_1335_, v___y_1336_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_);
if (lean_obj_tag(v___x_1345_) == 0)
{
lean_object* v_a_1346_; size_t v___x_1347_; size_t v___x_1348_; 
v_a_1346_ = lean_ctor_get(v___x_1345_, 0);
lean_inc(v_a_1346_);
lean_dec_ref_known(v___x_1345_, 1);
v___x_1347_ = ((size_t)1ULL);
v___x_1348_ = lean_usize_add(v_i_1331_, v___x_1347_);
v_i_1331_ = v___x_1348_;
v_b_1333_ = v_a_1346_;
goto _start;
}
else
{
return v___x_1345_;
}
}
else
{
lean_object* v___x_1350_; 
v___x_1350_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1350_, 0, v_b_1333_);
return v___x_1350_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10___boxed(lean_object* v_auxDeclToFullName_1351_, lean_object* v_as_1352_, lean_object* v_i_1353_, lean_object* v_stop_1354_, lean_object* v_b_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_){
_start:
{
size_t v_i_boxed_1365_; size_t v_stop_boxed_1366_; lean_object* v_res_1367_; 
v_i_boxed_1365_ = lean_unbox_usize(v_i_1353_);
lean_dec(v_i_1353_);
v_stop_boxed_1366_ = lean_unbox_usize(v_stop_1354_);
lean_dec(v_stop_1354_);
v_res_1367_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(v_auxDeclToFullName_1351_, v_as_1352_, v_i_boxed_1365_, v_stop_boxed_1366_, v_b_1355_, v___y_1356_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
lean_dec(v___y_1361_);
lean_dec_ref(v___y_1360_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
lean_dec(v___y_1357_);
lean_dec_ref(v___y_1356_);
lean_dec_ref(v_as_1352_);
lean_dec(v_auxDeclToFullName_1351_);
return v_res_1367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11___boxed(lean_object* v_auxDeclToFullName_1368_, lean_object* v_x_1369_, lean_object* v_x_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11(v_auxDeclToFullName_1368_, v_x_1369_, v_x_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_);
lean_dec(v___y_1378_);
lean_dec_ref(v___y_1377_);
lean_dec(v___y_1376_);
lean_dec_ref(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec_ref(v___y_1373_);
lean_dec(v___y_1372_);
lean_dec_ref(v___y_1371_);
lean_dec(v_auxDeclToFullName_1368_);
return v_res_1380_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0(void){
_start:
{
lean_object* v___x_1381_; 
v___x_1381_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9(lean_object* v_auxDeclToFullName_1382_, lean_object* v_x_1383_, size_t v_x_1384_, size_t v_x_1385_, lean_object* v_x_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
if (lean_obj_tag(v_x_1383_) == 0)
{
lean_object* v_cs_1396_; lean_object* v___x_1397_; size_t v___x_1398_; lean_object* v_j_1399_; lean_object* v___x_1400_; size_t v___x_1401_; size_t v___x_1402_; size_t v___x_1403_; size_t v___x_1404_; size_t v___x_1405_; size_t v___x_1406_; lean_object* v___x_1407_; 
v_cs_1396_ = lean_ctor_get(v_x_1383_, 0);
lean_inc_ref(v_cs_1396_);
lean_dec_ref_known(v_x_1383_, 1);
v___x_1397_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___closed__0);
v___x_1398_ = lean_usize_shift_right(v_x_1384_, v_x_1385_);
v_j_1399_ = lean_usize_to_nat(v___x_1398_);
v___x_1400_ = lean_array_get_borrowed(v___x_1397_, v_cs_1396_, v_j_1399_);
v___x_1401_ = ((size_t)1ULL);
v___x_1402_ = lean_usize_shift_left(v___x_1401_, v_x_1385_);
v___x_1403_ = lean_usize_sub(v___x_1402_, v___x_1401_);
v___x_1404_ = lean_usize_land(v_x_1384_, v___x_1403_);
v___x_1405_ = ((size_t)5ULL);
v___x_1406_ = lean_usize_sub(v_x_1385_, v___x_1405_);
lean_inc(v___x_1400_);
v___x_1407_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9(v_auxDeclToFullName_1382_, v___x_1400_, v___x_1404_, v___x_1406_, v_x_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1407_) == 0)
{
lean_object* v_a_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; uint8_t v___x_1412_; 
v_a_1408_ = lean_ctor_get(v___x_1407_, 0);
lean_inc(v_a_1408_);
v___x_1409_ = lean_unsigned_to_nat(1u);
v___x_1410_ = lean_nat_add(v_j_1399_, v___x_1409_);
lean_dec(v_j_1399_);
v___x_1411_ = lean_array_get_size(v_cs_1396_);
v___x_1412_ = lean_nat_dec_lt(v___x_1410_, v___x_1411_);
if (v___x_1412_ == 0)
{
lean_dec(v___x_1410_);
lean_dec(v_a_1408_);
lean_dec_ref(v_cs_1396_);
return v___x_1407_;
}
else
{
uint8_t v___x_1413_; 
v___x_1413_ = lean_nat_dec_le(v___x_1411_, v___x_1411_);
if (v___x_1413_ == 0)
{
if (v___x_1412_ == 0)
{
lean_dec(v___x_1410_);
lean_dec(v_a_1408_);
lean_dec_ref(v_cs_1396_);
return v___x_1407_;
}
else
{
size_t v___x_1414_; size_t v___x_1415_; lean_object* v___x_1416_; 
lean_dec_ref_known(v___x_1407_, 1);
v___x_1414_ = lean_usize_of_nat(v___x_1410_);
lean_dec(v___x_1410_);
v___x_1415_ = lean_usize_of_nat(v___x_1411_);
v___x_1416_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(v_auxDeclToFullName_1382_, v_cs_1396_, v___x_1414_, v___x_1415_, v_a_1408_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec_ref(v_cs_1396_);
return v___x_1416_;
}
}
else
{
size_t v___x_1417_; size_t v___x_1418_; lean_object* v___x_1419_; 
lean_dec_ref_known(v___x_1407_, 1);
v___x_1417_ = lean_usize_of_nat(v___x_1410_);
lean_dec(v___x_1410_);
v___x_1418_ = lean_usize_of_nat(v___x_1411_);
v___x_1419_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9_spec__10(v_auxDeclToFullName_1382_, v_cs_1396_, v___x_1417_, v___x_1418_, v_a_1408_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec_ref(v_cs_1396_);
return v___x_1419_;
}
}
}
else
{
lean_dec(v_j_1399_);
lean_dec_ref(v_cs_1396_);
return v___x_1407_;
}
}
else
{
lean_object* v_vs_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1440_; 
v_vs_1420_ = lean_ctor_get(v_x_1383_, 0);
v_isSharedCheck_1440_ = !lean_is_exclusive(v_x_1383_);
if (v_isSharedCheck_1440_ == 0)
{
v___x_1422_ = v_x_1383_;
v_isShared_1423_ = v_isSharedCheck_1440_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_vs_1420_);
lean_dec(v_x_1383_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1440_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1424_; lean_object* v___x_1425_; uint8_t v___x_1426_; 
v___x_1424_ = lean_usize_to_nat(v_x_1384_);
v___x_1425_ = lean_array_get_size(v_vs_1420_);
v___x_1426_ = lean_nat_dec_lt(v___x_1424_, v___x_1425_);
if (v___x_1426_ == 0)
{
lean_object* v___x_1428_; 
lean_dec(v___x_1424_);
lean_dec_ref(v_vs_1420_);
if (v_isShared_1423_ == 0)
{
lean_ctor_set_tag(v___x_1422_, 0);
lean_ctor_set(v___x_1422_, 0, v_x_1386_);
v___x_1428_ = v___x_1422_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v_x_1386_);
v___x_1428_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
return v___x_1428_;
}
}
else
{
uint8_t v___x_1430_; 
v___x_1430_ = lean_nat_dec_le(v___x_1425_, v___x_1425_);
if (v___x_1430_ == 0)
{
if (v___x_1426_ == 0)
{
lean_object* v___x_1432_; 
lean_dec(v___x_1424_);
lean_dec_ref(v_vs_1420_);
if (v_isShared_1423_ == 0)
{
lean_ctor_set_tag(v___x_1422_, 0);
lean_ctor_set(v___x_1422_, 0, v_x_1386_);
v___x_1432_ = v___x_1422_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1433_; 
v_reuseFailAlloc_1433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1433_, 0, v_x_1386_);
v___x_1432_ = v_reuseFailAlloc_1433_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
return v___x_1432_;
}
}
else
{
size_t v___x_1434_; size_t v___x_1435_; lean_object* v___x_1436_; 
lean_del_object(v___x_1422_);
v___x_1434_ = lean_usize_of_nat(v___x_1424_);
lean_dec(v___x_1424_);
v___x_1435_ = lean_usize_of_nat(v___x_1425_);
v___x_1436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1382_, v_vs_1420_, v___x_1434_, v___x_1435_, v_x_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec_ref(v_vs_1420_);
return v___x_1436_;
}
}
else
{
size_t v___x_1437_; size_t v___x_1438_; lean_object* v___x_1439_; 
lean_del_object(v___x_1422_);
v___x_1437_ = lean_usize_of_nat(v___x_1424_);
lean_dec(v___x_1424_);
v___x_1438_ = lean_usize_of_nat(v___x_1425_);
v___x_1439_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1382_, v_vs_1420_, v___x_1437_, v___x_1438_, v_x_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec_ref(v_vs_1420_);
return v___x_1439_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9___boxed(lean_object* v_auxDeclToFullName_1441_, lean_object* v_x_1442_, lean_object* v_x_1443_, lean_object* v_x_1444_, lean_object* v_x_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_){
_start:
{
size_t v_x_9486__boxed_1455_; size_t v_x_9487__boxed_1456_; lean_object* v_res_1457_; 
v_x_9486__boxed_1455_ = lean_unbox_usize(v_x_1443_);
lean_dec(v_x_1443_);
v_x_9487__boxed_1456_ = lean_unbox_usize(v_x_1444_);
lean_dec(v_x_1444_);
v_res_1457_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9(v_auxDeclToFullName_1441_, v_x_1442_, v_x_9486__boxed_1455_, v_x_9487__boxed_1456_, v_x_1445_, v___y_1446_, v___y_1447_, v___y_1448_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
lean_dec(v___y_1451_);
lean_dec_ref(v___y_1450_);
lean_dec(v___y_1449_);
lean_dec_ref(v___y_1448_);
lean_dec(v___y_1447_);
lean_dec_ref(v___y_1446_);
lean_dec(v_auxDeclToFullName_1441_);
return v_res_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8(lean_object* v_auxDeclToFullName_1458_, lean_object* v_t_1459_, lean_object* v_init_1460_, lean_object* v_start_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_){
_start:
{
lean_object* v___x_1471_; uint8_t v___x_1472_; 
v___x_1471_ = lean_unsigned_to_nat(0u);
v___x_1472_ = lean_nat_dec_eq(v_start_1461_, v___x_1471_);
if (v___x_1472_ == 0)
{
lean_object* v_root_1473_; lean_object* v_tail_1474_; size_t v_shift_1475_; lean_object* v_tailOff_1476_; uint8_t v___x_1477_; 
v_root_1473_ = lean_ctor_get(v_t_1459_, 0);
lean_inc_ref(v_root_1473_);
v_tail_1474_ = lean_ctor_get(v_t_1459_, 1);
lean_inc_ref(v_tail_1474_);
v_shift_1475_ = lean_ctor_get_usize(v_t_1459_, 4);
v_tailOff_1476_ = lean_ctor_get(v_t_1459_, 3);
lean_inc(v_tailOff_1476_);
lean_dec_ref(v_t_1459_);
v___x_1477_ = lean_nat_dec_le(v_tailOff_1476_, v_start_1461_);
if (v___x_1477_ == 0)
{
size_t v___x_1478_; lean_object* v___x_1479_; 
lean_dec(v_tailOff_1476_);
v___x_1478_ = lean_usize_of_nat(v_start_1461_);
v___x_1479_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__9(v_auxDeclToFullName_1458_, v_root_1473_, v___x_1478_, v_shift_1475_, v_init_1460_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
if (lean_obj_tag(v___x_1479_) == 0)
{
lean_object* v_a_1480_; lean_object* v___x_1481_; uint8_t v___x_1482_; 
v_a_1480_ = lean_ctor_get(v___x_1479_, 0);
lean_inc(v_a_1480_);
v___x_1481_ = lean_array_get_size(v_tail_1474_);
v___x_1482_ = lean_nat_dec_lt(v___x_1471_, v___x_1481_);
if (v___x_1482_ == 0)
{
lean_dec(v_a_1480_);
lean_dec_ref(v_tail_1474_);
return v___x_1479_;
}
else
{
uint8_t v___x_1483_; 
v___x_1483_ = lean_nat_dec_le(v___x_1481_, v___x_1481_);
if (v___x_1483_ == 0)
{
if (v___x_1482_ == 0)
{
lean_dec(v_a_1480_);
lean_dec_ref(v_tail_1474_);
return v___x_1479_;
}
else
{
size_t v___x_1484_; size_t v___x_1485_; lean_object* v___x_1486_; 
lean_dec_ref_known(v___x_1479_, 1);
v___x_1484_ = ((size_t)0ULL);
v___x_1485_ = lean_usize_of_nat(v___x_1481_);
v___x_1486_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1474_, v___x_1484_, v___x_1485_, v_a_1480_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1474_);
return v___x_1486_;
}
}
else
{
size_t v___x_1487_; size_t v___x_1488_; lean_object* v___x_1489_; 
lean_dec_ref_known(v___x_1479_, 1);
v___x_1487_ = ((size_t)0ULL);
v___x_1488_ = lean_usize_of_nat(v___x_1481_);
v___x_1489_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1474_, v___x_1487_, v___x_1488_, v_a_1480_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1474_);
return v___x_1489_;
}
}
}
else
{
lean_dec_ref(v_tail_1474_);
return v___x_1479_;
}
}
else
{
lean_object* v___x_1490_; lean_object* v___x_1491_; uint8_t v___x_1492_; 
lean_dec_ref(v_root_1473_);
v___x_1490_ = lean_nat_sub(v_start_1461_, v_tailOff_1476_);
lean_dec(v_tailOff_1476_);
v___x_1491_ = lean_array_get_size(v_tail_1474_);
v___x_1492_ = lean_nat_dec_lt(v___x_1490_, v___x_1491_);
if (v___x_1492_ == 0)
{
lean_object* v___x_1493_; 
lean_dec(v___x_1490_);
lean_dec_ref(v_tail_1474_);
v___x_1493_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1493_, 0, v_init_1460_);
return v___x_1493_;
}
else
{
uint8_t v___x_1494_; 
v___x_1494_ = lean_nat_dec_le(v___x_1491_, v___x_1491_);
if (v___x_1494_ == 0)
{
if (v___x_1492_ == 0)
{
lean_object* v___x_1495_; 
lean_dec(v___x_1490_);
lean_dec_ref(v_tail_1474_);
v___x_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1495_, 0, v_init_1460_);
return v___x_1495_;
}
else
{
size_t v___x_1496_; size_t v___x_1497_; lean_object* v___x_1498_; 
v___x_1496_ = lean_usize_of_nat(v___x_1490_);
lean_dec(v___x_1490_);
v___x_1497_ = lean_usize_of_nat(v___x_1491_);
v___x_1498_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1474_, v___x_1496_, v___x_1497_, v_init_1460_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1474_);
return v___x_1498_;
}
}
else
{
size_t v___x_1499_; size_t v___x_1500_; lean_object* v___x_1501_; 
v___x_1499_ = lean_usize_of_nat(v___x_1490_);
lean_dec(v___x_1490_);
v___x_1500_ = lean_usize_of_nat(v___x_1491_);
v___x_1501_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1474_, v___x_1499_, v___x_1500_, v_init_1460_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1474_);
return v___x_1501_;
}
}
}
}
else
{
lean_object* v_root_1502_; lean_object* v_tail_1503_; lean_object* v___x_1504_; 
v_root_1502_ = lean_ctor_get(v_t_1459_, 0);
lean_inc_ref(v_root_1502_);
v_tail_1503_ = lean_ctor_get(v_t_1459_, 1);
lean_inc_ref(v_tail_1503_);
lean_dec_ref(v_t_1459_);
v___x_1504_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__11(v_auxDeclToFullName_1458_, v_root_1502_, v_init_1460_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
if (lean_obj_tag(v___x_1504_) == 0)
{
lean_object* v_a_1505_; lean_object* v___x_1506_; uint8_t v___x_1507_; 
v_a_1505_ = lean_ctor_get(v___x_1504_, 0);
lean_inc(v_a_1505_);
v___x_1506_ = lean_array_get_size(v_tail_1503_);
v___x_1507_ = lean_nat_dec_lt(v___x_1471_, v___x_1506_);
if (v___x_1507_ == 0)
{
lean_dec(v_a_1505_);
lean_dec_ref(v_tail_1503_);
return v___x_1504_;
}
else
{
uint8_t v___x_1508_; 
v___x_1508_ = lean_nat_dec_le(v___x_1506_, v___x_1506_);
if (v___x_1508_ == 0)
{
if (v___x_1507_ == 0)
{
lean_dec(v_a_1505_);
lean_dec_ref(v_tail_1503_);
return v___x_1504_;
}
else
{
size_t v___x_1509_; size_t v___x_1510_; lean_object* v___x_1511_; 
lean_dec_ref_known(v___x_1504_, 1);
v___x_1509_ = ((size_t)0ULL);
v___x_1510_ = lean_usize_of_nat(v___x_1506_);
v___x_1511_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1503_, v___x_1509_, v___x_1510_, v_a_1505_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1503_);
return v___x_1511_;
}
}
else
{
size_t v___x_1512_; size_t v___x_1513_; lean_object* v___x_1514_; 
lean_dec_ref_known(v___x_1504_, 1);
v___x_1512_ = ((size_t)0ULL);
v___x_1513_ = lean_usize_of_nat(v___x_1506_);
v___x_1514_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8_spec__10(v_auxDeclToFullName_1458_, v_tail_1503_, v___x_1512_, v___x_1513_, v_a_1505_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec_ref(v_tail_1503_);
return v___x_1514_;
}
}
}
else
{
lean_dec_ref(v_tail_1503_);
return v___x_1504_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8___boxed(lean_object* v_auxDeclToFullName_1515_, lean_object* v_t_1516_, lean_object* v_init_1517_, lean_object* v_start_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_){
_start:
{
lean_object* v_res_1528_; 
v_res_1528_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8(v_auxDeclToFullName_1515_, v_t_1516_, v_init_1517_, v_start_1518_, v___y_1519_, v___y_1520_, v___y_1521_, v___y_1522_, v___y_1523_, v___y_1524_, v___y_1525_, v___y_1526_);
lean_dec(v___y_1526_);
lean_dec_ref(v___y_1525_);
lean_dec(v___y_1524_);
lean_dec_ref(v___y_1523_);
lean_dec(v___y_1522_);
lean_dec_ref(v___y_1521_);
lean_dec(v___y_1520_);
lean_dec_ref(v___y_1519_);
lean_dec(v_start_1518_);
lean_dec(v_auxDeclToFullName_1515_);
return v_res_1528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6(lean_object* v_auxDeclToFullName_1529_, lean_object* v_lctx_1530_, lean_object* v_init_1531_, lean_object* v_start_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_){
_start:
{
lean_object* v_decls_1542_; lean_object* v___x_1543_; 
v_decls_1542_ = lean_ctor_get(v_lctx_1530_, 1);
lean_inc_ref(v_decls_1542_);
lean_dec_ref(v_lctx_1530_);
v___x_1543_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6_spec__8(v_auxDeclToFullName_1529_, v_decls_1542_, v_init_1531_, v_start_1532_, v___y_1533_, v___y_1534_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_, v___y_1539_, v___y_1540_);
return v___x_1543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6___boxed(lean_object* v_auxDeclToFullName_1544_, lean_object* v_lctx_1545_, lean_object* v_init_1546_, lean_object* v_start_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v_res_1557_; 
v_res_1557_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6(v_auxDeclToFullName_1544_, v_lctx_1545_, v_init_1546_, v_start_1547_, v___y_1548_, v___y_1549_, v___y_1550_, v___y_1551_, v___y_1552_, v___y_1553_, v___y_1554_, v___y_1555_);
lean_dec(v___y_1555_);
lean_dec_ref(v___y_1554_);
lean_dec(v___y_1553_);
lean_dec_ref(v___y_1552_);
lean_dec(v___y_1551_);
lean_dec_ref(v___y_1550_);
lean_dec(v___y_1549_);
lean_dec_ref(v___y_1548_);
lean_dec(v_start_1547_);
lean_dec(v_auxDeclToFullName_1544_);
return v_res_1557_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1558_; 
v___x_1558_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1558_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1(void){
_start:
{
lean_object* v___x_1559_; lean_object* v___x_1560_; 
v___x_1559_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__0);
v___x_1560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1560_, 0, v___x_1559_);
return v___x_1560_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2(void){
_start:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
v___x_1561_ = lean_unsigned_to_nat(32u);
v___x_1562_ = lean_mk_empty_array_with_capacity(v___x_1561_);
v___x_1563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1563_, 0, v___x_1562_);
return v___x_1563_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3(void){
_start:
{
size_t v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; 
v___x_1564_ = ((size_t)5ULL);
v___x_1565_ = lean_unsigned_to_nat(0u);
v___x_1566_ = lean_unsigned_to_nat(32u);
v___x_1567_ = lean_mk_empty_array_with_capacity(v___x_1566_);
v___x_1568_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__2);
v___x_1569_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1569_, 0, v___x_1568_);
lean_ctor_set(v___x_1569_, 1, v___x_1567_);
lean_ctor_set(v___x_1569_, 2, v___x_1565_);
lean_ctor_set(v___x_1569_, 3, v___x_1565_);
lean_ctor_set_usize(v___x_1569_, 4, v___x_1564_);
return v___x_1569_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4(void){
_start:
{
lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; 
v___x_1570_ = lean_box(1);
v___x_1571_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__3);
v___x_1572_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__1);
v___x_1573_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1572_);
lean_ctor_set(v___x_1573_, 1, v___x_1571_);
lean_ctor_set(v___x_1573_, 2, v___x_1570_);
return v___x_1573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3(lean_object* v_lctx_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_){
_start:
{
lean_object* v_auxDeclToFullName_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; 
v_auxDeclToFullName_1584_ = lean_ctor_get(v_lctx_1574_, 2);
lean_inc(v_auxDeclToFullName_1584_);
v___x_1585_ = lean_unsigned_to_nat(0u);
v___x_1586_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___closed__4);
v___x_1587_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__6(v_auxDeclToFullName_1584_, v_lctx_1574_, v___x_1586_, v___x_1585_, v___y_1575_, v___y_1576_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_, v___y_1582_);
lean_dec(v_auxDeclToFullName_1584_);
return v___x_1587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3___boxed(lean_object* v_lctx_1588_, lean_object* v___y_1589_, lean_object* v___y_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_){
_start:
{
lean_object* v_res_1598_; 
v_res_1598_ = lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3(v_lctx_1588_, v___y_1589_, v___y_1590_, v___y_1591_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_, v___y_1596_);
lean_dec(v___y_1596_);
lean_dec_ref(v___y_1595_);
lean_dec(v___y_1594_);
lean_dec_ref(v___y_1593_);
lean_dec(v___y_1592_);
lean_dec_ref(v___y_1591_);
lean_dec(v___y_1590_);
lean_dec_ref(v___y_1589_);
return v_res_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2(lean_object* v_mvarId_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_){
_start:
{
lean_object* v___x_1609_; lean_object* v_mctx_1610_; lean_object* v_mvarDecl_1611_; lean_object* v_userName_1612_; lean_object* v_lctx_1613_; lean_object* v_type_1614_; lean_object* v_depth_1615_; lean_object* v_localInstances_1616_; uint8_t v_kind_1617_; lean_object* v_numScopeArgs_1618_; lean_object* v_index_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1682_; 
v___x_1609_ = lean_st_ref_get(v___y_1605_);
v_mctx_1610_ = lean_ctor_get(v___x_1609_, 0);
lean_inc_ref(v_mctx_1610_);
lean_dec(v___x_1609_);
lean_inc(v_mvarId_1599_);
v_mvarDecl_1611_ = l_Lean_MetavarContext_getDecl(v_mctx_1610_, v_mvarId_1599_);
lean_dec_ref(v_mctx_1610_);
v_userName_1612_ = lean_ctor_get(v_mvarDecl_1611_, 0);
v_lctx_1613_ = lean_ctor_get(v_mvarDecl_1611_, 1);
v_type_1614_ = lean_ctor_get(v_mvarDecl_1611_, 2);
v_depth_1615_ = lean_ctor_get(v_mvarDecl_1611_, 3);
v_localInstances_1616_ = lean_ctor_get(v_mvarDecl_1611_, 4);
v_kind_1617_ = lean_ctor_get_uint8(v_mvarDecl_1611_, sizeof(void*)*7);
v_numScopeArgs_1618_ = lean_ctor_get(v_mvarDecl_1611_, 5);
v_index_1619_ = lean_ctor_get(v_mvarDecl_1611_, 6);
v_isSharedCheck_1682_ = !lean_is_exclusive(v_mvarDecl_1611_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1621_ = v_mvarDecl_1611_;
v_isShared_1622_ = v_isSharedCheck_1682_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_index_1619_);
lean_inc(v_numScopeArgs_1618_);
lean_inc(v_localInstances_1616_);
lean_inc(v_depth_1615_);
lean_inc(v_type_1614_);
lean_inc(v_lctx_1613_);
lean_inc(v_userName_1612_);
lean_dec(v_mvarDecl_1611_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1682_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1623_; 
v___x_1623_ = lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3(v_lctx_1613_, v___y_1600_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_, v___y_1605_, v___y_1606_, v___y_1607_);
if (lean_obj_tag(v___x_1623_) == 0)
{
lean_object* v_a_1624_; lean_object* v___x_1625_; lean_object* v_a_1626_; lean_object* v___x_1628_; uint8_t v_isShared_1629_; uint8_t v_isSharedCheck_1673_; 
v_a_1624_ = lean_ctor_get(v___x_1623_, 0);
lean_inc(v_a_1624_);
lean_dec_ref_known(v___x_1623_, 1);
v___x_1625_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_type_1614_, v___y_1605_);
v_a_1626_ = lean_ctor_get(v___x_1625_, 0);
v_isSharedCheck_1673_ = !lean_is_exclusive(v___x_1625_);
if (v_isSharedCheck_1673_ == 0)
{
v___x_1628_ = v___x_1625_;
v_isShared_1629_ = v_isSharedCheck_1673_;
goto v_resetjp_1627_;
}
else
{
lean_inc(v_a_1626_);
lean_dec(v___x_1625_);
v___x_1628_ = lean_box(0);
v_isShared_1629_ = v_isSharedCheck_1673_;
goto v_resetjp_1627_;
}
v_resetjp_1627_:
{
lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v_fst_1632_; lean_object* v_snd_1633_; lean_object* v___x_1634_; lean_object* v_mctx_1635_; lean_object* v_cache_1636_; lean_object* v_zetaDeltaFVarIds_1637_; lean_object* v_postponed_1638_; lean_object* v_diag_1639_; lean_object* v___x_1641_; uint8_t v_isShared_1642_; uint8_t v_isSharedCheck_1672_; 
v___x_1630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1630_, 0, v_a_1624_);
lean_ctor_set(v___x_1630_, 1, v_a_1626_);
v___x_1631_ = lean_sharecommon_quick(v___x_1630_);
lean_dec_ref_known(v___x_1630_, 2);
v_fst_1632_ = lean_ctor_get(v___x_1631_, 0);
lean_inc(v_fst_1632_);
v_snd_1633_ = lean_ctor_get(v___x_1631_, 1);
lean_inc(v_snd_1633_);
lean_dec(v___x_1631_);
v___x_1634_ = lean_st_ref_take(v___y_1605_);
v_mctx_1635_ = lean_ctor_get(v___x_1634_, 0);
v_cache_1636_ = lean_ctor_get(v___x_1634_, 1);
v_zetaDeltaFVarIds_1637_ = lean_ctor_get(v___x_1634_, 2);
v_postponed_1638_ = lean_ctor_get(v___x_1634_, 3);
v_diag_1639_ = lean_ctor_get(v___x_1634_, 4);
v_isSharedCheck_1672_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1672_ == 0)
{
v___x_1641_ = v___x_1634_;
v_isShared_1642_ = v_isSharedCheck_1672_;
goto v_resetjp_1640_;
}
else
{
lean_inc(v_diag_1639_);
lean_inc(v_postponed_1638_);
lean_inc(v_zetaDeltaFVarIds_1637_);
lean_inc(v_cache_1636_);
lean_inc(v_mctx_1635_);
lean_dec(v___x_1634_);
v___x_1641_ = lean_box(0);
v_isShared_1642_ = v_isSharedCheck_1672_;
goto v_resetjp_1640_;
}
v_resetjp_1640_:
{
lean_object* v_depth_1643_; lean_object* v_levelAssignDepth_1644_; lean_object* v_lmvarCounter_1645_; lean_object* v_mvarCounter_1646_; lean_object* v_lDecls_1647_; lean_object* v_decls_1648_; lean_object* v_userNames_1649_; lean_object* v_lAssignment_1650_; lean_object* v_eAssignment_1651_; lean_object* v_dAssignment_1652_; lean_object* v___x_1654_; uint8_t v_isShared_1655_; uint8_t v_isSharedCheck_1671_; 
v_depth_1643_ = lean_ctor_get(v_mctx_1635_, 0);
v_levelAssignDepth_1644_ = lean_ctor_get(v_mctx_1635_, 1);
v_lmvarCounter_1645_ = lean_ctor_get(v_mctx_1635_, 2);
v_mvarCounter_1646_ = lean_ctor_get(v_mctx_1635_, 3);
v_lDecls_1647_ = lean_ctor_get(v_mctx_1635_, 4);
v_decls_1648_ = lean_ctor_get(v_mctx_1635_, 5);
v_userNames_1649_ = lean_ctor_get(v_mctx_1635_, 6);
v_lAssignment_1650_ = lean_ctor_get(v_mctx_1635_, 7);
v_eAssignment_1651_ = lean_ctor_get(v_mctx_1635_, 8);
v_dAssignment_1652_ = lean_ctor_get(v_mctx_1635_, 9);
v_isSharedCheck_1671_ = !lean_is_exclusive(v_mctx_1635_);
if (v_isSharedCheck_1671_ == 0)
{
v___x_1654_ = v_mctx_1635_;
v_isShared_1655_ = v_isSharedCheck_1671_;
goto v_resetjp_1653_;
}
else
{
lean_inc(v_dAssignment_1652_);
lean_inc(v_eAssignment_1651_);
lean_inc(v_lAssignment_1650_);
lean_inc(v_userNames_1649_);
lean_inc(v_decls_1648_);
lean_inc(v_lDecls_1647_);
lean_inc(v_mvarCounter_1646_);
lean_inc(v_lmvarCounter_1645_);
lean_inc(v_levelAssignDepth_1644_);
lean_inc(v_depth_1643_);
lean_dec(v_mctx_1635_);
v___x_1654_ = lean_box(0);
v_isShared_1655_ = v_isSharedCheck_1671_;
goto v_resetjp_1653_;
}
v_resetjp_1653_:
{
lean_object* v___x_1657_; 
if (v_isShared_1622_ == 0)
{
lean_ctor_set(v___x_1621_, 2, v_snd_1633_);
lean_ctor_set(v___x_1621_, 1, v_fst_1632_);
v___x_1657_ = v___x_1621_;
goto v_reusejp_1656_;
}
else
{
lean_object* v_reuseFailAlloc_1670_; 
v_reuseFailAlloc_1670_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_1670_, 0, v_userName_1612_);
lean_ctor_set(v_reuseFailAlloc_1670_, 1, v_fst_1632_);
lean_ctor_set(v_reuseFailAlloc_1670_, 2, v_snd_1633_);
lean_ctor_set(v_reuseFailAlloc_1670_, 3, v_depth_1615_);
lean_ctor_set(v_reuseFailAlloc_1670_, 4, v_localInstances_1616_);
lean_ctor_set(v_reuseFailAlloc_1670_, 5, v_numScopeArgs_1618_);
lean_ctor_set(v_reuseFailAlloc_1670_, 6, v_index_1619_);
lean_ctor_set_uint8(v_reuseFailAlloc_1670_, sizeof(void*)*7, v_kind_1617_);
v___x_1657_ = v_reuseFailAlloc_1670_;
goto v_reusejp_1656_;
}
v_reusejp_1656_:
{
lean_object* v___x_1658_; lean_object* v___x_1660_; 
v___x_1658_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic_modifyMetavarDecl___at___00Mathlib_Tactic_modifyLocalContext___at___00Mathlib_Tactic_modifyLocalDecl___at___00Mathlib_Tactic_renameBVarHyp_spec__0_spec__1_spec__3_spec__7___redArg(v_decls_1648_, v_mvarId_1599_, v___x_1657_);
if (v_isShared_1655_ == 0)
{
lean_ctor_set(v___x_1654_, 5, v___x_1658_);
v___x_1660_ = v___x_1654_;
goto v_reusejp_1659_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v_depth_1643_);
lean_ctor_set(v_reuseFailAlloc_1669_, 1, v_levelAssignDepth_1644_);
lean_ctor_set(v_reuseFailAlloc_1669_, 2, v_lmvarCounter_1645_);
lean_ctor_set(v_reuseFailAlloc_1669_, 3, v_mvarCounter_1646_);
lean_ctor_set(v_reuseFailAlloc_1669_, 4, v_lDecls_1647_);
lean_ctor_set(v_reuseFailAlloc_1669_, 5, v___x_1658_);
lean_ctor_set(v_reuseFailAlloc_1669_, 6, v_userNames_1649_);
lean_ctor_set(v_reuseFailAlloc_1669_, 7, v_lAssignment_1650_);
lean_ctor_set(v_reuseFailAlloc_1669_, 8, v_eAssignment_1651_);
lean_ctor_set(v_reuseFailAlloc_1669_, 9, v_dAssignment_1652_);
v___x_1660_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1659_;
}
v_reusejp_1659_:
{
lean_object* v___x_1662_; 
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v___x_1660_);
v___x_1662_ = v___x_1641_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1668_; 
v_reuseFailAlloc_1668_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1668_, 0, v___x_1660_);
lean_ctor_set(v_reuseFailAlloc_1668_, 1, v_cache_1636_);
lean_ctor_set(v_reuseFailAlloc_1668_, 2, v_zetaDeltaFVarIds_1637_);
lean_ctor_set(v_reuseFailAlloc_1668_, 3, v_postponed_1638_);
lean_ctor_set(v_reuseFailAlloc_1668_, 4, v_diag_1639_);
v___x_1662_ = v_reuseFailAlloc_1668_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1666_; 
v___x_1663_ = lean_st_ref_set(v___y_1605_, v___x_1662_);
v___x_1664_ = lean_box(0);
if (v_isShared_1629_ == 0)
{
lean_ctor_set(v___x_1628_, 0, v___x_1664_);
v___x_1666_ = v___x_1628_;
goto v_reusejp_1665_;
}
else
{
lean_object* v_reuseFailAlloc_1667_; 
v_reuseFailAlloc_1667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1667_, 0, v___x_1664_);
v___x_1666_ = v_reuseFailAlloc_1667_;
goto v_reusejp_1665_;
}
v_reusejp_1665_:
{
return v___x_1666_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1681_; 
lean_del_object(v___x_1621_);
lean_dec(v_index_1619_);
lean_dec(v_numScopeArgs_1618_);
lean_dec_ref(v_localInstances_1616_);
lean_dec(v_depth_1615_);
lean_dec_ref(v_type_1614_);
lean_dec(v_userName_1612_);
lean_dec(v_mvarId_1599_);
v_a_1674_ = lean_ctor_get(v___x_1623_, 0);
v_isSharedCheck_1681_ = !lean_is_exclusive(v___x_1623_);
if (v_isSharedCheck_1681_ == 0)
{
v___x_1676_ = v___x_1623_;
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_a_1674_);
lean_dec(v___x_1623_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
lean_object* v___x_1679_; 
if (v_isShared_1677_ == 0)
{
v___x_1679_ = v___x_1676_;
goto v_reusejp_1678_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_a_1674_);
v___x_1679_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1678_;
}
v_reusejp_1678_:
{
return v___x_1679_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2___boxed(lean_object* v_mvarId_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_){
_start:
{
lean_object* v_res_1693_; 
v_res_1693_ = lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2(v_mvarId_1683_, v___y_1684_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_, v___y_1691_);
lean_dec(v___y_1691_);
lean_dec_ref(v___y_1690_);
lean_dec(v___y_1689_);
lean_dec_ref(v___y_1688_);
lean_dec(v___y_1687_);
lean_dec_ref(v___y_1686_);
lean_dec(v___y_1685_);
lean_dec_ref(v___y_1684_);
return v_res_1693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1(lean_object* v_x_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_, lean_object* v_a_1700_, lean_object* v_a_1701_, lean_object* v_a_1702_, lean_object* v_a_1703_){
_start:
{
lean_object* v___x_1705_; uint8_t v___x_1706_; 
v___x_1705_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192_____00__closed__3));
lean_inc(v_x_1695_);
v___x_1706_ = l_Lean_Syntax_isOfKind(v_x_1695_, v___x_1705_);
if (v___x_1706_ == 0)
{
lean_object* v___x_1707_; 
lean_dec(v_x_1695_);
v___x_1707_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__0___redArg();
return v___x_1707_;
}
else
{
lean_object* v___f_1708_; lean_object* v___x_1709_; lean_object* v_old_1710_; lean_object* v___x_1711_; lean_object* v_new_1712_; lean_object* v___y_1714_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
v___f_1708_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___closed__0));
v___x_1709_ = lean_unsigned_to_nat(1u);
v_old_1710_ = l_Lean_Syntax_getArg(v_x_1695_, v___x_1709_);
v___x_1711_ = lean_unsigned_to_nat(3u);
v_new_1712_ = l_Lean_Syntax_getArg(v_x_1695_, v___x_1711_);
v___x_1736_ = lean_unsigned_to_nat(4u);
v___x_1737_ = l_Lean_Syntax_getArg(v_x_1695_, v___x_1736_);
lean_dec(v_x_1695_);
v___x_1738_ = l_Lean_Syntax_getOptional_x3f(v___x_1737_);
lean_dec(v___x_1737_);
if (lean_obj_tag(v___x_1738_) == 0)
{
lean_object* v___x_1739_; 
v___x_1739_ = lean_box(0);
v___y_1714_ = v___x_1739_;
goto v___jp_1713_;
}
else
{
lean_object* v_val_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1747_; 
v_val_1740_ = lean_ctor_get(v___x_1738_, 0);
v_isSharedCheck_1747_ = !lean_is_exclusive(v___x_1738_);
if (v_isSharedCheck_1747_ == 0)
{
v___x_1742_ = v___x_1738_;
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_val_1740_);
lean_dec(v___x_1738_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1745_; 
if (v_isShared_1743_ == 0)
{
v___x_1745_ = v___x_1742_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v_val_1740_);
v___x_1745_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
v___y_1714_ = v___x_1745_;
goto v___jp_1713_;
}
}
}
v___jp_1713_:
{
lean_object* v___x_1715_; 
v___x_1715_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1697_, v_a_1700_, v_a_1701_, v_a_1702_, v_a_1703_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___x_1717_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
lean_inc_n(v_a_1716_, 2);
lean_dec_ref_known(v___x_1715_, 1);
v___x_1717_ = lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2(v_a_1716_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_, v_a_1700_, v_a_1701_, v_a_1702_, v_a_1703_);
if (lean_obj_tag(v___x_1717_) == 0)
{
lean_dec_ref_known(v___x_1717_, 1);
if (lean_obj_tag(v___y_1714_) == 0)
{
lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; 
v___x_1718_ = l_Lean_TSyntax_getId(v_old_1710_);
lean_dec(v_old_1710_);
v___x_1719_ = l_Lean_TSyntax_getId(v_new_1712_);
lean_dec(v_new_1712_);
v___x_1720_ = lp_mathlib_Mathlib_Tactic_renameBVarTarget(v_a_1716_, v___x_1718_, v___x_1719_, v_a_1700_, v_a_1701_, v_a_1702_, v_a_1703_);
return v___x_1720_;
}
else
{
lean_object* v_val_1721_; lean_object* v___f_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___f_1726_; lean_object* v___x_1727_; 
v_val_1721_ = lean_ctor_get(v___y_1714_, 0);
lean_inc(v_val_1721_);
lean_dec_ref_known(v___y_1714_, 1);
lean_inc(v_a_1716_);
lean_inc(v_new_1712_);
lean_inc(v_old_1710_);
v___f_1722_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__1___boxed), 13, 3);
lean_closure_set(v___f_1722_, 0, v_old_1710_);
lean_closure_set(v___f_1722_, 1, v_new_1712_);
lean_closure_set(v___f_1722_, 2, v_a_1716_);
v___x_1723_ = l_Lean_Elab_Tactic_expandLocation(v_val_1721_);
lean_dec(v_val_1721_);
v___x_1724_ = l_Lean_TSyntax_getId(v_old_1710_);
lean_dec(v_old_1710_);
v___x_1725_ = l_Lean_TSyntax_getId(v_new_1712_);
lean_dec(v_new_1712_);
v___f_1726_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___lam__2___boxed), 12, 3);
lean_closure_set(v___f_1726_, 0, v_a_1716_);
lean_closure_set(v___f_1726_, 1, v___x_1724_);
lean_closure_set(v___f_1726_, 2, v___x_1725_);
v___x_1727_ = l_Lean_Elab_Tactic_withLocation(v___x_1723_, v___f_1722_, v___f_1726_, v___f_1708_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_, v_a_1700_, v_a_1701_, v_a_1702_, v_a_1703_);
lean_dec(v___x_1723_);
return v___x_1727_;
}
}
else
{
lean_dec(v_a_1716_);
lean_dec(v___y_1714_);
lean_dec(v_new_1712_);
lean_dec(v_old_1710_);
return v___x_1717_;
}
}
else
{
lean_object* v_a_1728_; lean_object* v___x_1730_; uint8_t v_isShared_1731_; uint8_t v_isSharedCheck_1735_; 
lean_dec(v___y_1714_);
lean_dec(v_new_1712_);
lean_dec(v_old_1710_);
v_a_1728_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_1735_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1735_ == 0)
{
v___x_1730_ = v___x_1715_;
v_isShared_1731_ = v_isSharedCheck_1735_;
goto v_resetjp_1729_;
}
else
{
lean_inc(v_a_1728_);
lean_dec(v___x_1715_);
v___x_1730_ = lean_box(0);
v_isShared_1731_ = v_isSharedCheck_1735_;
goto v_resetjp_1729_;
}
v_resetjp_1729_:
{
lean_object* v___x_1733_; 
if (v_isShared_1731_ == 0)
{
v___x_1733_ = v___x_1730_;
goto v_reusejp_1732_;
}
else
{
lean_object* v_reuseFailAlloc_1734_; 
v_reuseFailAlloc_1734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1734_, 0, v_a_1728_);
v___x_1733_ = v_reuseFailAlloc_1734_;
goto v_reusejp_1732_;
}
v_reusejp_1732_:
{
return v___x_1733_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1___boxed(lean_object* v_x_1748_, lean_object* v_a_1749_, lean_object* v_a_1750_, lean_object* v_a_1751_, lean_object* v_a_1752_, lean_object* v_a_1753_, lean_object* v_a_1754_, lean_object* v_a_1755_, lean_object* v_a_1756_, lean_object* v_a_1757_){
_start:
{
lean_object* v_res_1758_; 
v_res_1758_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1(v_x_1748_, v_a_1749_, v_a_1750_, v_a_1751_, v_a_1752_, v_a_1753_, v_a_1754_, v_a_1755_, v_a_1756_);
lean_dec(v_a_1756_);
lean_dec_ref(v_a_1755_);
lean_dec(v_a_1754_);
lean_dec_ref(v_a_1753_);
lean_dec(v_a_1752_);
lean_dec_ref(v_a_1751_);
lean_dec(v_a_1750_);
lean_dec_ref(v_a_1749_);
return v_res_1758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1(lean_object* v_00_u03b1_1759_, lean_object* v_msg_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_){
_start:
{
lean_object* v___x_1770_; 
v___x_1770_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___redArg(v_msg_1760_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
return v___x_1770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1___boxed(lean_object* v_00_u03b1_1771_, lean_object* v_msg_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_){
_start:
{
lean_object* v_res_1782_; 
v_res_1782_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__1(v_00_u03b1_1771_, v_msg_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_, v___y_1780_);
lean_dec(v___y_1780_);
lean_dec_ref(v___y_1779_);
lean_dec(v___y_1778_);
lean_dec_ref(v___y_1777_);
lean_dec(v___y_1776_);
lean_dec_ref(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
return v_res_1782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4(lean_object* v_e_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_){
_start:
{
lean_object* v___x_1793_; 
v___x_1793_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___redArg(v_e_1783_, v___y_1789_);
return v___x_1793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4___boxed(lean_object* v_e_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_){
_start:
{
lean_object* v_res_1804_; 
v_res_1804_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__4(v_e_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_);
lean_dec(v___y_1802_);
lean_dec_ref(v___y_1801_);
lean_dec(v___y_1800_);
lean_dec_ref(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec_ref(v___y_1797_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
return v_res_1804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4(lean_object* v_00_u03b4_1805_, lean_object* v_t_1806_, lean_object* v_k_1807_){
_start:
{
lean_object* v___x_1808_; 
v___x_1808_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___redArg(v_t_1806_, v_k_1807_);
return v___x_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b4_1809_, lean_object* v_t_1810_, lean_object* v_k_1811_){
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic___aux__Mathlib__Tactic__RenameBVar______elabRules__Mathlib__Tactic__tacticRename__bvar___u2192______1_spec__2_spec__3_spec__4(v_00_u03b4_1809_, v_t_1810_, v_k_1811_);
lean_dec(v_k_1811_);
lean_dec(v_t_1810_);
return v_res_1812_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192____ = _init_lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticRename__bvar___u2192____);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
}
#ifdef __cplusplus
}
#endif
