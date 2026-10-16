// Lean compiler output
// Module: Batteries.Lean.Meta.Basic
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Intro import Lean.Meta.SynthInstance
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
lean_object* lean_array_push(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_modifyUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_throwMaxRecDepthAt___redArg(lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* l_Lean_LocalContext_sortFVarsByContextOrder(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableMVarId_hash___boxed(lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* l_Lean_Meta_tactic_hygienic;
lean_object* l_Lean_Option_set___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqMVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableMVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__1_value;
static const lean_string_object lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "unknown metavariable '\?"};
static const lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3;
static const lean_string_object lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__4 = (const lean_object*)&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_declareExprMVar(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_MetavarContext_isExprMVarDeclared(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_isExprMVarDeclared___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Lean_MetavarContext_unassignedExprMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___closed__0 = (const lean_object*)&lp_batteries_Lean_MetavarContext_unassignedExprMVars___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_synthInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_getTypeCleanup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_getTypeCleanup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Meta_unhygienic___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Meta_unhygienic___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Meta_saturate1___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ST_Prim_mkRef___boxed, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_MetavarContext_unassignedExprMVars___closed__0_value)} };
static const lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__4___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_saturate1___redArg___lam__4___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0(lean_object* v_hyps_1_, lean_object* v_toPure_2_, lean_object* v_____do__lift_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = l_Lean_LocalContext_sortFVarsByContextOrder(v_____do__lift_3_, v_hyps_1_);
v___x_5_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0___boxed(lean_object* v_hyps_6_, lean_object* v_toPure_7_, lean_object* v_____do__lift_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0(v_hyps_6_, v_toPure_7_, v_____do__lift_8_);
lean_dec_ref(v_____do__lift_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg(lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_hyps_12_){
_start:
{
lean_object* v_toApplicative_13_; lean_object* v_toBind_14_; lean_object* v_toPure_15_; lean_object* v___f_16_; lean_object* v___x_17_; 
v_toApplicative_13_ = lean_ctor_get(v_inst_10_, 0);
lean_inc_ref(v_toApplicative_13_);
v_toBind_14_ = lean_ctor_get(v_inst_10_, 1);
lean_inc(v_toBind_14_);
lean_dec_ref(v_inst_10_);
v_toPure_15_ = lean_ctor_get(v_toApplicative_13_, 1);
lean_inc(v_toPure_15_);
lean_dec_ref(v_toApplicative_13_);
v___f_16_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_16_, 0, v_hyps_12_);
lean_closure_set(v___f_16_, 1, v_toPure_15_);
v___x_17_ = lean_apply_4(v_toBind_14_, lean_box(0), lean_box(0), v_inst_11_, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_sortFVarsByContextOrder(lean_object* v_m_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_hyps_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_batteries_Lean_Meta_sortFVarsByContextOrder___redArg(v_inst_19_, v_inst_20_, v_hyps_21_);
return v___x_22_;
}
}
static lean_object* _init_lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = ((lean_object*)(lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__2));
v___x_27_ = l_Lean_stringToMessageData(v___x_26_);
return v___x_27_;
}
}
static lean_object* _init_lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = ((lean_object*)(lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__4));
v___x_30_ = l_Lean_stringToMessageData(v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_mctx_33_, lean_object* v_mvarId_34_){
_start:
{
lean_object* v_toApplicative_35_; lean_object* v_toPure_36_; lean_object* v_decls_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v_toApplicative_35_ = lean_ctor_get(v_inst_31_, 0);
v_toPure_36_ = lean_ctor_get(v_toApplicative_35_, 1);
v_decls_37_ = lean_ctor_get(v_mctx_33_, 5);
v___x_38_ = ((lean_object*)(lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__0));
v___x_39_ = ((lean_object*)(lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__1));
lean_inc(v_mvarId_34_);
v___x_40_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_38_, v___x_39_, v_decls_37_, v_mvarId_34_);
if (lean_obj_tag(v___x_40_) == 1)
{
lean_object* v_val_41_; lean_object* v___x_42_; 
lean_inc(v_toPure_36_);
lean_dec(v_mvarId_34_);
lean_dec_ref(v_inst_32_);
lean_dec_ref(v_inst_31_);
v_val_41_ = lean_ctor_get(v___x_40_, 0);
lean_inc(v_val_41_);
lean_dec_ref_known(v___x_40_, 1);
v___x_42_ = lean_apply_2(v_toPure_36_, lean_box(0), v_val_41_);
return v___x_42_;
}
else
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
lean_dec(v___x_40_);
v___x_43_ = lean_obj_once(&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3, &lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3_once, _init_lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__3);
v___x_44_ = l_Lean_MessageData_ofName(v_mvarId_34_);
v___x_45_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_43_);
lean_ctor_set(v___x_45_, 1, v___x_44_);
v___x_46_ = lean_obj_once(&lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5, &lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5_once, _init_lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___closed__5);
v___x_47_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_47_, 0, v___x_45_);
lean_ctor_set(v___x_47_, 1, v___x_46_);
v___x_48_ = l_Lean_throwError___redArg(v_inst_31_, v_inst_32_, v___x_47_);
return v___x_48_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg___boxed(lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_mctx_51_, lean_object* v_mvarId_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(v_inst_49_, v_inst_50_, v_mctx_51_, v_mvarId_52_);
lean_dec_ref(v_mctx_51_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl(lean_object* v_m_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_mctx_57_, lean_object* v_mvarId_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(v_inst_55_, v_inst_56_, v_mctx_57_, v_mvarId_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___boxed(lean_object* v_m_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_mctx_63_, lean_object* v_mvarId_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_batteries_Lean_MetavarContext_getExprMVarDecl(v_m_60_, v_inst_61_, v_inst_62_, v_mctx_63_, v_mvarId_64_);
lean_dec_ref(v_mctx_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_x_66_, lean_object* v_x_67_, lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
lean_object* v_ks_70_; lean_object* v_vs_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_95_; 
v_ks_70_ = lean_ctor_get(v_x_66_, 0);
v_vs_71_ = lean_ctor_get(v_x_66_, 1);
v_isSharedCheck_95_ = !lean_is_exclusive(v_x_66_);
if (v_isSharedCheck_95_ == 0)
{
v___x_73_ = v_x_66_;
v_isShared_74_ = v_isSharedCheck_95_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_vs_71_);
lean_inc(v_ks_70_);
lean_dec(v_x_66_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_95_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_75_ = lean_array_get_size(v_ks_70_);
v___x_76_ = lean_nat_dec_lt(v_x_67_, v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_80_; 
lean_dec(v_x_67_);
v___x_77_ = lean_array_push(v_ks_70_, v_x_68_);
v___x_78_ = lean_array_push(v_vs_71_, v_x_69_);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 1, v___x_78_);
lean_ctor_set(v___x_73_, 0, v___x_77_);
v___x_80_ = v___x_73_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v___x_77_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v___x_78_);
v___x_80_ = v_reuseFailAlloc_81_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
return v___x_80_;
}
}
else
{
lean_object* v_k_x27_82_; uint8_t v___x_83_; 
v_k_x27_82_ = lean_array_fget_borrowed(v_ks_70_, v_x_67_);
v___x_83_ = l_Lean_instBEqMVarId_beq(v_x_68_, v_k_x27_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_85_; 
if (v_isShared_74_ == 0)
{
v___x_85_ = v___x_73_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v_ks_70_);
lean_ctor_set(v_reuseFailAlloc_89_, 1, v_vs_71_);
v___x_85_ = v_reuseFailAlloc_89_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = lean_nat_add(v_x_67_, v___x_86_);
lean_dec(v_x_67_);
v_x_66_ = v___x_85_;
v_x_67_ = v___x_87_;
goto _start;
}
}
else
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_93_; 
v___x_90_ = lean_array_fset(v_ks_70_, v_x_67_, v_x_68_);
v___x_91_ = lean_array_fset(v_vs_71_, v_x_67_, v_x_69_);
lean_dec(v_x_67_);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 1, v___x_91_);
lean_ctor_set(v___x_73_, 0, v___x_90_);
v___x_93_ = v___x_73_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_90_);
lean_ctor_set(v_reuseFailAlloc_94_, 1, v___x_91_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1___redArg(lean_object* v_n_96_, lean_object* v_k_97_, lean_object* v_v_98_){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_unsigned_to_nat(0u);
v___x_100_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2___redArg(v_n_96_, v___x_99_, v_k_97_, v_v_98_);
return v___x_100_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(lean_object* v_x_102_, size_t v_x_103_, size_t v_x_104_, lean_object* v_x_105_, lean_object* v_x_106_){
_start:
{
if (lean_obj_tag(v_x_102_) == 0)
{
lean_object* v_es_107_; size_t v___x_108_; size_t v___x_109_; lean_object* v_j_110_; lean_object* v___x_111_; uint8_t v___x_112_; 
v_es_107_ = lean_ctor_get(v_x_102_, 0);
v___x_108_ = ((size_t)31ULL);
v___x_109_ = lean_usize_land(v_x_103_, v___x_108_);
v_j_110_ = lean_usize_to_nat(v___x_109_);
v___x_111_ = lean_array_get_size(v_es_107_);
v___x_112_ = lean_nat_dec_lt(v_j_110_, v___x_111_);
if (v___x_112_ == 0)
{
lean_dec(v_j_110_);
lean_dec(v_x_106_);
lean_dec(v_x_105_);
return v_x_102_;
}
else
{
lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_151_; 
lean_inc_ref(v_es_107_);
v_isSharedCheck_151_ = !lean_is_exclusive(v_x_102_);
if (v_isSharedCheck_151_ == 0)
{
lean_object* v_unused_152_; 
v_unused_152_ = lean_ctor_get(v_x_102_, 0);
lean_dec(v_unused_152_);
v___x_114_ = v_x_102_;
v_isShared_115_ = v_isSharedCheck_151_;
goto v_resetjp_113_;
}
else
{
lean_dec(v_x_102_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_151_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v_v_116_; lean_object* v___x_117_; lean_object* v_xs_x27_118_; lean_object* v___y_120_; 
v_v_116_ = lean_array_fget(v_es_107_, v_j_110_);
v___x_117_ = lean_box(0);
v_xs_x27_118_ = lean_array_fset(v_es_107_, v_j_110_, v___x_117_);
switch(lean_obj_tag(v_v_116_))
{
case 0:
{
lean_object* v_key_125_; lean_object* v_val_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_136_; 
v_key_125_ = lean_ctor_get(v_v_116_, 0);
v_val_126_ = lean_ctor_get(v_v_116_, 1);
v_isSharedCheck_136_ = !lean_is_exclusive(v_v_116_);
if (v_isSharedCheck_136_ == 0)
{
v___x_128_ = v_v_116_;
v_isShared_129_ = v_isSharedCheck_136_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_val_126_);
lean_inc(v_key_125_);
lean_dec(v_v_116_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_136_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
uint8_t v___x_130_; 
v___x_130_ = l_Lean_instBEqMVarId_beq(v_x_105_, v_key_125_);
if (v___x_130_ == 0)
{
lean_object* v___x_131_; lean_object* v___x_132_; 
lean_del_object(v___x_128_);
v___x_131_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_125_, v_val_126_, v_x_105_, v_x_106_);
v___x_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
v___y_120_ = v___x_132_;
goto v___jp_119_;
}
else
{
lean_object* v___x_134_; 
lean_dec(v_val_126_);
lean_dec(v_key_125_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 1, v_x_106_);
lean_ctor_set(v___x_128_, 0, v_x_105_);
v___x_134_ = v___x_128_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v_x_105_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v_x_106_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
v___y_120_ = v___x_134_;
goto v___jp_119_;
}
}
}
}
case 1:
{
lean_object* v_node_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_149_; 
v_node_137_ = lean_ctor_get(v_v_116_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v_v_116_);
if (v_isSharedCheck_149_ == 0)
{
v___x_139_ = v_v_116_;
v_isShared_140_ = v_isSharedCheck_149_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_node_137_);
lean_dec(v_v_116_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_149_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
size_t v___x_141_; size_t v___x_142_; size_t v___x_143_; size_t v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
v___x_141_ = ((size_t)5ULL);
v___x_142_ = lean_usize_shift_right(v_x_103_, v___x_141_);
v___x_143_ = ((size_t)1ULL);
v___x_144_ = lean_usize_add(v_x_104_, v___x_143_);
v___x_145_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(v_node_137_, v___x_142_, v___x_144_, v_x_105_, v_x_106_);
if (v_isShared_140_ == 0)
{
lean_ctor_set(v___x_139_, 0, v___x_145_);
v___x_147_ = v___x_139_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
v___y_120_ = v___x_147_;
goto v___jp_119_;
}
}
}
default: 
{
lean_object* v___x_150_; 
v___x_150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_150_, 0, v_x_105_);
lean_ctor_set(v___x_150_, 1, v_x_106_);
v___y_120_ = v___x_150_;
goto v___jp_119_;
}
}
v___jp_119_:
{
lean_object* v___x_121_; lean_object* v___x_123_; 
v___x_121_ = lean_array_fset(v_xs_x27_118_, v_j_110_, v___y_120_);
lean_dec(v_j_110_);
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 0, v___x_121_);
v___x_123_ = v___x_114_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_121_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
}
else
{
lean_object* v_ks_153_; lean_object* v_vs_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_174_; 
v_ks_153_ = lean_ctor_get(v_x_102_, 0);
v_vs_154_ = lean_ctor_get(v_x_102_, 1);
v_isSharedCheck_174_ = !lean_is_exclusive(v_x_102_);
if (v_isSharedCheck_174_ == 0)
{
v___x_156_ = v_x_102_;
v_isShared_157_ = v_isSharedCheck_174_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_vs_154_);
lean_inc(v_ks_153_);
lean_dec(v_x_102_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_174_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_159_; 
if (v_isShared_157_ == 0)
{
v___x_159_ = v___x_156_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_173_; 
v_reuseFailAlloc_173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_173_, 0, v_ks_153_);
lean_ctor_set(v_reuseFailAlloc_173_, 1, v_vs_154_);
v___x_159_ = v_reuseFailAlloc_173_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
lean_object* v_newNode_160_; uint8_t v___y_162_; size_t v___x_168_; uint8_t v___x_169_; 
v_newNode_160_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1___redArg(v___x_159_, v_x_105_, v_x_106_);
v___x_168_ = ((size_t)7ULL);
v___x_169_ = lean_usize_dec_le(v___x_168_, v_x_104_);
if (v___x_169_ == 0)
{
lean_object* v___x_170_; lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_170_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_160_);
v___x_171_ = lean_unsigned_to_nat(4u);
v___x_172_ = lean_nat_dec_lt(v___x_170_, v___x_171_);
lean_dec(v___x_170_);
v___y_162_ = v___x_172_;
goto v___jp_161_;
}
else
{
v___y_162_ = v___x_169_;
goto v___jp_161_;
}
v___jp_161_:
{
if (v___y_162_ == 0)
{
lean_object* v_ks_163_; lean_object* v_vs_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v_ks_163_ = lean_ctor_get(v_newNode_160_, 0);
lean_inc_ref(v_ks_163_);
v_vs_164_ = lean_ctor_get(v_newNode_160_, 1);
lean_inc_ref(v_vs_164_);
lean_dec_ref(v_newNode_160_);
v___x_165_ = lean_unsigned_to_nat(0u);
v___x_166_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___closed__0);
v___x_167_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg(v_x_104_, v_ks_163_, v_vs_164_, v___x_165_, v___x_166_);
lean_dec_ref(v_vs_164_);
lean_dec_ref(v_ks_163_);
return v___x_167_;
}
else
{
return v_newNode_160_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg(size_t v_depth_175_, lean_object* v_keys_176_, lean_object* v_vals_177_, lean_object* v_i_178_, lean_object* v_entries_179_){
_start:
{
lean_object* v___x_180_; uint8_t v___x_181_; 
v___x_180_ = lean_array_get_size(v_keys_176_);
v___x_181_ = lean_nat_dec_lt(v_i_178_, v___x_180_);
if (v___x_181_ == 0)
{
lean_dec(v_i_178_);
return v_entries_179_;
}
else
{
lean_object* v_k_182_; lean_object* v_v_183_; uint64_t v___x_184_; size_t v_h_185_; size_t v___x_186_; lean_object* v___x_187_; size_t v___x_188_; size_t v___x_189_; size_t v___x_190_; size_t v_h_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_k_182_ = lean_array_fget_borrowed(v_keys_176_, v_i_178_);
v_v_183_ = lean_array_fget_borrowed(v_vals_177_, v_i_178_);
v___x_184_ = l_Lean_instHashableMVarId_hash(v_k_182_);
v_h_185_ = lean_uint64_to_usize(v___x_184_);
v___x_186_ = ((size_t)5ULL);
v___x_187_ = lean_unsigned_to_nat(1u);
v___x_188_ = ((size_t)1ULL);
v___x_189_ = lean_usize_sub(v_depth_175_, v___x_188_);
v___x_190_ = lean_usize_mul(v___x_186_, v___x_189_);
v_h_191_ = lean_usize_shift_right(v_h_185_, v___x_190_);
v___x_192_ = lean_nat_add(v_i_178_, v___x_187_);
lean_dec(v_i_178_);
lean_inc(v_v_183_);
lean_inc(v_k_182_);
v___x_193_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(v_entries_179_, v_h_191_, v_depth_175_, v_k_182_, v_v_183_);
v_i_178_ = v___x_192_;
v_entries_179_ = v___x_193_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_depth_195_, lean_object* v_keys_196_, lean_object* v_vals_197_, lean_object* v_i_198_, lean_object* v_entries_199_){
_start:
{
size_t v_depth_boxed_200_; lean_object* v_res_201_; 
v_depth_boxed_200_ = lean_unbox_usize(v_depth_195_);
lean_dec(v_depth_195_);
v_res_201_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg(v_depth_boxed_200_, v_keys_196_, v_vals_197_, v_i_198_, v_entries_199_);
lean_dec_ref(v_vals_197_);
lean_dec_ref(v_keys_196_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg___boxed(lean_object* v_x_202_, lean_object* v_x_203_, lean_object* v_x_204_, lean_object* v_x_205_, lean_object* v_x_206_){
_start:
{
size_t v_x_352__boxed_207_; size_t v_x_353__boxed_208_; lean_object* v_res_209_; 
v_x_352__boxed_207_ = lean_unbox_usize(v_x_203_);
lean_dec(v_x_203_);
v_x_353__boxed_208_ = lean_unbox_usize(v_x_204_);
lean_dec(v_x_204_);
v_res_209_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(v_x_202_, v_x_352__boxed_207_, v_x_353__boxed_208_, v_x_205_, v_x_206_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0___redArg(lean_object* v_x_210_, lean_object* v_x_211_, lean_object* v_x_212_){
_start:
{
uint64_t v___x_213_; size_t v___x_214_; size_t v___x_215_; lean_object* v___x_216_; 
v___x_213_ = l_Lean_instHashableMVarId_hash(v_x_211_);
v___x_214_ = lean_uint64_to_usize(v___x_213_);
v___x_215_ = ((size_t)1ULL);
v___x_216_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(v_x_210_, v___x_214_, v___x_215_, v_x_211_, v_x_212_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_declareExprMVar(lean_object* v_mctx_217_, lean_object* v_mvarId_218_, lean_object* v_mdecl_219_){
_start:
{
lean_object* v_depth_220_; lean_object* v_levelAssignDepth_221_; lean_object* v_lmvarCounter_222_; lean_object* v_mvarCounter_223_; lean_object* v_lDecls_224_; lean_object* v_decls_225_; lean_object* v_userNames_226_; lean_object* v_lAssignment_227_; lean_object* v_eAssignment_228_; lean_object* v_dAssignment_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_237_; 
v_depth_220_ = lean_ctor_get(v_mctx_217_, 0);
v_levelAssignDepth_221_ = lean_ctor_get(v_mctx_217_, 1);
v_lmvarCounter_222_ = lean_ctor_get(v_mctx_217_, 2);
v_mvarCounter_223_ = lean_ctor_get(v_mctx_217_, 3);
v_lDecls_224_ = lean_ctor_get(v_mctx_217_, 4);
v_decls_225_ = lean_ctor_get(v_mctx_217_, 5);
v_userNames_226_ = lean_ctor_get(v_mctx_217_, 6);
v_lAssignment_227_ = lean_ctor_get(v_mctx_217_, 7);
v_eAssignment_228_ = lean_ctor_get(v_mctx_217_, 8);
v_dAssignment_229_ = lean_ctor_get(v_mctx_217_, 9);
v_isSharedCheck_237_ = !lean_is_exclusive(v_mctx_217_);
if (v_isSharedCheck_237_ == 0)
{
v___x_231_ = v_mctx_217_;
v_isShared_232_ = v_isSharedCheck_237_;
goto v_resetjp_230_;
}
else
{
lean_inc(v_dAssignment_229_);
lean_inc(v_eAssignment_228_);
lean_inc(v_lAssignment_227_);
lean_inc(v_userNames_226_);
lean_inc(v_decls_225_);
lean_inc(v_lDecls_224_);
lean_inc(v_mvarCounter_223_);
lean_inc(v_lmvarCounter_222_);
lean_inc(v_levelAssignDepth_221_);
lean_inc(v_depth_220_);
lean_dec(v_mctx_217_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_237_;
goto v_resetjp_230_;
}
v_resetjp_230_:
{
lean_object* v___x_233_; lean_object* v___x_235_; 
v___x_233_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0___redArg(v_decls_225_, v_mvarId_218_, v_mdecl_219_);
if (v_isShared_232_ == 0)
{
lean_ctor_set(v___x_231_, 5, v___x_233_);
v___x_235_ = v___x_231_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_depth_220_);
lean_ctor_set(v_reuseFailAlloc_236_, 1, v_levelAssignDepth_221_);
lean_ctor_set(v_reuseFailAlloc_236_, 2, v_lmvarCounter_222_);
lean_ctor_set(v_reuseFailAlloc_236_, 3, v_mvarCounter_223_);
lean_ctor_set(v_reuseFailAlloc_236_, 4, v_lDecls_224_);
lean_ctor_set(v_reuseFailAlloc_236_, 5, v___x_233_);
lean_ctor_set(v_reuseFailAlloc_236_, 6, v_userNames_226_);
lean_ctor_set(v_reuseFailAlloc_236_, 7, v_lAssignment_227_);
lean_ctor_set(v_reuseFailAlloc_236_, 8, v_eAssignment_228_);
lean_ctor_set(v_reuseFailAlloc_236_, 9, v_dAssignment_229_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0(lean_object* v_00_u03b2_238_, lean_object* v_x_239_, lean_object* v_x_240_, lean_object* v_x_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0___redArg(v_x_239_, v_x_240_, v_x_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0(lean_object* v_00_u03b2_243_, lean_object* v_x_244_, size_t v_x_245_, size_t v_x_246_, lean_object* v_x_247_, lean_object* v_x_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___redArg(v_x_244_, v_x_245_, v_x_246_, v_x_247_, v_x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0___boxed(lean_object* v_00_u03b2_250_, lean_object* v_x_251_, lean_object* v_x_252_, lean_object* v_x_253_, lean_object* v_x_254_, lean_object* v_x_255_){
_start:
{
size_t v_x_546__boxed_256_; size_t v_x_547__boxed_257_; lean_object* v_res_258_; 
v_x_546__boxed_256_ = lean_unbox_usize(v_x_252_);
lean_dec(v_x_252_);
v_x_547__boxed_257_ = lean_unbox_usize(v_x_253_);
lean_dec(v_x_253_);
v_res_258_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0(v_00_u03b2_250_, v_x_251_, v_x_546__boxed_256_, v_x_547__boxed_257_, v_x_254_, v_x_255_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_259_, lean_object* v_n_260_, lean_object* v_k_261_, lean_object* v_v_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1___redArg(v_n_260_, v_k_261_, v_v_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_264_, size_t v_depth_265_, lean_object* v_keys_266_, lean_object* v_vals_267_, lean_object* v_heq_268_, lean_object* v_i_269_, lean_object* v_entries_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___redArg(v_depth_265_, v_keys_266_, v_vals_267_, v_i_269_, v_entries_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_272_, lean_object* v_depth_273_, lean_object* v_keys_274_, lean_object* v_vals_275_, lean_object* v_heq_276_, lean_object* v_i_277_, lean_object* v_entries_278_){
_start:
{
size_t v_depth_boxed_279_; lean_object* v_res_280_; 
v_depth_boxed_279_ = lean_unbox_usize(v_depth_273_);
lean_dec(v_depth_273_);
v_res_280_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__2(v_00_u03b2_272_, v_depth_boxed_279_, v_keys_274_, v_vals_275_, v_heq_276_, v_i_277_, v_entries_278_);
lean_dec_ref(v_vals_275_);
lean_dec_ref(v_keys_274_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_281_, lean_object* v_x_282_, lean_object* v_x_283_, lean_object* v_x_284_, lean_object* v_x_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0_spec__0_spec__1_spec__2___redArg(v_x_282_, v_x_283_, v_x_284_, v_x_285_);
return v___x_286_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_287_, lean_object* v_i_288_, lean_object* v_k_289_){
_start:
{
lean_object* v___x_290_; uint8_t v___x_291_; 
v___x_290_ = lean_array_get_size(v_keys_287_);
v___x_291_ = lean_nat_dec_lt(v_i_288_, v___x_290_);
if (v___x_291_ == 0)
{
lean_dec(v_i_288_);
return v___x_291_;
}
else
{
lean_object* v_k_x27_292_; uint8_t v___x_293_; 
v_k_x27_292_ = lean_array_fget_borrowed(v_keys_287_, v_i_288_);
v___x_293_ = l_Lean_instBEqMVarId_beq(v_k_289_, v_k_x27_292_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = lean_unsigned_to_nat(1u);
v___x_295_ = lean_nat_add(v_i_288_, v___x_294_);
lean_dec(v_i_288_);
v_i_288_ = v___x_295_;
goto _start;
}
else
{
lean_dec(v_i_288_);
return v___x_293_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_297_, lean_object* v_i_298_, lean_object* v_k_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg(v_keys_297_, v_i_298_, v_k_299_);
lean_dec(v_k_299_);
lean_dec_ref(v_keys_297_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg(lean_object* v_x_302_, size_t v_x_303_, lean_object* v_x_304_){
_start:
{
if (lean_obj_tag(v_x_302_) == 0)
{
lean_object* v_es_305_; lean_object* v___x_306_; size_t v___x_307_; size_t v___x_308_; lean_object* v_j_309_; lean_object* v___x_310_; 
v_es_305_ = lean_ctor_get(v_x_302_, 0);
v___x_306_ = lean_box(2);
v___x_307_ = ((size_t)31ULL);
v___x_308_ = lean_usize_land(v_x_303_, v___x_307_);
v_j_309_ = lean_usize_to_nat(v___x_308_);
v___x_310_ = lean_array_get_borrowed(v___x_306_, v_es_305_, v_j_309_);
lean_dec(v_j_309_);
switch(lean_obj_tag(v___x_310_))
{
case 0:
{
lean_object* v_key_311_; uint8_t v___x_312_; 
v_key_311_ = lean_ctor_get(v___x_310_, 0);
v___x_312_ = l_Lean_instBEqMVarId_beq(v_x_304_, v_key_311_);
return v___x_312_;
}
case 1:
{
lean_object* v_node_313_; size_t v___x_314_; size_t v___x_315_; 
v_node_313_ = lean_ctor_get(v___x_310_, 0);
v___x_314_ = ((size_t)5ULL);
v___x_315_ = lean_usize_shift_right(v_x_303_, v___x_314_);
v_x_302_ = v_node_313_;
v_x_303_ = v___x_315_;
goto _start;
}
default: 
{
uint8_t v___x_317_; 
v___x_317_ = 0;
return v___x_317_;
}
}
}
else
{
lean_object* v_ks_318_; lean_object* v___x_319_; uint8_t v___x_320_; 
v_ks_318_ = lean_ctor_get(v_x_302_, 0);
v___x_319_ = lean_unsigned_to_nat(0u);
v___x_320_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg(v_ks_318_, v___x_319_, v_x_304_);
return v___x_320_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg___boxed(lean_object* v_x_321_, lean_object* v_x_322_, lean_object* v_x_323_){
_start:
{
size_t v_x_139__boxed_324_; uint8_t v_res_325_; lean_object* v_r_326_; 
v_x_139__boxed_324_ = lean_unbox_usize(v_x_322_);
lean_dec(v_x_322_);
v_res_325_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg(v_x_321_, v_x_139__boxed_324_, v_x_323_);
lean_dec(v_x_323_);
lean_dec_ref(v_x_321_);
v_r_326_ = lean_box(v_res_325_);
return v_r_326_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(lean_object* v_x_327_, lean_object* v_x_328_){
_start:
{
uint64_t v___x_329_; size_t v___x_330_; uint8_t v___x_331_; 
v___x_329_ = l_Lean_instHashableMVarId_hash(v_x_328_);
v___x_330_ = lean_uint64_to_usize(v___x_329_);
v___x_331_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg(v_x_327_, v___x_330_, v_x_328_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg___boxed(lean_object* v_x_332_, lean_object* v_x_333_){
_start:
{
uint8_t v_res_334_; lean_object* v_r_335_; 
v_res_334_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_x_332_, v_x_333_);
lean_dec(v_x_333_);
lean_dec_ref(v_x_332_);
v_r_335_ = lean_box(v_res_334_);
return v_r_335_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(lean_object* v_mctx_336_, lean_object* v_mvarId_337_){
_start:
{
lean_object* v_eAssignment_338_; lean_object* v_dAssignment_339_; uint8_t v___x_340_; 
v_eAssignment_338_ = lean_ctor_get(v_mctx_336_, 8);
v_dAssignment_339_ = lean_ctor_get(v_mctx_336_, 9);
v___x_340_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_eAssignment_338_, v_mvarId_337_);
if (v___x_340_ == 0)
{
uint8_t v___x_341_; 
v___x_341_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_dAssignment_339_, v_mvarId_337_);
return v___x_341_;
}
else
{
return v___x_340_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned___boxed(lean_object* v_mctx_342_, lean_object* v_mvarId_343_){
_start:
{
uint8_t v_res_344_; lean_object* v_r_345_; 
v_res_344_ = lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(v_mctx_342_, v_mvarId_343_);
lean_dec(v_mvarId_343_);
lean_dec_ref(v_mctx_342_);
v_r_345_ = lean_box(v_res_344_);
return v_r_345_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0(lean_object* v_00_u03b2_346_, lean_object* v_x_347_, lean_object* v_x_348_){
_start:
{
uint8_t v___x_349_; 
v___x_349_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_x_347_, v_x_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___boxed(lean_object* v_00_u03b2_350_, lean_object* v_x_351_, lean_object* v_x_352_){
_start:
{
uint8_t v_res_353_; lean_object* v_r_354_; 
v_res_353_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0(v_00_u03b2_350_, v_x_351_, v_x_352_);
lean_dec(v_x_352_);
lean_dec_ref(v_x_351_);
v_r_354_ = lean_box(v_res_353_);
return v_r_354_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0(lean_object* v_00_u03b2_355_, lean_object* v_x_356_, size_t v_x_357_, lean_object* v_x_358_){
_start:
{
uint8_t v___x_359_; 
v___x_359_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___redArg(v_x_356_, v_x_357_, v_x_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0___boxed(lean_object* v_00_u03b2_360_, lean_object* v_x_361_, lean_object* v_x_362_, lean_object* v_x_363_){
_start:
{
size_t v_x_204__boxed_364_; uint8_t v_res_365_; lean_object* v_r_366_; 
v_x_204__boxed_364_ = lean_unbox_usize(v_x_362_);
lean_dec(v_x_362_);
v_res_365_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0(v_00_u03b2_360_, v_x_361_, v_x_204__boxed_364_, v_x_363_);
lean_dec(v_x_363_);
lean_dec_ref(v_x_361_);
v_r_366_ = lean_box(v_res_365_);
return v_r_366_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_367_, lean_object* v_keys_368_, lean_object* v_vals_369_, lean_object* v_heq_370_, lean_object* v_i_371_, lean_object* v_k_372_){
_start:
{
uint8_t v___x_373_; 
v___x_373_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___redArg(v_keys_368_, v_i_371_, v_k_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_374_, lean_object* v_keys_375_, lean_object* v_vals_376_, lean_object* v_heq_377_, lean_object* v_i_378_, lean_object* v_k_379_){
_start:
{
uint8_t v_res_380_; lean_object* v_r_381_; 
v_res_380_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0_spec__0_spec__1(v_00_u03b2_374_, v_keys_375_, v_vals_376_, v_heq_377_, v_i_378_, v_k_379_);
lean_dec(v_k_379_);
lean_dec_ref(v_vals_376_);
lean_dec_ref(v_keys_375_);
v_r_381_ = lean_box(v_res_380_);
return v_r_381_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_MetavarContext_isExprMVarDeclared(lean_object* v_mctx_382_, lean_object* v_mvarId_383_){
_start:
{
lean_object* v_decls_384_; uint8_t v___x_385_; 
v_decls_384_ = lean_ctor_get(v_mctx_382_, 5);
v___x_385_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_decls_384_, v_mvarId_383_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_isExprMVarDeclared___boxed(lean_object* v_mctx_386_, lean_object* v_mvarId_387_){
_start:
{
uint8_t v_res_388_; lean_object* v_r_389_; 
v_res_388_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_mctx_386_, v_mvarId_387_);
lean_dec(v_mvarId_387_);
lean_dec_ref(v_mctx_386_);
v_r_389_ = lean_box(v_res_388_);
return v_r_389_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2(lean_object* v_xs_390_, lean_object* v_v_391_, lean_object* v_i_392_){
_start:
{
lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_393_ = lean_array_get_size(v_xs_390_);
v___x_394_ = lean_nat_dec_lt(v_i_392_, v___x_393_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; 
lean_dec(v_i_392_);
v___x_395_ = lean_box(0);
return v___x_395_;
}
else
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = lean_array_fget_borrowed(v_xs_390_, v_i_392_);
v___x_397_ = l_Lean_instBEqMVarId_beq(v___x_396_, v_v_391_);
if (v___x_397_ == 0)
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = lean_unsigned_to_nat(1u);
v___x_399_ = lean_nat_add(v_i_392_, v___x_398_);
lean_dec(v_i_392_);
v_i_392_ = v___x_399_;
goto _start;
}
else
{
lean_object* v___x_401_; 
v___x_401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_401_, 0, v_i_392_);
return v___x_401_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_xs_402_, lean_object* v_v_403_, lean_object* v_i_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2(v_xs_402_, v_v_403_, v_i_404_);
lean_dec(v_v_403_);
lean_dec_ref(v_xs_402_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1(lean_object* v_xs_406_, lean_object* v_v_407_){
_start:
{
lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_408_ = lean_unsigned_to_nat(0u);
v___x_409_ = lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1_spec__2(v_xs_406_, v_v_407_, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1___boxed(lean_object* v_xs_410_, lean_object* v_v_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1(v_xs_410_, v_v_411_);
lean_dec(v_v_411_);
lean_dec_ref(v_xs_410_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(lean_object* v_x_413_, size_t v_x_414_, lean_object* v_x_415_){
_start:
{
if (lean_obj_tag(v_x_413_) == 0)
{
lean_object* v_es_416_; lean_object* v___x_417_; size_t v___x_418_; size_t v___x_419_; lean_object* v_j_420_; lean_object* v_entry_421_; 
v_es_416_ = lean_ctor_get(v_x_413_, 0);
v___x_417_ = lean_box(2);
v___x_418_ = ((size_t)31ULL);
v___x_419_ = lean_usize_land(v_x_414_, v___x_418_);
v_j_420_ = lean_usize_to_nat(v___x_419_);
v_entry_421_ = lean_array_get(v___x_417_, v_es_416_, v_j_420_);
switch(lean_obj_tag(v_entry_421_))
{
case 0:
{
lean_object* v_key_422_; uint8_t v___x_423_; 
v_key_422_ = lean_ctor_get(v_entry_421_, 0);
lean_inc(v_key_422_);
lean_dec_ref_known(v_entry_421_, 2);
v___x_423_ = l_Lean_instBEqMVarId_beq(v_x_415_, v_key_422_);
lean_dec(v_key_422_);
if (v___x_423_ == 0)
{
lean_dec(v_j_420_);
return v_x_413_;
}
else
{
lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_431_; 
lean_inc_ref(v_es_416_);
v_isSharedCheck_431_ = !lean_is_exclusive(v_x_413_);
if (v_isSharedCheck_431_ == 0)
{
lean_object* v_unused_432_; 
v_unused_432_ = lean_ctor_get(v_x_413_, 0);
lean_dec(v_unused_432_);
v___x_425_ = v_x_413_;
v_isShared_426_ = v_isSharedCheck_431_;
goto v_resetjp_424_;
}
else
{
lean_dec(v_x_413_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_431_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_427_; lean_object* v___x_429_; 
v___x_427_ = lean_array_set(v_es_416_, v_j_420_, v___x_417_);
lean_dec(v_j_420_);
if (v_isShared_426_ == 0)
{
lean_ctor_set(v___x_425_, 0, v___x_427_);
v___x_429_ = v___x_425_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_427_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
case 1:
{
lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_467_; 
lean_inc_ref(v_es_416_);
v_isSharedCheck_467_ = !lean_is_exclusive(v_x_413_);
if (v_isSharedCheck_467_ == 0)
{
lean_object* v_unused_468_; 
v_unused_468_ = lean_ctor_get(v_x_413_, 0);
lean_dec(v_unused_468_);
v___x_434_ = v_x_413_;
v_isShared_435_ = v_isSharedCheck_467_;
goto v_resetjp_433_;
}
else
{
lean_dec(v_x_413_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_467_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v_node_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_466_; 
v_node_436_ = lean_ctor_get(v_entry_421_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v_entry_421_);
if (v_isSharedCheck_466_ == 0)
{
v___x_438_ = v_entry_421_;
v_isShared_439_ = v_isSharedCheck_466_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_node_436_);
lean_dec(v_entry_421_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_466_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
size_t v___x_440_; lean_object* v_entries_441_; size_t v___x_442_; lean_object* v_newNode_443_; lean_object* v___x_444_; 
v___x_440_ = ((size_t)5ULL);
v_entries_441_ = lean_array_set(v_es_416_, v_j_420_, v___x_417_);
v___x_442_ = lean_usize_shift_right(v_x_414_, v___x_440_);
v_newNode_443_ = lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(v_node_436_, v___x_442_, v_x_415_);
lean_inc_ref(v_newNode_443_);
v___x_444_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_443_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_object* v___x_446_; 
if (v_isShared_439_ == 0)
{
lean_ctor_set(v___x_438_, 0, v_newNode_443_);
v___x_446_ = v___x_438_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_newNode_443_);
v___x_446_ = v_reuseFailAlloc_451_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_447_; lean_object* v___x_449_; 
v___x_447_ = lean_array_set(v_entries_441_, v_j_420_, v___x_446_);
lean_dec(v_j_420_);
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_447_);
v___x_449_ = v___x_434_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_447_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
else
{
lean_object* v_val_452_; lean_object* v_fst_453_; lean_object* v_snd_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_465_; 
lean_dec_ref(v_newNode_443_);
lean_del_object(v___x_438_);
v_val_452_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_val_452_);
lean_dec_ref_known(v___x_444_, 1);
v_fst_453_ = lean_ctor_get(v_val_452_, 0);
v_snd_454_ = lean_ctor_get(v_val_452_, 1);
v_isSharedCheck_465_ = !lean_is_exclusive(v_val_452_);
if (v_isSharedCheck_465_ == 0)
{
v___x_456_ = v_val_452_;
v_isShared_457_ = v_isSharedCheck_465_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_snd_454_);
lean_inc(v_fst_453_);
lean_dec(v_val_452_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_465_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_459_; 
if (v_isShared_457_ == 0)
{
v___x_459_ = v___x_456_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v_fst_453_);
lean_ctor_set(v_reuseFailAlloc_464_, 1, v_snd_454_);
v___x_459_ = v_reuseFailAlloc_464_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
lean_object* v___x_460_; lean_object* v___x_462_; 
v___x_460_ = lean_array_set(v_entries_441_, v_j_420_, v___x_459_);
lean_dec(v_j_420_);
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_460_);
v___x_462_ = v___x_434_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v___x_460_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
}
}
}
default: 
{
lean_dec(v_j_420_);
return v_x_413_;
}
}
}
else
{
lean_object* v_ks_469_; lean_object* v_vs_470_; lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_484_; 
v_ks_469_ = lean_ctor_get(v_x_413_, 0);
v_vs_470_ = lean_ctor_get(v_x_413_, 1);
v_isSharedCheck_484_ = !lean_is_exclusive(v_x_413_);
if (v_isSharedCheck_484_ == 0)
{
v___x_472_ = v_x_413_;
v_isShared_473_ = v_isSharedCheck_484_;
goto v_resetjp_471_;
}
else
{
lean_inc(v_vs_470_);
lean_inc(v_ks_469_);
lean_dec(v_x_413_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_484_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
lean_object* v___x_474_; 
v___x_474_ = lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0_spec__1(v_ks_469_, v_x_415_);
if (lean_obj_tag(v___x_474_) == 0)
{
lean_object* v___x_476_; 
if (v_isShared_473_ == 0)
{
v___x_476_ = v___x_472_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_ks_469_);
lean_ctor_set(v_reuseFailAlloc_477_, 1, v_vs_470_);
v___x_476_ = v_reuseFailAlloc_477_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
return v___x_476_;
}
}
else
{
lean_object* v_val_478_; lean_object* v_keys_x27_479_; lean_object* v_vals_x27_480_; lean_object* v___x_482_; 
v_val_478_ = lean_ctor_get(v___x_474_, 0);
lean_inc_n(v_val_478_, 2);
lean_dec_ref_known(v___x_474_, 1);
v_keys_x27_479_ = l_Array_eraseIdx___redArg(v_ks_469_, v_val_478_);
v_vals_x27_480_ = l_Array_eraseIdx___redArg(v_vs_470_, v_val_478_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 1, v_vals_x27_480_);
lean_ctor_set(v___x_472_, 0, v_keys_x27_479_);
v___x_482_ = v___x_472_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v_keys_x27_479_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v_vals_x27_480_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
return v___x_482_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg___boxed(lean_object* v_x_485_, lean_object* v_x_486_, lean_object* v_x_487_){
_start:
{
size_t v_x_181__boxed_488_; lean_object* v_res_489_; 
v_x_181__boxed_488_ = lean_unbox_usize(v_x_486_);
lean_dec(v_x_486_);
v_res_489_ = lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(v_x_485_, v_x_181__boxed_488_, v_x_487_);
lean_dec(v_x_487_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(lean_object* v_x_490_, lean_object* v_x_491_){
_start:
{
uint64_t v___x_492_; size_t v_h_493_; lean_object* v___x_494_; 
v___x_492_ = l_Lean_instHashableMVarId_hash(v_x_491_);
v_h_493_ = lean_uint64_to_usize(v___x_492_);
v___x_494_ = lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(v_x_490_, v_h_493_, v_x_491_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg___boxed(lean_object* v_x_495_, lean_object* v_x_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(v_x_495_, v_x_496_);
lean_dec(v_x_496_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(lean_object* v_mctx_498_, lean_object* v_mvarId_499_){
_start:
{
lean_object* v_depth_500_; lean_object* v_levelAssignDepth_501_; lean_object* v_lmvarCounter_502_; lean_object* v_mvarCounter_503_; lean_object* v_lDecls_504_; lean_object* v_decls_505_; lean_object* v_userNames_506_; lean_object* v_lAssignment_507_; lean_object* v_eAssignment_508_; lean_object* v_dAssignment_509_; lean_object* v___x_511_; uint8_t v_isShared_512_; uint8_t v_isSharedCheck_518_; 
v_depth_500_ = lean_ctor_get(v_mctx_498_, 0);
v_levelAssignDepth_501_ = lean_ctor_get(v_mctx_498_, 1);
v_lmvarCounter_502_ = lean_ctor_get(v_mctx_498_, 2);
v_mvarCounter_503_ = lean_ctor_get(v_mctx_498_, 3);
v_lDecls_504_ = lean_ctor_get(v_mctx_498_, 4);
v_decls_505_ = lean_ctor_get(v_mctx_498_, 5);
v_userNames_506_ = lean_ctor_get(v_mctx_498_, 6);
v_lAssignment_507_ = lean_ctor_get(v_mctx_498_, 7);
v_eAssignment_508_ = lean_ctor_get(v_mctx_498_, 8);
v_dAssignment_509_ = lean_ctor_get(v_mctx_498_, 9);
v_isSharedCheck_518_ = !lean_is_exclusive(v_mctx_498_);
if (v_isSharedCheck_518_ == 0)
{
v___x_511_ = v_mctx_498_;
v_isShared_512_ = v_isSharedCheck_518_;
goto v_resetjp_510_;
}
else
{
lean_inc(v_dAssignment_509_);
lean_inc(v_eAssignment_508_);
lean_inc(v_lAssignment_507_);
lean_inc(v_userNames_506_);
lean_inc(v_decls_505_);
lean_inc(v_lDecls_504_);
lean_inc(v_mvarCounter_503_);
lean_inc(v_lmvarCounter_502_);
lean_inc(v_levelAssignDepth_501_);
lean_inc(v_depth_500_);
lean_dec(v_mctx_498_);
v___x_511_ = lean_box(0);
v_isShared_512_ = v_isSharedCheck_518_;
goto v_resetjp_510_;
}
v_resetjp_510_:
{
lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_513_ = lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(v_eAssignment_508_, v_mvarId_499_);
v___x_514_ = lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(v_dAssignment_509_, v_mvarId_499_);
if (v_isShared_512_ == 0)
{
lean_ctor_set(v___x_511_, 9, v___x_514_);
lean_ctor_set(v___x_511_, 8, v___x_513_);
v___x_516_ = v___x_511_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_depth_500_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_levelAssignDepth_501_);
lean_ctor_set(v_reuseFailAlloc_517_, 2, v_lmvarCounter_502_);
lean_ctor_set(v_reuseFailAlloc_517_, 3, v_mvarCounter_503_);
lean_ctor_set(v_reuseFailAlloc_517_, 4, v_lDecls_504_);
lean_ctor_set(v_reuseFailAlloc_517_, 5, v_decls_505_);
lean_ctor_set(v_reuseFailAlloc_517_, 6, v_userNames_506_);
lean_ctor_set(v_reuseFailAlloc_517_, 7, v_lAssignment_507_);
lean_ctor_set(v_reuseFailAlloc_517_, 8, v___x_513_);
lean_ctor_set(v_reuseFailAlloc_517_, 9, v___x_514_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment___boxed(lean_object* v_mctx_519_, lean_object* v_mvarId_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(v_mctx_519_, v_mvarId_520_);
lean_dec(v_mvarId_520_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0(lean_object* v_00_u03b2_522_, lean_object* v_x_523_, lean_object* v_x_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___redArg(v_x_523_, v_x_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0___boxed(lean_object* v_00_u03b2_526_, lean_object* v_x_527_, lean_object* v_x_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_batteries_Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0(v_00_u03b2_526_, v_x_527_, v_x_528_);
lean_dec(v_x_528_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0(lean_object* v_00_u03b2_530_, lean_object* v_x_531_, size_t v_x_532_, lean_object* v_x_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___redArg(v_x_531_, v_x_532_, v_x_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0___boxed(lean_object* v_00_u03b2_535_, lean_object* v_x_536_, lean_object* v_x_537_, lean_object* v_x_538_){
_start:
{
size_t v_x_353__boxed_539_; lean_object* v_res_540_; 
v_x_353__boxed_539_ = lean_unbox_usize(v_x_537_);
lean_dec(v_x_537_);
v_res_540_ = lp_batteries_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Lean_MetavarContext_eraseExprMVarAssignment_spec__0_spec__0(v_00_u03b2_535_, v_x_536_, v_x_353__boxed_539_, v_x_538_);
lean_dec(v_x_538_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0(lean_object* v_eAssignment_541_, uint8_t v_includeDelayed_542_, lean_object* v_dAssignment_543_, lean_object* v_x_544_, lean_object* v_____s_545_){
_start:
{
lean_object* v_fst_546_; uint8_t v___x_550_; 
v_fst_546_ = lean_ctor_get(v_x_544_, 0);
lean_inc(v_fst_546_);
lean_dec_ref(v_x_544_);
v___x_550_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_eAssignment_541_, v_fst_546_);
if (v___x_550_ == 0)
{
if (v_includeDelayed_542_ == 0)
{
uint8_t v___x_551_; 
v___x_551_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned_spec__0___redArg(v_dAssignment_543_, v_fst_546_);
if (v___x_551_ == 0)
{
goto v___jp_547_;
}
else
{
lean_object* v___x_552_; 
lean_dec(v_fst_546_);
v___x_552_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_552_, 0, v_____s_545_);
return v___x_552_;
}
}
else
{
goto v___jp_547_;
}
}
else
{
lean_object* v___x_553_; 
lean_dec(v_fst_546_);
v___x_553_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_553_, 0, v_____s_545_);
return v___x_553_;
}
v___jp_547_:
{
lean_object* v_result_548_; lean_object* v___x_549_; 
v_result_548_ = lean_array_push(v_____s_545_, v_fst_546_);
v___x_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_549_, 0, v_result_548_);
return v___x_549_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0___boxed(lean_object* v_eAssignment_554_, lean_object* v_includeDelayed_555_, lean_object* v_dAssignment_556_, lean_object* v_x_557_, lean_object* v_____s_558_){
_start:
{
uint8_t v_includeDelayed_boxed_559_; lean_object* v_res_560_; 
v_includeDelayed_boxed_559_ = lean_unbox(v_includeDelayed_555_);
v_res_560_ = lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0(v_eAssignment_554_, v_includeDelayed_boxed_559_, v_dAssignment_556_, v_x_557_, v_____s_558_);
lean_dec_ref(v_dAssignment_556_);
lean_dec_ref(v_eAssignment_554_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_f_561_, lean_object* v_keys_562_, lean_object* v_vals_563_, lean_object* v_i_564_, lean_object* v_acc_565_){
_start:
{
lean_object* v___x_566_; uint8_t v___x_567_; 
v___x_566_ = lean_array_get_size(v_keys_562_);
v___x_567_ = lean_nat_dec_lt(v_i_564_, v___x_566_);
if (v___x_567_ == 0)
{
lean_object* v___x_568_; 
lean_dec(v_i_564_);
lean_dec_ref(v_f_561_);
v___x_568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_568_, 0, v_acc_565_);
return v___x_568_;
}
else
{
lean_object* v_k_569_; lean_object* v_v_570_; lean_object* v___x_571_; 
v_k_569_ = lean_array_fget_borrowed(v_keys_562_, v_i_564_);
v_v_570_ = lean_array_fget_borrowed(v_vals_563_, v_i_564_);
lean_inc_ref(v_f_561_);
lean_inc(v_v_570_);
lean_inc(v_k_569_);
v___x_571_ = lean_apply_3(v_f_561_, v_acc_565_, v_k_569_, v_v_570_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_dec(v_i_564_);
lean_dec_ref(v_f_561_);
return v___x_571_;
}
else
{
lean_object* v_a_572_; lean_object* v___x_573_; lean_object* v___x_574_; 
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
lean_dec_ref_known(v___x_571_, 1);
v___x_573_ = lean_unsigned_to_nat(1u);
v___x_574_ = lean_nat_add(v_i_564_, v___x_573_);
lean_dec(v_i_564_);
v_i_564_ = v___x_574_;
v_acc_565_ = v_a_572_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_f_576_, lean_object* v_keys_577_, lean_object* v_vals_578_, lean_object* v_i_579_, lean_object* v_acc_580_){
_start:
{
lean_object* v_res_581_; 
v_res_581_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_f_576_, v_keys_577_, v_vals_578_, v_i_579_, v_acc_580_);
lean_dec_ref(v_vals_578_);
lean_dec_ref(v_keys_577_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(lean_object* v_f_582_, lean_object* v_x_583_, lean_object* v_x_584_){
_start:
{
if (lean_obj_tag(v_x_583_) == 0)
{
lean_object* v_es_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_605_; 
v_es_585_ = lean_ctor_get(v_x_583_, 0);
v_isSharedCheck_605_ = !lean_is_exclusive(v_x_583_);
if (v_isSharedCheck_605_ == 0)
{
v___x_587_ = v_x_583_;
v_isShared_588_ = v_isSharedCheck_605_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_es_585_);
lean_dec(v_x_583_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_605_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_589_; lean_object* v___x_590_; uint8_t v___x_591_; 
v___x_589_ = lean_unsigned_to_nat(0u);
v___x_590_ = lean_array_get_size(v_es_585_);
v___x_591_ = lean_nat_dec_lt(v___x_589_, v___x_590_);
if (v___x_591_ == 0)
{
lean_object* v___x_593_; 
lean_dec_ref(v_es_585_);
lean_dec_ref(v_f_582_);
if (v_isShared_588_ == 0)
{
lean_ctor_set_tag(v___x_587_, 1);
lean_ctor_set(v___x_587_, 0, v_x_584_);
v___x_593_ = v___x_587_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_x_584_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
else
{
uint8_t v___x_595_; 
v___x_595_ = lean_nat_dec_le(v___x_590_, v___x_590_);
if (v___x_595_ == 0)
{
if (v___x_591_ == 0)
{
lean_object* v___x_597_; 
lean_dec_ref(v_es_585_);
lean_dec_ref(v_f_582_);
if (v_isShared_588_ == 0)
{
lean_ctor_set_tag(v___x_587_, 1);
lean_ctor_set(v___x_587_, 0, v_x_584_);
v___x_597_ = v___x_587_;
goto v_reusejp_596_;
}
else
{
lean_object* v_reuseFailAlloc_598_; 
v_reuseFailAlloc_598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_598_, 0, v_x_584_);
v___x_597_ = v_reuseFailAlloc_598_;
goto v_reusejp_596_;
}
v_reusejp_596_:
{
return v___x_597_;
}
}
else
{
size_t v___x_599_; size_t v___x_600_; lean_object* v___x_601_; 
lean_del_object(v___x_587_);
v___x_599_ = ((size_t)0ULL);
v___x_600_ = lean_usize_of_nat(v___x_590_);
v___x_601_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(v_f_582_, v_es_585_, v___x_599_, v___x_600_, v_x_584_);
lean_dec_ref(v_es_585_);
return v___x_601_;
}
}
else
{
size_t v___x_602_; size_t v___x_603_; lean_object* v___x_604_; 
lean_del_object(v___x_587_);
v___x_602_ = ((size_t)0ULL);
v___x_603_ = lean_usize_of_nat(v___x_590_);
v___x_604_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(v_f_582_, v_es_585_, v___x_602_, v___x_603_, v_x_584_);
lean_dec_ref(v_es_585_);
return v___x_604_;
}
}
}
}
else
{
lean_object* v_ks_606_; lean_object* v_vs_607_; lean_object* v___x_608_; lean_object* v___x_609_; 
v_ks_606_ = lean_ctor_get(v_x_583_, 0);
lean_inc_ref(v_ks_606_);
v_vs_607_ = lean_ctor_get(v_x_583_, 1);
lean_inc_ref(v_vs_607_);
lean_dec_ref_known(v_x_583_, 2);
v___x_608_ = lean_unsigned_to_nat(0u);
v___x_609_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_f_582_, v_ks_606_, v_vs_607_, v___x_608_, v_x_584_);
lean_dec_ref(v_vs_607_);
lean_dec_ref(v_ks_606_);
return v___x_609_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_f_610_, lean_object* v_as_611_, size_t v_i_612_, size_t v_stop_613_, lean_object* v_b_614_){
_start:
{
lean_object* v_a_616_; lean_object* v___y_621_; uint8_t v___x_623_; 
v___x_623_ = lean_usize_dec_eq(v_i_612_, v_stop_613_);
if (v___x_623_ == 0)
{
lean_object* v___x_624_; 
v___x_624_ = lean_array_uget_borrowed(v_as_611_, v_i_612_);
switch(lean_obj_tag(v___x_624_))
{
case 0:
{
lean_object* v_key_625_; lean_object* v_val_626_; lean_object* v___x_627_; 
v_key_625_ = lean_ctor_get(v___x_624_, 0);
v_val_626_ = lean_ctor_get(v___x_624_, 1);
lean_inc_ref(v_f_610_);
lean_inc(v_val_626_);
lean_inc(v_key_625_);
v___x_627_ = lean_apply_3(v_f_610_, v_b_614_, v_key_625_, v_val_626_);
v___y_621_ = v___x_627_;
goto v___jp_620_;
}
case 1:
{
lean_object* v_node_628_; lean_object* v___x_629_; 
v_node_628_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_node_628_);
lean_inc_ref(v_f_610_);
v___x_629_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(v_f_610_, v_node_628_, v_b_614_);
v___y_621_ = v___x_629_;
goto v___jp_620_;
}
default: 
{
v_a_616_ = v_b_614_;
goto v___jp_615_;
}
}
}
else
{
lean_object* v___x_630_; 
lean_dec_ref(v_f_610_);
v___x_630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_630_, 0, v_b_614_);
return v___x_630_;
}
v___jp_615_:
{
size_t v___x_617_; size_t v___x_618_; 
v___x_617_ = ((size_t)1ULL);
v___x_618_ = lean_usize_add(v_i_612_, v___x_617_);
v_i_612_ = v___x_618_;
v_b_614_ = v_a_616_;
goto _start;
}
v___jp_620_:
{
if (lean_obj_tag(v___y_621_) == 0)
{
lean_dec_ref(v_f_610_);
return v___y_621_;
}
else
{
lean_object* v_a_622_; 
v_a_622_ = lean_ctor_get(v___y_621_, 0);
lean_inc(v_a_622_);
lean_dec_ref_known(v___y_621_, 1);
v_a_616_ = v_a_622_;
goto v___jp_615_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_f_631_, lean_object* v_as_632_, lean_object* v_i_633_, lean_object* v_stop_634_, lean_object* v_b_635_){
_start:
{
size_t v_i_boxed_636_; size_t v_stop_boxed_637_; lean_object* v_res_638_; 
v_i_boxed_636_ = lean_unbox_usize(v_i_633_);
lean_dec(v_i_633_);
v_stop_boxed_637_ = lean_unbox_usize(v_stop_634_);
lean_dec(v_stop_634_);
v_res_638_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(v_f_631_, v_as_632_, v_i_boxed_636_, v_stop_boxed_637_, v_b_635_);
lean_dec_ref(v_as_632_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg___lam__0(lean_object* v_f_639_, lean_object* v_s_640_, lean_object* v_a_641_, lean_object* v_b_642_){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_643_, 0, v_a_641_);
lean_ctor_set(v___x_643_, 1, v_b_642_);
v___x_644_ = lean_apply_2(v_f_639_, v___x_643_, v_s_640_);
if (lean_obj_tag(v___x_644_) == 0)
{
lean_object* v_a_645_; lean_object* v___x_647_; uint8_t v_isShared_648_; uint8_t v_isSharedCheck_652_; 
v_a_645_ = lean_ctor_get(v___x_644_, 0);
v_isSharedCheck_652_ = !lean_is_exclusive(v___x_644_);
if (v_isSharedCheck_652_ == 0)
{
v___x_647_ = v___x_644_;
v_isShared_648_ = v_isSharedCheck_652_;
goto v_resetjp_646_;
}
else
{
lean_inc(v_a_645_);
lean_dec(v___x_644_);
v___x_647_ = lean_box(0);
v_isShared_648_ = v_isSharedCheck_652_;
goto v_resetjp_646_;
}
v_resetjp_646_:
{
lean_object* v___x_650_; 
if (v_isShared_648_ == 0)
{
v___x_650_ = v___x_647_;
goto v_reusejp_649_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v_a_645_);
v___x_650_ = v_reuseFailAlloc_651_;
goto v_reusejp_649_;
}
v_reusejp_649_:
{
return v___x_650_;
}
}
}
else
{
lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_660_; 
v_a_653_ = lean_ctor_get(v___x_644_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_644_);
if (v_isSharedCheck_660_ == 0)
{
v___x_655_ = v___x_644_;
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_644_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_658_; 
if (v_isShared_656_ == 0)
{
v___x_658_ = v___x_655_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_a_653_);
v___x_658_ = v_reuseFailAlloc_659_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
return v___x_658_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg(lean_object* v_map_661_, lean_object* v_init_662_, lean_object* v_f_663_){
_start:
{
lean_object* v___f_664_; lean_object* v___x_665_; lean_object* v_a_666_; 
v___f_664_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg___lam__0), 4, 1);
lean_closure_set(v___f_664_, 0, v_f_663_);
lean_inc_ref(v_map_661_);
v___x_665_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(v___f_664_, v_map_661_, v_init_662_);
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref(v___x_665_);
return v_a_666_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg___boxed(lean_object* v_map_667_, lean_object* v_init_668_, lean_object* v_f_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg(v_map_667_, v_init_668_, v_f_669_);
lean_dec_ref(v_map_667_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars(lean_object* v_mctx_673_, uint8_t v_includeDelayed_674_){
_start:
{
lean_object* v_decls_675_; lean_object* v_eAssignment_676_; lean_object* v_dAssignment_677_; lean_object* v___x_678_; lean_object* v___f_679_; lean_object* v_result_680_; lean_object* v___x_681_; 
v_decls_675_ = lean_ctor_get(v_mctx_673_, 5);
lean_inc_ref(v_decls_675_);
v_eAssignment_676_ = lean_ctor_get(v_mctx_673_, 8);
lean_inc_ref(v_eAssignment_676_);
v_dAssignment_677_ = lean_ctor_get(v_mctx_673_, 9);
lean_inc_ref(v_dAssignment_677_);
lean_dec_ref(v_mctx_673_);
v___x_678_ = lean_box(v_includeDelayed_674_);
v___f_679_ = lean_alloc_closure((void*)(lp_batteries_Lean_MetavarContext_unassignedExprMVars___lam__0___boxed), 5, 3);
lean_closure_set(v___f_679_, 0, v_eAssignment_676_);
lean_closure_set(v___f_679_, 1, v___x_678_);
lean_closure_set(v___f_679_, 2, v_dAssignment_677_);
v_result_680_ = ((lean_object*)(lp_batteries_Lean_MetavarContext_unassignedExprMVars___closed__0));
v___x_681_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg(v_decls_675_, v_result_680_, v___f_679_);
lean_dec_ref(v_decls_675_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars___boxed(lean_object* v_mctx_682_, lean_object* v_includeDelayed_683_){
_start:
{
uint8_t v_includeDelayed_boxed_684_; lean_object* v_res_685_; 
v_includeDelayed_boxed_684_ = lean_unbox(v_includeDelayed_683_);
v_res_685_ = lp_batteries_Lean_MetavarContext_unassignedExprMVars(v_mctx_682_, v_includeDelayed_boxed_684_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0(lean_object* v_00_u03c3_686_, lean_object* v_00_u03b2_687_, lean_object* v_map_688_, lean_object* v_init_689_, lean_object* v_f_690_){
_start:
{
lean_object* v___x_691_; 
v___x_691_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___redArg(v_map_688_, v_init_689_, v_f_690_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0___boxed(lean_object* v_00_u03c3_692_, lean_object* v_00_u03b2_693_, lean_object* v_map_694_, lean_object* v_init_695_, lean_object* v_f_696_){
_start:
{
lean_object* v_res_697_; 
v_res_697_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0(v_00_u03c3_692_, v_00_u03b2_693_, v_map_694_, v_init_695_, v_f_696_);
lean_dec_ref(v_map_694_);
return v_res_697_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0___redArg(lean_object* v_map_698_, lean_object* v_f_699_, lean_object* v_init_700_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(v_f_699_, v_map_698_, v_init_700_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0(lean_object* v_00_u03c3_702_, lean_object* v_00_u03c3_703_, lean_object* v_00_u03b2_704_, lean_object* v_map_705_, lean_object* v_f_706_, lean_object* v_init_707_){
_start:
{
lean_object* v___x_708_; 
v___x_708_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(v_f_706_, v_map_705_, v_init_707_);
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_709_, lean_object* v_00_u03c3_710_, lean_object* v_00_u03b1_711_, lean_object* v_00_u03b2_712_, lean_object* v_f_713_, lean_object* v_x_714_, lean_object* v_x_715_){
_start:
{
lean_object* v___x_716_; 
v___x_716_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1___redArg(v_f_713_, v_x_714_, v_x_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_717_, lean_object* v_00_u03b2_718_, lean_object* v_00_u03c3_719_, lean_object* v_00_u03c3_720_, lean_object* v_f_721_, lean_object* v_as_722_, size_t v_i_723_, size_t v_stop_724_, lean_object* v_b_725_){
_start:
{
lean_object* v___x_726_; 
v___x_726_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___redArg(v_f_721_, v_as_722_, v_i_723_, v_stop_724_, v_b_725_);
return v___x_726_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_727_, lean_object* v_00_u03b2_728_, lean_object* v_00_u03c3_729_, lean_object* v_00_u03c3_730_, lean_object* v_f_731_, lean_object* v_as_732_, lean_object* v_i_733_, lean_object* v_stop_734_, lean_object* v_b_735_){
_start:
{
size_t v_i_boxed_736_; size_t v_stop_boxed_737_; lean_object* v_res_738_; 
v_i_boxed_736_ = lean_unbox_usize(v_i_733_);
lean_dec(v_i_733_);
v_stop_boxed_737_ = lean_unbox_usize(v_stop_734_);
lean_dec(v_stop_734_);
v_res_738_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_727_, v_00_u03b2_728_, v_00_u03c3_729_, v_00_u03c3_730_, v_f_731_, v_as_732_, v_i_boxed_736_, v_stop_boxed_737_, v_b_735_);
lean_dec_ref(v_as_732_);
return v_res_738_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03c3_739_, lean_object* v_00_u03c3_740_, lean_object* v_00_u03b1_741_, lean_object* v_00_u03b2_742_, lean_object* v_f_743_, lean_object* v_keys_744_, lean_object* v_vals_745_, lean_object* v_heq_746_, lean_object* v_i_747_, lean_object* v_acc_748_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_f_743_, v_keys_744_, v_vals_745_, v_i_747_, v_acc_748_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03c3_750_, lean_object* v_00_u03c3_751_, lean_object* v_00_u03b1_752_, lean_object* v_00_u03b2_753_, lean_object* v_f_754_, lean_object* v_keys_755_, lean_object* v_vals_756_, lean_object* v_heq_757_, lean_object* v_i_758_, lean_object* v_acc_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_MetavarContext_unassignedExprMVars_spec__0_spec__0_spec__1_spec__3(v_00_u03c3_750_, v_00_u03c3_751_, v_00_u03b1_752_, v_00_u03b2_753_, v_f_754_, v_keys_755_, v_vals_756_, v_heq_757_, v_i_758_, v_acc_759_);
lean_dec_ref(v_vals_756_);
lean_dec_ref(v_keys_755_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0(lean_object* v_mvarId_761_, lean_object* v_toPure_762_, lean_object* v_____do__lift_763_){
_start:
{
uint8_t v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_764_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_____do__lift_763_, v_mvarId_761_);
v___x_765_ = lean_box(v___x_764_);
v___x_766_ = lean_apply_2(v_toPure_762_, lean_box(0), v___x_765_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0___boxed(lean_object* v_mvarId_767_, lean_object* v_toPure_768_, lean_object* v_____do__lift_769_){
_start:
{
lean_object* v_res_770_; 
v_res_770_ = lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0(v_mvarId_767_, v_toPure_768_, v_____do__lift_769_);
lean_dec_ref(v_____do__lift_769_);
lean_dec(v_mvarId_767_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___redArg(lean_object* v_inst_771_, lean_object* v_inst_772_, lean_object* v_mvarId_773_){
_start:
{
lean_object* v_toApplicative_774_; lean_object* v_toBind_775_; lean_object* v_getMCtx_776_; lean_object* v_toPure_777_; lean_object* v___f_778_; lean_object* v___x_779_; 
v_toApplicative_774_ = lean_ctor_get(v_inst_771_, 0);
lean_inc_ref(v_toApplicative_774_);
v_toBind_775_ = lean_ctor_get(v_inst_771_, 1);
lean_inc(v_toBind_775_);
lean_dec_ref(v_inst_771_);
v_getMCtx_776_ = lean_ctor_get(v_inst_772_, 0);
lean_inc(v_getMCtx_776_);
lean_dec_ref(v_inst_772_);
v_toPure_777_ = lean_ctor_get(v_toApplicative_774_, 1);
lean_inc(v_toPure_777_);
lean_dec_ref(v_toApplicative_774_);
v___f_778_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_isDeclared___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_778_, 0, v_mvarId_773_);
lean_closure_set(v___f_778_, 1, v_toPure_777_);
v___x_779_ = lean_apply_4(v_toBind_775_, lean_box(0), lean_box(0), v_getMCtx_776_, v___f_778_);
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared(lean_object* v_m_780_, lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_mvarId_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lp_batteries_Lean_MVarId_isDeclared___redArg(v_inst_781_, v_inst_782_, v_mvarId_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0(lean_object* v_mvarId_785_, lean_object* v_x_786_){
_start:
{
lean_object* v___x_787_; 
v___x_787_ = lp_batteries_Lean_MetavarContext_eraseExprMVarAssignment(v_x_786_, v_mvarId_785_);
return v___x_787_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0___boxed(lean_object* v_mvarId_788_, lean_object* v_x_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0(v_mvarId_788_, v_x_789_);
lean_dec(v_mvarId_788_);
return v_res_790_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment___redArg(lean_object* v_inst_791_, lean_object* v_mvarId_792_){
_start:
{
lean_object* v_modifyMCtx_793_; lean_object* v___f_794_; lean_object* v___x_795_; 
v_modifyMCtx_793_ = lean_ctor_get(v_inst_791_, 1);
lean_inc(v_modifyMCtx_793_);
lean_dec_ref(v_inst_791_);
v___f_794_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_eraseAssignment___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_794_, 0, v_mvarId_792_);
v___x_795_ = lean_apply_1(v_modifyMCtx_793_, v___f_794_);
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_eraseAssignment(lean_object* v_m_796_, lean_object* v_inst_797_, lean_object* v_mvarId_798_){
_start:
{
lean_object* v___x_799_; 
v___x_799_ = lp_batteries_Lean_MVarId_eraseAssignment___redArg(v_inst_797_, v_mvarId_798_);
return v___x_799_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg(lean_object* v_mvarId_800_, lean_object* v_val_801_, lean_object* v___y_802_){
_start:
{
lean_object* v___x_804_; lean_object* v_mctx_805_; lean_object* v_cache_806_; lean_object* v_zetaDeltaFVarIds_807_; lean_object* v_postponed_808_; lean_object* v_diag_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_837_; 
v___x_804_ = lean_st_ref_take(v___y_802_);
v_mctx_805_ = lean_ctor_get(v___x_804_, 0);
v_cache_806_ = lean_ctor_get(v___x_804_, 1);
v_zetaDeltaFVarIds_807_ = lean_ctor_get(v___x_804_, 2);
v_postponed_808_ = lean_ctor_get(v___x_804_, 3);
v_diag_809_ = lean_ctor_get(v___x_804_, 4);
v_isSharedCheck_837_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_837_ == 0)
{
v___x_811_ = v___x_804_;
v_isShared_812_ = v_isSharedCheck_837_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_diag_809_);
lean_inc(v_postponed_808_);
lean_inc(v_zetaDeltaFVarIds_807_);
lean_inc(v_cache_806_);
lean_inc(v_mctx_805_);
lean_dec(v___x_804_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_837_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v_depth_813_; lean_object* v_levelAssignDepth_814_; lean_object* v_lmvarCounter_815_; lean_object* v_mvarCounter_816_; lean_object* v_lDecls_817_; lean_object* v_decls_818_; lean_object* v_userNames_819_; lean_object* v_lAssignment_820_; lean_object* v_eAssignment_821_; lean_object* v_dAssignment_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_836_; 
v_depth_813_ = lean_ctor_get(v_mctx_805_, 0);
v_levelAssignDepth_814_ = lean_ctor_get(v_mctx_805_, 1);
v_lmvarCounter_815_ = lean_ctor_get(v_mctx_805_, 2);
v_mvarCounter_816_ = lean_ctor_get(v_mctx_805_, 3);
v_lDecls_817_ = lean_ctor_get(v_mctx_805_, 4);
v_decls_818_ = lean_ctor_get(v_mctx_805_, 5);
v_userNames_819_ = lean_ctor_get(v_mctx_805_, 6);
v_lAssignment_820_ = lean_ctor_get(v_mctx_805_, 7);
v_eAssignment_821_ = lean_ctor_get(v_mctx_805_, 8);
v_dAssignment_822_ = lean_ctor_get(v_mctx_805_, 9);
v_isSharedCheck_836_ = !lean_is_exclusive(v_mctx_805_);
if (v_isSharedCheck_836_ == 0)
{
v___x_824_ = v_mctx_805_;
v_isShared_825_ = v_isSharedCheck_836_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_dAssignment_822_);
lean_inc(v_eAssignment_821_);
lean_inc(v_lAssignment_820_);
lean_inc(v_userNames_819_);
lean_inc(v_decls_818_);
lean_inc(v_lDecls_817_);
lean_inc(v_mvarCounter_816_);
lean_inc(v_lmvarCounter_815_);
lean_inc(v_levelAssignDepth_814_);
lean_inc(v_depth_813_);
lean_dec(v_mctx_805_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_836_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___x_826_; lean_object* v___x_828_; 
v___x_826_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MetavarContext_declareExprMVar_spec__0___redArg(v_eAssignment_821_, v_mvarId_800_, v_val_801_);
if (v_isShared_825_ == 0)
{
lean_ctor_set(v___x_824_, 8, v___x_826_);
v___x_828_ = v___x_824_;
goto v_reusejp_827_;
}
else
{
lean_object* v_reuseFailAlloc_835_; 
v_reuseFailAlloc_835_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_835_, 0, v_depth_813_);
lean_ctor_set(v_reuseFailAlloc_835_, 1, v_levelAssignDepth_814_);
lean_ctor_set(v_reuseFailAlloc_835_, 2, v_lmvarCounter_815_);
lean_ctor_set(v_reuseFailAlloc_835_, 3, v_mvarCounter_816_);
lean_ctor_set(v_reuseFailAlloc_835_, 4, v_lDecls_817_);
lean_ctor_set(v_reuseFailAlloc_835_, 5, v_decls_818_);
lean_ctor_set(v_reuseFailAlloc_835_, 6, v_userNames_819_);
lean_ctor_set(v_reuseFailAlloc_835_, 7, v_lAssignment_820_);
lean_ctor_set(v_reuseFailAlloc_835_, 8, v___x_826_);
lean_ctor_set(v_reuseFailAlloc_835_, 9, v_dAssignment_822_);
v___x_828_ = v_reuseFailAlloc_835_;
goto v_reusejp_827_;
}
v_reusejp_827_:
{
lean_object* v___x_830_; 
if (v_isShared_812_ == 0)
{
lean_ctor_set(v___x_811_, 0, v___x_828_);
v___x_830_ = v___x_811_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_828_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v_cache_806_);
lean_ctor_set(v_reuseFailAlloc_834_, 2, v_zetaDeltaFVarIds_807_);
lean_ctor_set(v_reuseFailAlloc_834_, 3, v_postponed_808_);
lean_ctor_set(v_reuseFailAlloc_834_, 4, v_diag_809_);
v___x_830_ = v_reuseFailAlloc_834_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; 
v___x_831_ = lean_st_ref_set(v___y_802_, v___x_830_);
v___x_832_ = lean_box(0);
v___x_833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_833_, 0, v___x_832_);
return v___x_833_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg___boxed(lean_object* v_mvarId_838_, lean_object* v_val_839_, lean_object* v___y_840_, lean_object* v___y_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg(v_mvarId_838_, v_val_839_, v___y_840_);
lean_dec(v___y_840_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_synthInstance(lean_object* v_g_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_){
_start:
{
lean_object* v___x_849_; 
lean_inc(v_g_843_);
v___x_849_ = l_Lean_MVarId_getType(v_g_843_, v_a_844_, v_a_845_, v_a_846_, v_a_847_);
if (lean_obj_tag(v___x_849_) == 0)
{
lean_object* v_a_850_; lean_object* v___x_851_; lean_object* v___x_852_; 
v_a_850_ = lean_ctor_get(v___x_849_, 0);
lean_inc(v_a_850_);
lean_dec_ref_known(v___x_849_, 1);
v___x_851_ = lean_box(0);
v___x_852_ = l_Lean_Meta_synthInstance(v_a_850_, v___x_851_, v_a_844_, v_a_845_, v_a_846_, v_a_847_);
if (lean_obj_tag(v___x_852_) == 0)
{
lean_object* v_a_853_; lean_object* v___x_854_; 
v_a_853_ = lean_ctor_get(v___x_852_, 0);
lean_inc(v_a_853_);
lean_dec_ref_known(v___x_852_, 1);
v___x_854_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg(v_g_843_, v_a_853_, v_a_845_);
return v___x_854_;
}
else
{
lean_object* v_a_855_; lean_object* v___x_857_; uint8_t v_isShared_858_; uint8_t v_isSharedCheck_862_; 
lean_dec(v_g_843_);
v_a_855_ = lean_ctor_get(v___x_852_, 0);
v_isSharedCheck_862_ = !lean_is_exclusive(v___x_852_);
if (v_isSharedCheck_862_ == 0)
{
v___x_857_ = v___x_852_;
v_isShared_858_ = v_isSharedCheck_862_;
goto v_resetjp_856_;
}
else
{
lean_inc(v_a_855_);
lean_dec(v___x_852_);
v___x_857_ = lean_box(0);
v_isShared_858_ = v_isSharedCheck_862_;
goto v_resetjp_856_;
}
v_resetjp_856_:
{
lean_object* v___x_860_; 
if (v_isShared_858_ == 0)
{
v___x_860_ = v___x_857_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v_a_855_);
v___x_860_ = v_reuseFailAlloc_861_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
return v___x_860_;
}
}
}
}
else
{
lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_870_; 
lean_dec(v_g_843_);
v_a_863_ = lean_ctor_get(v___x_849_, 0);
v_isSharedCheck_870_ = !lean_is_exclusive(v___x_849_);
if (v_isSharedCheck_870_ == 0)
{
v___x_865_ = v___x_849_;
v_isShared_866_ = v_isSharedCheck_870_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___x_849_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_870_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v___x_868_; 
if (v_isShared_866_ == 0)
{
v___x_868_ = v___x_865_;
goto v_reusejp_867_;
}
else
{
lean_object* v_reuseFailAlloc_869_; 
v_reuseFailAlloc_869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_869_, 0, v_a_863_);
v___x_868_ = v_reuseFailAlloc_869_;
goto v_reusejp_867_;
}
v_reusejp_867_:
{
return v___x_868_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_synthInstance___boxed(lean_object* v_g_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_, lean_object* v_a_875_, lean_object* v_a_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_batteries_Lean_MVarId_synthInstance(v_g_871_, v_a_872_, v_a_873_, v_a_874_, v_a_875_);
lean_dec(v_a_875_);
lean_dec_ref(v_a_874_);
lean_dec(v_a_873_);
lean_dec_ref(v_a_872_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0(lean_object* v_mvarId_878_, lean_object* v_val_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_){
_start:
{
lean_object* v___x_885_; 
v___x_885_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___redArg(v_mvarId_878_, v_val_879_, v___y_881_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0___boxed(lean_object* v_mvarId_886_, lean_object* v_val_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_batteries_Lean_MVarId_assign___at___00Lean_MVarId_synthInstance_spec__0(v_mvarId_886_, v_val_887_, v___y_888_, v___y_889_, v___y_890_, v___y_891_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg(lean_object* v_e_894_, lean_object* v___y_895_){
_start:
{
uint8_t v___x_897_; 
v___x_897_ = l_Lean_Expr_hasMVar(v_e_894_);
if (v___x_897_ == 0)
{
lean_object* v___x_898_; 
v___x_898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_898_, 0, v_e_894_);
return v___x_898_;
}
else
{
lean_object* v___x_899_; lean_object* v_mctx_900_; lean_object* v___x_901_; lean_object* v_fst_902_; lean_object* v_snd_903_; lean_object* v___x_904_; lean_object* v_cache_905_; lean_object* v_zetaDeltaFVarIds_906_; lean_object* v_postponed_907_; lean_object* v_diag_908_; lean_object* v___x_910_; uint8_t v_isShared_911_; uint8_t v_isSharedCheck_917_; 
v___x_899_ = lean_st_ref_get(v___y_895_);
v_mctx_900_ = lean_ctor_get(v___x_899_, 0);
lean_inc_ref(v_mctx_900_);
lean_dec(v___x_899_);
v___x_901_ = l_Lean_instantiateMVarsCore(v_mctx_900_, v_e_894_);
v_fst_902_ = lean_ctor_get(v___x_901_, 0);
lean_inc(v_fst_902_);
v_snd_903_ = lean_ctor_get(v___x_901_, 1);
lean_inc(v_snd_903_);
lean_dec_ref(v___x_901_);
v___x_904_ = lean_st_ref_take(v___y_895_);
v_cache_905_ = lean_ctor_get(v___x_904_, 1);
v_zetaDeltaFVarIds_906_ = lean_ctor_get(v___x_904_, 2);
v_postponed_907_ = lean_ctor_get(v___x_904_, 3);
v_diag_908_ = lean_ctor_get(v___x_904_, 4);
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_904_);
if (v_isSharedCheck_917_ == 0)
{
lean_object* v_unused_918_; 
v_unused_918_ = lean_ctor_get(v___x_904_, 0);
lean_dec(v_unused_918_);
v___x_910_ = v___x_904_;
v_isShared_911_ = v_isSharedCheck_917_;
goto v_resetjp_909_;
}
else
{
lean_inc(v_diag_908_);
lean_inc(v_postponed_907_);
lean_inc(v_zetaDeltaFVarIds_906_);
lean_inc(v_cache_905_);
lean_dec(v___x_904_);
v___x_910_ = lean_box(0);
v_isShared_911_ = v_isSharedCheck_917_;
goto v_resetjp_909_;
}
v_resetjp_909_:
{
lean_object* v___x_913_; 
if (v_isShared_911_ == 0)
{
lean_ctor_set(v___x_910_, 0, v_snd_903_);
v___x_913_ = v___x_910_;
goto v_reusejp_912_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v_snd_903_);
lean_ctor_set(v_reuseFailAlloc_916_, 1, v_cache_905_);
lean_ctor_set(v_reuseFailAlloc_916_, 2, v_zetaDeltaFVarIds_906_);
lean_ctor_set(v_reuseFailAlloc_916_, 3, v_postponed_907_);
lean_ctor_set(v_reuseFailAlloc_916_, 4, v_diag_908_);
v___x_913_ = v_reuseFailAlloc_916_;
goto v_reusejp_912_;
}
v_reusejp_912_:
{
lean_object* v___x_914_; lean_object* v___x_915_; 
v___x_914_ = lean_st_ref_set(v___y_895_, v___x_913_);
v___x_915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_915_, 0, v_fst_902_);
return v___x_915_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg___boxed(lean_object* v_e_919_, lean_object* v___y_920_, lean_object* v___y_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg(v_e_919_, v___y_920_);
lean_dec(v___y_920_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0(lean_object* v_e_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_){
_start:
{
lean_object* v___x_929_; 
v___x_929_ = lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg(v_e_923_, v___y_925_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___boxed(lean_object* v_e_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_){
_start:
{
lean_object* v_res_936_; 
v_res_936_ = lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0(v_e_930_, v___y_931_, v___y_932_, v___y_933_, v___y_934_);
lean_dec(v___y_934_);
lean_dec_ref(v___y_933_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
return v_res_936_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_getTypeCleanup(lean_object* v_mvarId_937_, lean_object* v_a_938_, lean_object* v_a_939_, lean_object* v_a_940_, lean_object* v_a_941_){
_start:
{
lean_object* v___x_943_; 
v___x_943_ = l_Lean_MVarId_getType(v_mvarId_937_, v_a_938_, v_a_939_, v_a_940_, v_a_941_);
if (lean_obj_tag(v___x_943_) == 0)
{
lean_object* v_a_944_; lean_object* v___x_945_; lean_object* v_a_946_; lean_object* v___x_948_; uint8_t v_isShared_949_; uint8_t v_isSharedCheck_954_; 
v_a_944_ = lean_ctor_get(v___x_943_, 0);
lean_inc(v_a_944_);
lean_dec_ref_known(v___x_943_, 1);
v___x_945_ = lp_batteries_Lean_instantiateMVars___at___00Lean_MVarId_getTypeCleanup_spec__0___redArg(v_a_944_, v_a_939_);
v_a_946_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_954_ == 0)
{
v___x_948_ = v___x_945_;
v_isShared_949_ = v_isSharedCheck_954_;
goto v_resetjp_947_;
}
else
{
lean_inc(v_a_946_);
lean_dec(v___x_945_);
v___x_948_ = lean_box(0);
v_isShared_949_ = v_isSharedCheck_954_;
goto v_resetjp_947_;
}
v_resetjp_947_:
{
lean_object* v___x_950_; lean_object* v___x_952_; 
v___x_950_ = l_Lean_Expr_cleanupAnnotations(v_a_946_);
if (v_isShared_949_ == 0)
{
lean_ctor_set(v___x_948_, 0, v___x_950_);
v___x_952_ = v___x_948_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v___x_950_);
v___x_952_ = v_reuseFailAlloc_953_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
return v___x_952_;
}
}
}
else
{
return v___x_943_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_getTypeCleanup___boxed(lean_object* v_mvarId_955_, lean_object* v_a_956_, lean_object* v_a_957_, lean_object* v_a_958_, lean_object* v_a_959_, lean_object* v_a_960_){
_start:
{
lean_object* v_res_961_; 
v_res_961_ = lp_batteries_Lean_MVarId_getTypeCleanup(v_mvarId_955_, v_a_956_, v_a_957_, v_a_958_, v_a_959_);
lean_dec(v_a_959_);
lean_dec_ref(v_a_958_);
lean_dec(v_a_957_);
lean_dec_ref(v_a_956_);
return v_res_961_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0(uint8_t v_includeDelayed_962_, lean_object* v_toPure_963_, lean_object* v_____do__lift_964_){
_start:
{
lean_object* v___x_965_; lean_object* v___x_966_; 
v___x_965_ = lp_batteries_Lean_MetavarContext_unassignedExprMVars(v_____do__lift_964_, v_includeDelayed_962_);
v___x_966_ = lean_apply_2(v_toPure_963_, lean_box(0), v___x_965_);
return v___x_966_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0___boxed(lean_object* v_includeDelayed_967_, lean_object* v_toPure_968_, lean_object* v_____do__lift_969_){
_start:
{
uint8_t v_includeDelayed_boxed_970_; lean_object* v_res_971_; 
v_includeDelayed_boxed_970_ = lean_unbox(v_includeDelayed_967_);
v_res_971_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0(v_includeDelayed_boxed_970_, v_toPure_968_, v_____do__lift_969_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg(lean_object* v_inst_972_, lean_object* v_inst_973_, uint8_t v_includeDelayed_974_){
_start:
{
lean_object* v_toApplicative_975_; lean_object* v_toBind_976_; lean_object* v_getMCtx_977_; lean_object* v_toPure_978_; lean_object* v___x_979_; lean_object* v___f_980_; lean_object* v___x_981_; 
v_toApplicative_975_ = lean_ctor_get(v_inst_972_, 0);
lean_inc_ref(v_toApplicative_975_);
v_toBind_976_ = lean_ctor_get(v_inst_972_, 1);
lean_inc(v_toBind_976_);
lean_dec_ref(v_inst_972_);
v_getMCtx_977_ = lean_ctor_get(v_inst_973_, 0);
lean_inc(v_getMCtx_977_);
lean_dec_ref(v_inst_973_);
v_toPure_978_ = lean_ctor_get(v_toApplicative_975_, 1);
lean_inc(v_toPure_978_);
lean_dec_ref(v_toApplicative_975_);
v___x_979_ = lean_box(v_includeDelayed_974_);
v___f_980_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_980_, 0, v___x_979_);
lean_closure_set(v___f_980_, 1, v_toPure_978_);
v___x_981_ = lean_apply_4(v_toBind_976_, lean_box(0), lean_box(0), v_getMCtx_977_, v___f_980_);
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg___boxed(lean_object* v_inst_982_, lean_object* v_inst_983_, lean_object* v_includeDelayed_984_){
_start:
{
uint8_t v_includeDelayed_boxed_985_; lean_object* v_res_986_; 
v_includeDelayed_boxed_985_ = lean_unbox(v_includeDelayed_984_);
v_res_986_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg(v_inst_982_, v_inst_983_, v_includeDelayed_boxed_985_);
return v_res_986_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars(lean_object* v_m_987_, lean_object* v_inst_988_, lean_object* v_inst_989_, uint8_t v_includeDelayed_990_){
_start:
{
lean_object* v___x_991_; 
v___x_991_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___redArg(v_inst_988_, v_inst_989_, v_includeDelayed_990_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___boxed(lean_object* v_m_992_, lean_object* v_inst_993_, lean_object* v_inst_994_, lean_object* v_includeDelayed_995_){
_start:
{
uint8_t v_includeDelayed_boxed_996_; lean_object* v_res_997_; 
v_includeDelayed_boxed_996_ = lean_unbox(v_includeDelayed_995_);
v_res_997_ = lp_batteries_Lean_Meta_getUnassignedExprMVars(v_m_992_, v_inst_993_, v_inst_994_, v_includeDelayed_boxed_996_);
return v_res_997_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic___redArg___lam__0(lean_object* v___x_998_, lean_object* v_x_999_){
_start:
{
lean_object* v___x_1000_; uint8_t v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_1000_ = l_Lean_Meta_tactic_hygienic;
v___x_1001_ = 0;
v___x_1002_ = lean_box(v___x_1001_);
v___x_1003_ = l_Lean_Option_set___redArg(v___x_998_, v_x_999_, v___x_1000_, v___x_1002_);
return v___x_1003_;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_unhygienic___redArg___closed__0(void){
_start:
{
lean_object* v___x_1004_; lean_object* v___f_1005_; 
v___x_1004_ = l_Lean_KVMap_instValueBool;
v___f_1005_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_unhygienic___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1005_, 0, v___x_1004_);
return v___f_1005_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic___redArg(lean_object* v_inst_1006_, lean_object* v_x_1007_){
_start:
{
lean_object* v___f_1008_; lean_object* v___x_1009_; 
v___f_1008_ = lean_obj_once(&lp_batteries_Lean_Meta_unhygienic___redArg___closed__0, &lp_batteries_Lean_Meta_unhygienic___redArg___closed__0_once, _init_lp_batteries_Lean_Meta_unhygienic___redArg___closed__0);
v___x_1009_ = lean_apply_3(v_inst_1006_, lean_box(0), v___f_1008_, v_x_1007_);
return v___x_1009_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_unhygienic(lean_object* v_m_1010_, lean_object* v_00_u03b1_1011_, lean_object* v_inst_1012_, lean_object* v_x_1013_){
_start:
{
lean_object* v___x_1014_; 
v___x_1014_ = lp_batteries_Lean_Meta_unhygienic___redArg(v_inst_1012_, v_x_1013_);
return v___x_1014_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__0(lean_object* v_toPure_1015_, lean_object* v_r_1016_, lean_object* v_____r_1017_){
_start:
{
lean_object* v___x_1018_; 
v___x_1018_ = lean_apply_2(v_toPure_1015_, lean_box(0), v_r_1016_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__1(lean_object* v_prefix_1019_, lean_object* v_toPure_1020_, lean_object* v_setNGen_1021_, lean_object* v_toBind_1022_, lean_object* v_ngen_1023_){
_start:
{
lean_object* v_namePrefix_1024_; lean_object* v_idx_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1038_; 
v_namePrefix_1024_ = lean_ctor_get(v_ngen_1023_, 0);
v_idx_1025_ = lean_ctor_get(v_ngen_1023_, 1);
v_isSharedCheck_1038_ = !lean_is_exclusive(v_ngen_1023_);
if (v_isSharedCheck_1038_ == 0)
{
v___x_1027_ = v_ngen_1023_;
v_isShared_1028_ = v_isSharedCheck_1038_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_idx_1025_);
lean_inc(v_namePrefix_1024_);
lean_dec(v_ngen_1023_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1038_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
lean_object* v_r_1029_; lean_object* v___f_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1034_; 
lean_inc(v_idx_1025_);
v_r_1029_ = l_Lean_Name_num___override(v_prefix_1019_, v_idx_1025_);
v___f_1030_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1030_, 0, v_toPure_1020_);
lean_closure_set(v___f_1030_, 1, v_r_1029_);
v___x_1031_ = lean_unsigned_to_nat(1u);
v___x_1032_ = lean_nat_add(v_idx_1025_, v___x_1031_);
lean_dec(v_idx_1025_);
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 1, v___x_1032_);
v___x_1034_ = v___x_1027_;
goto v_reusejp_1033_;
}
else
{
lean_object* v_reuseFailAlloc_1037_; 
v_reuseFailAlloc_1037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1037_, 0, v_namePrefix_1024_);
lean_ctor_set(v_reuseFailAlloc_1037_, 1, v___x_1032_);
v___x_1034_ = v_reuseFailAlloc_1037_;
goto v_reusejp_1033_;
}
v_reusejp_1033_:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; 
v___x_1035_ = lean_apply_1(v_setNGen_1021_, v___x_1034_);
v___x_1036_ = lean_apply_4(v_toBind_1022_, lean_box(0), lean_box(0), v___x_1035_, v___f_1030_);
return v___x_1036_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg(lean_object* v_inst_1039_, lean_object* v_inst_1040_, lean_object* v_prefix_1041_){
_start:
{
lean_object* v_toApplicative_1042_; lean_object* v_toBind_1043_; lean_object* v_getNGen_1044_; lean_object* v_setNGen_1045_; lean_object* v_toPure_1046_; lean_object* v___f_1047_; lean_object* v___x_1048_; 
v_toApplicative_1042_ = lean_ctor_get(v_inst_1039_, 0);
lean_inc_ref(v_toApplicative_1042_);
v_toBind_1043_ = lean_ctor_get(v_inst_1039_, 1);
lean_inc_n(v_toBind_1043_, 2);
lean_dec_ref(v_inst_1039_);
v_getNGen_1044_ = lean_ctor_get(v_inst_1040_, 0);
lean_inc(v_getNGen_1044_);
v_setNGen_1045_ = lean_ctor_get(v_inst_1040_, 1);
lean_inc(v_setNGen_1045_);
lean_dec_ref(v_inst_1040_);
v_toPure_1046_ = lean_ctor_get(v_toApplicative_1042_, 1);
lean_inc(v_toPure_1046_);
lean_dec_ref(v_toApplicative_1042_);
v___f_1047_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg___lam__1), 5, 4);
lean_closure_set(v___f_1047_, 0, v_prefix_1041_);
lean_closure_set(v___f_1047_, 1, v_toPure_1046_);
lean_closure_set(v___f_1047_, 2, v_setNGen_1045_);
lean_closure_set(v___f_1047_, 3, v_toBind_1043_);
v___x_1048_ = lean_apply_4(v_toBind_1043_, lean_box(0), lean_box(0), v_getNGen_1044_, v___f_1047_);
return v___x_1048_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_mkFreshIdWithPrefix(lean_object* v_m_1049_, lean_object* v_inst_1050_, lean_object* v_inst_1051_, lean_object* v_prefix_1052_){
_start:
{
lean_object* v___x_1053_; 
v___x_1053_ = lp_batteries_Lean_Meta_mkFreshIdWithPrefix___redArg(v_inst_1050_, v_inst_1051_, v_prefix_1052_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__0(lean_object* v_goal_1054_, lean_object* v_s_1055_){
_start:
{
lean_object* v___x_1056_; 
v___x_1056_ = lean_array_push(v_s_1055_, v_goal_1054_);
return v___x_1056_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__2(lean_object* v_acc_1057_, lean_object* v___f_1058_, lean_object* v_inst_1059_, lean_object* v_toApplicative_1060_, lean_object* v_inst_1061_, lean_object* v___f_1062_, lean_object* v_____do__lift_1063_){
_start:
{
if (lean_obj_tag(v_____do__lift_1063_) == 0)
{
lean_object* v___x_1064_; lean_object* v___x_1065_; 
lean_dec(v___f_1062_);
lean_dec_ref(v_inst_1061_);
lean_dec_ref(v_toApplicative_1060_);
v___x_1064_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_modifyUnsafe___boxed), 5, 4);
lean_closure_set(v___x_1064_, 0, lean_box(0));
lean_closure_set(v___x_1064_, 1, lean_box(0));
lean_closure_set(v___x_1064_, 2, v_acc_1057_);
lean_closure_set(v___x_1064_, 3, v___f_1058_);
v___x_1065_ = lean_apply_2(v_inst_1059_, lean_box(0), v___x_1064_);
return v___x_1065_;
}
else
{
lean_object* v_val_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; uint8_t v___x_1070_; 
lean_dec(v_inst_1059_);
lean_dec_ref(v___f_1058_);
lean_dec(v_acc_1057_);
v_val_1066_ = lean_ctor_get(v_____do__lift_1063_, 0);
lean_inc(v_val_1066_);
lean_dec_ref_known(v_____do__lift_1063_, 1);
v___x_1067_ = lean_unsigned_to_nat(0u);
v___x_1068_ = lean_array_get_size(v_val_1066_);
v___x_1069_ = lean_box(0);
v___x_1070_ = lean_nat_dec_lt(v___x_1067_, v___x_1068_);
if (v___x_1070_ == 0)
{
lean_object* v_toPure_1071_; lean_object* v___x_1072_; 
lean_dec(v_val_1066_);
lean_dec(v___f_1062_);
lean_dec_ref(v_inst_1061_);
v_toPure_1071_ = lean_ctor_get(v_toApplicative_1060_, 1);
lean_inc(v_toPure_1071_);
lean_dec_ref(v_toApplicative_1060_);
v___x_1072_ = lean_apply_2(v_toPure_1071_, lean_box(0), v___x_1069_);
return v___x_1072_;
}
else
{
uint8_t v___x_1073_; 
v___x_1073_ = lean_nat_dec_le(v___x_1068_, v___x_1068_);
if (v___x_1073_ == 0)
{
if (v___x_1070_ == 0)
{
lean_object* v_toPure_1074_; lean_object* v___x_1075_; 
lean_dec(v_val_1066_);
lean_dec(v___f_1062_);
lean_dec_ref(v_inst_1061_);
v_toPure_1074_ = lean_ctor_get(v_toApplicative_1060_, 1);
lean_inc(v_toPure_1074_);
lean_dec_ref(v_toApplicative_1060_);
v___x_1075_ = lean_apply_2(v_toPure_1074_, lean_box(0), v___x_1069_);
return v___x_1075_;
}
else
{
size_t v___x_1076_; size_t v___x_1077_; lean_object* v___x_1078_; 
lean_dec_ref(v_toApplicative_1060_);
v___x_1076_ = ((size_t)0ULL);
v___x_1077_ = lean_usize_of_nat(v___x_1068_);
v___x_1078_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1061_, v___f_1062_, v_val_1066_, v___x_1076_, v___x_1077_, v___x_1069_);
return v___x_1078_;
}
}
else
{
size_t v___x_1079_; size_t v___x_1080_; lean_object* v___x_1081_; 
lean_dec_ref(v_toApplicative_1060_);
v___x_1079_ = ((size_t)0ULL);
v___x_1080_ = lean_usize_of_nat(v___x_1068_);
v___x_1081_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1061_, v___f_1062_, v_val_1066_, v___x_1079_, v___x_1080_, v___x_1069_);
return v___x_1081_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__3(lean_object* v_inst_1082_, lean_object* v_____do__lift_1083_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = l_Lean_throwMaxRecDepthAt___redArg(v_inst_1082_, v_____do__lift_1083_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4(lean_object* v_curr_1085_, lean_object* v_withRecDepth_1086_, lean_object* v___x_1087_, lean_object* v_inst_1088_, lean_object* v_toBind_1089_, lean_object* v___f_1090_, lean_object* v_max_1091_){
_start:
{
lean_object* v___x_1096_; uint8_t v___x_1097_; 
v___x_1096_ = lean_unsigned_to_nat(0u);
v___x_1097_ = lean_nat_dec_eq(v_max_1091_, v___x_1096_);
if (v___x_1097_ == 0)
{
uint8_t v___x_1098_; 
v___x_1098_ = lean_nat_dec_eq(v_curr_1085_, v_max_1091_);
if (v___x_1098_ == 0)
{
lean_dec(v___f_1090_);
lean_dec(v_toBind_1089_);
lean_dec_ref(v_inst_1088_);
goto v___jp_1092_;
}
else
{
lean_object* v_toMonadRef_1099_; lean_object* v_getRef_1100_; lean_object* v___x_1101_; 
lean_dec(v___x_1087_);
lean_dec(v_withRecDepth_1086_);
v_toMonadRef_1099_ = lean_ctor_get(v_inst_1088_, 1);
lean_inc_ref(v_toMonadRef_1099_);
lean_dec_ref(v_inst_1088_);
v_getRef_1100_ = lean_ctor_get(v_toMonadRef_1099_, 0);
lean_inc(v_getRef_1100_);
lean_dec_ref(v_toMonadRef_1099_);
v___x_1101_ = lean_apply_4(v_toBind_1089_, lean_box(0), lean_box(0), v_getRef_1100_, v___f_1090_);
return v___x_1101_;
}
}
else
{
lean_dec(v___f_1090_);
lean_dec(v_toBind_1089_);
lean_dec_ref(v_inst_1088_);
goto v___jp_1092_;
}
v___jp_1092_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1093_ = lean_unsigned_to_nat(1u);
v___x_1094_ = lean_nat_add(v_curr_1085_, v___x_1093_);
v___x_1095_ = lean_apply_3(v_withRecDepth_1086_, lean_box(0), v___x_1094_, v___x_1087_);
return v___x_1095_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4___boxed(lean_object* v_curr_1102_, lean_object* v_withRecDepth_1103_, lean_object* v___x_1104_, lean_object* v_inst_1105_, lean_object* v_toBind_1106_, lean_object* v___f_1107_, lean_object* v_max_1108_){
_start:
{
lean_object* v_res_1109_; 
v_res_1109_ = lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4(v_curr_1102_, v_withRecDepth_1103_, v___x_1104_, v_inst_1105_, v_toBind_1106_, v___f_1107_, v_max_1108_);
lean_dec(v_max_1108_);
lean_dec(v_curr_1102_);
return v_res_1109_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__5(lean_object* v_withRecDepth_1110_, lean_object* v___x_1111_, lean_object* v_inst_1112_, lean_object* v_toBind_1113_, lean_object* v___f_1114_, lean_object* v_getMaxRecDepth_1115_, lean_object* v_curr_1116_){
_start:
{
lean_object* v___f_1117_; lean_object* v___x_1118_; 
lean_inc(v_toBind_1113_);
v___f_1117_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__4___boxed), 7, 6);
lean_closure_set(v___f_1117_, 0, v_curr_1116_);
lean_closure_set(v___f_1117_, 1, v_withRecDepth_1110_);
lean_closure_set(v___f_1117_, 2, v___x_1111_);
lean_closure_set(v___f_1117_, 3, v_inst_1112_);
lean_closure_set(v___f_1117_, 4, v_toBind_1113_);
lean_closure_set(v___f_1117_, 5, v___f_1114_);
v___x_1118_ = lean_apply_4(v_toBind_1113_, lean_box(0), lean_box(0), v_getMaxRecDepth_1115_, v___f_1117_);
return v___x_1118_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg(lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_inst_1121_, lean_object* v_inst_1122_, lean_object* v_tac_1123_, lean_object* v_acc_1124_, lean_object* v_goal_1125_){
_start:
{
lean_object* v_toApplicative_1126_; lean_object* v_toBind_1127_; lean_object* v_withRecDepth_1128_; lean_object* v_getRecDepth_1129_; lean_object* v_getMaxRecDepth_1130_; lean_object* v___f_1131_; lean_object* v___f_1132_; lean_object* v___f_1133_; lean_object* v___f_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___f_1137_; lean_object* v___x_1138_; 
v_toApplicative_1126_ = lean_ctor_get(v_inst_1119_, 0);
lean_inc_ref(v_toApplicative_1126_);
v_toBind_1127_ = lean_ctor_get(v_inst_1119_, 1);
lean_inc_n(v_toBind_1127_, 3);
v_withRecDepth_1128_ = lean_ctor_get(v_inst_1121_, 0);
lean_inc(v_withRecDepth_1128_);
v_getRecDepth_1129_ = lean_ctor_get(v_inst_1121_, 1);
lean_inc(v_getRecDepth_1129_);
v_getMaxRecDepth_1130_ = lean_ctor_get(v_inst_1121_, 2);
lean_inc(v_getMaxRecDepth_1130_);
lean_inc(v_goal_1125_);
v___f_1131_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1131_, 0, v_goal_1125_);
lean_inc(v_acc_1124_);
lean_inc(v_tac_1123_);
lean_inc(v_inst_1122_);
lean_inc_ref_n(v_inst_1120_, 2);
lean_inc_ref(v_inst_1119_);
v___f_1132_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__1), 8, 6);
lean_closure_set(v___f_1132_, 0, v_inst_1119_);
lean_closure_set(v___f_1132_, 1, v_inst_1120_);
lean_closure_set(v___f_1132_, 2, v_inst_1121_);
lean_closure_set(v___f_1132_, 3, v_inst_1122_);
lean_closure_set(v___f_1132_, 4, v_tac_1123_);
lean_closure_set(v___f_1132_, 5, v_acc_1124_);
v___f_1133_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__2), 7, 6);
lean_closure_set(v___f_1133_, 0, v_acc_1124_);
lean_closure_set(v___f_1133_, 1, v___f_1131_);
lean_closure_set(v___f_1133_, 2, v_inst_1122_);
lean_closure_set(v___f_1133_, 3, v_toApplicative_1126_);
lean_closure_set(v___f_1133_, 4, v_inst_1119_);
lean_closure_set(v___f_1133_, 5, v___f_1132_);
v___f_1134_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__3), 2, 1);
lean_closure_set(v___f_1134_, 0, v_inst_1120_);
v___x_1135_ = lean_apply_1(v_tac_1123_, v_goal_1125_);
v___x_1136_ = lean_apply_4(v_toBind_1127_, lean_box(0), lean_box(0), v___x_1135_, v___f_1133_);
v___f_1137_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__5), 7, 6);
lean_closure_set(v___f_1137_, 0, v_withRecDepth_1128_);
lean_closure_set(v___f_1137_, 1, v___x_1136_);
lean_closure_set(v___f_1137_, 2, v_inst_1120_);
lean_closure_set(v___f_1137_, 3, v_toBind_1127_);
lean_closure_set(v___f_1137_, 4, v___f_1134_);
lean_closure_set(v___f_1137_, 5, v_getMaxRecDepth_1130_);
v___x_1138_ = lean_apply_4(v_toBind_1127_, lean_box(0), lean_box(0), v_getRecDepth_1129_, v___f_1137_);
return v___x_1138_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg___lam__1(lean_object* v_inst_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_, lean_object* v_inst_1142_, lean_object* v_tac_1143_, lean_object* v_acc_1144_, lean_object* v_x_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v___x_1147_; 
v___x_1147_ = lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg(v_inst_1139_, v_inst_1140_, v_inst_1141_, v_inst_1142_, v_tac_1143_, v_acc_1144_, v___y_1146_);
return v___x_1147_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go(lean_object* v_m_1148_, lean_object* v_inst_1149_, lean_object* v_inst_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_, lean_object* v_tac_1153_, lean_object* v_acc_1154_, lean_object* v_goal_1155_){
_start:
{
lean_object* v___x_1156_; 
v___x_1156_ = lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg(v_inst_1149_, v_inst_1150_, v_inst_1151_, v_inst_1152_, v_tac_1153_, v_acc_1154_, v_goal_1155_);
return v___x_1156_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__0(lean_object* v_toPure_1157_, lean_object* v_____do__lift_1158_){
_start:
{
lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_1159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1159_, 0, v_____do__lift_1158_);
v___x_1160_ = lean_apply_2(v_toPure_1157_, lean_box(0), v___x_1159_);
return v___x_1160_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__1(lean_object* v_acc_1161_, lean_object* v_inst_1162_, lean_object* v_toBind_1163_, lean_object* v___f_1164_, lean_object* v_____r_1165_){
_start:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1166_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_1166_, 0, lean_box(0));
lean_closure_set(v___x_1166_, 1, lean_box(0));
lean_closure_set(v___x_1166_, 2, v_acc_1161_);
v___x_1167_ = lean_apply_2(v_inst_1162_, lean_box(0), v___x_1166_);
v___x_1168_ = lean_apply_4(v_toBind_1163_, lean_box(0), lean_box(0), v___x_1167_, v___f_1164_);
return v___x_1168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__2(lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_inst_1171_, lean_object* v_inst_1172_, lean_object* v_tac_1173_, lean_object* v_acc_1174_, lean_object* v_x_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v___x_1177_; 
v___x_1177_ = lp_batteries___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___redArg(v_inst_1169_, v_inst_1170_, v_inst_1171_, v_inst_1172_, v_tac_1173_, v_acc_1174_, v___y_1176_);
return v___x_1177_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__3(lean_object* v_inst_1178_, lean_object* v_toBind_1179_, lean_object* v___f_1180_, lean_object* v_val_1181_, lean_object* v_toPure_1182_, lean_object* v_inst_1183_, lean_object* v_inst_1184_, lean_object* v_inst_1185_, lean_object* v_tac_1186_, lean_object* v_acc_1187_){
_start:
{
lean_object* v___f_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; uint8_t v___x_1192_; 
lean_inc(v_toBind_1179_);
lean_inc(v_inst_1178_);
lean_inc(v_acc_1187_);
v___f_1188_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__1), 5, 4);
lean_closure_set(v___f_1188_, 0, v_acc_1187_);
lean_closure_set(v___f_1188_, 1, v_inst_1178_);
lean_closure_set(v___f_1188_, 2, v_toBind_1179_);
lean_closure_set(v___f_1188_, 3, v___f_1180_);
v___x_1189_ = lean_unsigned_to_nat(0u);
v___x_1190_ = lean_array_get_size(v_val_1181_);
v___x_1191_ = lean_box(0);
v___x_1192_ = lean_nat_dec_lt(v___x_1189_, v___x_1190_);
if (v___x_1192_ == 0)
{
lean_object* v___x_1193_; lean_object* v___x_1194_; 
lean_dec(v_acc_1187_);
lean_dec(v_tac_1186_);
lean_dec_ref(v_inst_1185_);
lean_dec_ref(v_inst_1184_);
lean_dec_ref(v_inst_1183_);
lean_dec_ref(v_val_1181_);
lean_dec(v_inst_1178_);
v___x_1193_ = lean_apply_2(v_toPure_1182_, lean_box(0), v___x_1191_);
v___x_1194_ = lean_apply_4(v_toBind_1179_, lean_box(0), lean_box(0), v___x_1193_, v___f_1188_);
return v___x_1194_;
}
else
{
lean_object* v___f_1195_; uint8_t v___x_1196_; 
lean_inc_ref(v_inst_1183_);
v___f_1195_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__2), 8, 6);
lean_closure_set(v___f_1195_, 0, v_inst_1183_);
lean_closure_set(v___f_1195_, 1, v_inst_1184_);
lean_closure_set(v___f_1195_, 2, v_inst_1185_);
lean_closure_set(v___f_1195_, 3, v_inst_1178_);
lean_closure_set(v___f_1195_, 4, v_tac_1186_);
lean_closure_set(v___f_1195_, 5, v_acc_1187_);
v___x_1196_ = lean_nat_dec_le(v___x_1190_, v___x_1190_);
if (v___x_1196_ == 0)
{
if (v___x_1192_ == 0)
{
lean_object* v___x_1197_; lean_object* v___x_1198_; 
lean_dec_ref(v___f_1195_);
lean_dec_ref(v_inst_1183_);
lean_dec_ref(v_val_1181_);
v___x_1197_ = lean_apply_2(v_toPure_1182_, lean_box(0), v___x_1191_);
v___x_1198_ = lean_apply_4(v_toBind_1179_, lean_box(0), lean_box(0), v___x_1197_, v___f_1188_);
return v___x_1198_;
}
else
{
size_t v___x_1199_; size_t v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
lean_dec(v_toPure_1182_);
v___x_1199_ = ((size_t)0ULL);
v___x_1200_ = lean_usize_of_nat(v___x_1190_);
v___x_1201_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1183_, v___f_1195_, v_val_1181_, v___x_1199_, v___x_1200_, v___x_1191_);
v___x_1202_ = lean_apply_4(v_toBind_1179_, lean_box(0), lean_box(0), v___x_1201_, v___f_1188_);
return v___x_1202_;
}
}
else
{
size_t v___x_1203_; size_t v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; 
lean_dec(v_toPure_1182_);
v___x_1203_ = ((size_t)0ULL);
v___x_1204_ = lean_usize_of_nat(v___x_1190_);
v___x_1205_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1183_, v___f_1195_, v_val_1181_, v___x_1203_, v___x_1204_, v___x_1191_);
v___x_1206_ = lean_apply_4(v_toBind_1179_, lean_box(0), lean_box(0), v___x_1205_, v___f_1188_);
return v___x_1206_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg___lam__4(lean_object* v_inst_1209_, lean_object* v_toBind_1210_, lean_object* v___f_1211_, lean_object* v_toPure_1212_, lean_object* v_inst_1213_, lean_object* v_inst_1214_, lean_object* v_inst_1215_, lean_object* v_tac_1216_, lean_object* v_____x_1217_){
_start:
{
if (lean_obj_tag(v_____x_1217_) == 1)
{
lean_object* v_val_1218_; lean_object* v___f_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; 
v_val_1218_ = lean_ctor_get(v_____x_1217_, 0);
lean_inc(v_val_1218_);
lean_dec_ref_known(v_____x_1217_, 1);
lean_inc(v_toBind_1210_);
lean_inc(v_inst_1209_);
v___f_1219_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__3), 10, 9);
lean_closure_set(v___f_1219_, 0, v_inst_1209_);
lean_closure_set(v___f_1219_, 1, v_toBind_1210_);
lean_closure_set(v___f_1219_, 2, v___f_1211_);
lean_closure_set(v___f_1219_, 3, v_val_1218_);
lean_closure_set(v___f_1219_, 4, v_toPure_1212_);
lean_closure_set(v___f_1219_, 5, v_inst_1213_);
lean_closure_set(v___f_1219_, 6, v_inst_1214_);
lean_closure_set(v___f_1219_, 7, v_inst_1215_);
lean_closure_set(v___f_1219_, 8, v_tac_1216_);
v___x_1220_ = ((lean_object*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__4___closed__0));
v___x_1221_ = lean_apply_2(v_inst_1209_, lean_box(0), v___x_1220_);
v___x_1222_ = lean_apply_4(v_toBind_1210_, lean_box(0), lean_box(0), v___x_1221_, v___f_1219_);
return v___x_1222_;
}
else
{
lean_object* v___x_1223_; lean_object* v___x_1224_; 
lean_dec(v_____x_1217_);
lean_dec(v_tac_1216_);
lean_dec_ref(v_inst_1215_);
lean_dec_ref(v_inst_1214_);
lean_dec_ref(v_inst_1213_);
lean_dec(v___f_1211_);
lean_dec(v_toBind_1210_);
lean_dec(v_inst_1209_);
v___x_1223_ = lean_box(0);
v___x_1224_ = lean_apply_2(v_toPure_1212_, lean_box(0), v___x_1223_);
return v___x_1224_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1___redArg(lean_object* v_inst_1225_, lean_object* v_inst_1226_, lean_object* v_inst_1227_, lean_object* v_inst_1228_, lean_object* v_goal_1229_, lean_object* v_tac_1230_){
_start:
{
lean_object* v_toApplicative_1231_; lean_object* v_toBind_1232_; lean_object* v_toPure_1233_; lean_object* v___x_1234_; lean_object* v___f_1235_; lean_object* v___f_1236_; lean_object* v___x_1237_; 
v_toApplicative_1231_ = lean_ctor_get(v_inst_1225_, 0);
v_toBind_1232_ = lean_ctor_get(v_inst_1225_, 1);
lean_inc_n(v_toBind_1232_, 2);
v_toPure_1233_ = lean_ctor_get(v_toApplicative_1231_, 1);
lean_inc_n(v_toPure_1233_, 2);
lean_inc(v_tac_1230_);
v___x_1234_ = lean_apply_1(v_tac_1230_, v_goal_1229_);
v___f_1235_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1235_, 0, v_toPure_1233_);
v___f_1236_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_saturate1___redArg___lam__4), 9, 8);
lean_closure_set(v___f_1236_, 0, v_inst_1228_);
lean_closure_set(v___f_1236_, 1, v_toBind_1232_);
lean_closure_set(v___f_1236_, 2, v___f_1235_);
lean_closure_set(v___f_1236_, 3, v_toPure_1233_);
lean_closure_set(v___f_1236_, 4, v_inst_1225_);
lean_closure_set(v___f_1236_, 5, v_inst_1226_);
lean_closure_set(v___f_1236_, 6, v_inst_1227_);
lean_closure_set(v___f_1236_, 7, v_tac_1230_);
v___x_1237_ = lean_apply_4(v_toBind_1232_, lean_box(0), lean_box(0), v___x_1234_, v___f_1236_);
return v___x_1237_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_saturate1(lean_object* v_m_1238_, lean_object* v_inst_1239_, lean_object* v_inst_1240_, lean_object* v_inst_1241_, lean_object* v_inst_1242_, lean_object* v_goal_1243_, lean_object* v_tac_1244_){
_start:
{
lean_object* v___x_1245_; 
v___x_1245_ = lp_batteries_Lean_Meta_saturate1___redArg(v_inst_1239_, v_inst_1240_, v_inst_1241_, v_inst_1242_, v_goal_1243_, v_tac_1244_);
return v___x_1245_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_SynthInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_SynthInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
lean_object* initialize_Lean_Meta_SynthInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_SynthInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
