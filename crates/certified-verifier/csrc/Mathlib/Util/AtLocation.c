// Lean compiler output
// Module: Mathlib.Util.AtLocation
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Location public meta import Lean.Meta.Tactic.Simp.Main public import Lean.Elab.Tactic.Location
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
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getNondepPropHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_applySimpResultToLocalDecl(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheoremsArray_eraseTheorem(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setSimpTheorems(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isTrue(lean_object*);
lean_object* l_Lean_Meta_applySimpResultToTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkOfEqTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "not a nondependent Prop hypothesis"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged;
static const lean_string_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Tactic.BehaviorIfUnchanged.silent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Mathlib.Tactic.BehaviorIfUnchanged.warning"};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Mathlib.Tactic.BehaviorIfUnchanged.error"};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged = (const lean_object*)&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "` made no progress on the goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "` made no progress at `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Cannot run `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "` at `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "`, it is an implementation detail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "` made no progress anywhere"};
static const lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0(lean_object* v___y_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2_, v___y_5_, v___y_6_, v___y_7_, v___y_8_);
if (lean_obj_tag(v___x_10_) == 0)
{
lean_object* v_a_11_; lean_object* v___x_12_; 
v_a_11_ = lean_ctor_get(v___x_10_, 0);
lean_inc(v_a_11_);
lean_dec_ref_known(v___x_10_, 1);
v___x_12_ = l_Lean_MVarId_getNondepPropHyps(v_a_11_, v___y_5_, v___y_6_, v___y_7_, v___y_8_);
return v___x_12_;
}
else
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_20_; 
v_a_13_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_20_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_20_ == 0)
{
v___x_15_ = v___x_10_;
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_10_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0___boxed(lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__0(v___y_21_, v___y_22_, v___y_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_, v___y_28_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
lean_dec(v___y_26_);
lean_dec_ref(v___y_25_);
lean_dec(v___y_24_);
lean_dec_ref(v___y_23_);
lean_dec(v___y_22_);
lean_dec_ref(v___y_21_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(lean_object* v_msgData_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_){
_start:
{
lean_object* v___x_37_; lean_object* v_env_38_; lean_object* v___x_39_; lean_object* v_mctx_40_; lean_object* v_lctx_41_; lean_object* v_options_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_37_ = lean_st_ref_get(v___y_35_);
v_env_38_ = lean_ctor_get(v___x_37_, 0);
lean_inc_ref(v_env_38_);
lean_dec(v___x_37_);
v___x_39_ = lean_st_ref_get(v___y_33_);
v_mctx_40_ = lean_ctor_get(v___x_39_, 0);
lean_inc_ref(v_mctx_40_);
lean_dec(v___x_39_);
v_lctx_41_ = lean_ctor_get(v___y_32_, 2);
v_options_42_ = lean_ctor_get(v___y_34_, 2);
lean_inc_ref(v_options_42_);
lean_inc_ref(v_lctx_41_);
v___x_43_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_43_, 0, v_env_38_);
lean_ctor_set(v___x_43_, 1, v_mctx_40_);
lean_ctor_set(v___x_43_, 2, v_lctx_41_);
lean_ctor_set(v___x_43_, 3, v_options_42_);
v___x_44_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v_msgData_31_);
v___x_45_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2___boxed(lean_object* v_msgData_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(v_msgData_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(lean_object* v_msg_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v_ref_59_; lean_object* v___x_60_; lean_object* v_a_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_69_; 
v_ref_59_ = lean_ctor_get(v___y_56_, 5);
v___x_60_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(v_msg_53_, v___y_54_, v___y_55_, v___y_56_, v___y_57_);
v_a_61_ = lean_ctor_get(v___x_60_, 0);
v_isSharedCheck_69_ = !lean_is_exclusive(v___x_60_);
if (v_isSharedCheck_69_ == 0)
{
v___x_63_ = v___x_60_;
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_a_61_);
lean_dec(v___x_60_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___x_65_; lean_object* v___x_67_; 
lean_inc(v_ref_59_);
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v_ref_59_);
lean_ctor_set(v___x_65_, 1, v_a_61_);
if (v_isShared_64_ == 0)
{
lean_ctor_set_tag(v___x_63_, 1);
lean_ctor_set(v___x_63_, 0, v___x_65_);
v___x_67_ = v___x_63_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v___x_65_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg___boxed(lean_object* v_msg_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(v_msg_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
return v_res_76_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0(lean_object* v_a_77_, lean_object* v_as_78_, size_t v_i_79_, size_t v_stop_80_){
_start:
{
uint8_t v___x_81_; 
v___x_81_ = lean_usize_dec_eq(v_i_79_, v_stop_80_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_82_ = lean_array_uget_borrowed(v_as_78_, v_i_79_);
v___x_83_ = l_Lean_instBEqFVarId_beq(v_a_77_, v___x_82_);
if (v___x_83_ == 0)
{
size_t v___x_84_; size_t v___x_85_; 
v___x_84_ = ((size_t)1ULL);
v___x_85_ = lean_usize_add(v_i_79_, v___x_84_);
v_i_79_ = v___x_85_;
goto _start;
}
else
{
return v___x_83_;
}
}
else
{
uint8_t v___x_87_; 
v___x_87_ = 0;
return v___x_87_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0___boxed(lean_object* v_a_88_, lean_object* v_as_89_, lean_object* v_i_90_, lean_object* v_stop_91_){
_start:
{
size_t v_i_boxed_92_; size_t v_stop_boxed_93_; uint8_t v_res_94_; lean_object* v_r_95_; 
v_i_boxed_92_ = lean_unbox_usize(v_i_90_);
lean_dec(v_i_90_);
v_stop_boxed_93_ = lean_unbox_usize(v_stop_91_);
lean_dec(v_stop_91_);
v_res_94_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0(v_a_88_, v_as_89_, v_i_boxed_92_, v_stop_boxed_93_);
lean_dec_ref(v_as_89_);
lean_dec(v_a_88_);
v_r_95_ = lean_box(v_res_94_);
return v_r_95_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0(lean_object* v_as_96_, lean_object* v_a_97_){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; uint8_t v___x_100_; 
v___x_98_ = lean_unsigned_to_nat(0u);
v___x_99_ = lean_array_get_size(v_as_96_);
v___x_100_ = lean_nat_dec_lt(v___x_98_, v___x_99_);
if (v___x_100_ == 0)
{
return v___x_100_;
}
else
{
if (v___x_100_ == 0)
{
return v___x_100_;
}
else
{
size_t v___x_101_; size_t v___x_102_; uint8_t v___x_103_; 
v___x_101_ = ((size_t)0ULL);
v___x_102_ = lean_usize_of_nat(v___x_99_);
v___x_103_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0_spec__0(v_a_97_, v_as_96_, v___x_101_, v___x_102_);
return v___x_103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0___boxed(lean_object* v_as_104_, lean_object* v_a_105_){
_start:
{
uint8_t v_res_106_; lean_object* v_r_107_; 
v_res_106_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0(v_as_104_, v_a_105_);
lean_dec(v_a_105_);
lean_dec_ref(v_as_104_);
v_r_107_ = lean_box(v_res_106_);
return v_r_107_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__0));
v___x_110_ = l_Lean_stringToMessageData(v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1(lean_object* v_a_111_, lean_object* v_atLocal_112_, lean_object* v_fvarId_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
uint8_t v___x_123_; 
v___x_123_ = lp_mathlib_Array_contains___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__0(v_a_111_, v_fvarId_113_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v_fvarId_113_);
lean_dec_ref(v_atLocal_112_);
v___x_124_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1, &lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___closed__1);
v___x_125_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(v___x_124_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
return v___x_125_;
}
else
{
lean_object* v___x_126_; 
lean_inc(v___y_121_);
lean_inc_ref(v___y_120_);
lean_inc(v___y_119_);
lean_inc_ref(v___y_118_);
lean_inc(v___y_117_);
lean_inc_ref(v___y_116_);
lean_inc(v___y_115_);
lean_inc_ref(v___y_114_);
v___x_126_ = lean_apply_10(v_atLocal_112_, v_fvarId_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_, lean_box(0));
return v___x_126_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___boxed(lean_object* v_a_127_, lean_object* v_atLocal_128_, lean_object* v_fvarId_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1(v_a_127_, v_atLocal_128_, v_fvarId_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
lean_dec_ref(v_a_127_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation(lean_object* v_loc_141_, lean_object* v_atLocal_142_, lean_object* v_atTarget_143_, lean_object* v_failed_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
if (lean_obj_tag(v_loc_141_) == 0)
{
lean_object* v___f_154_; lean_object* v___x_155_; 
v___f_154_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___closed__0));
v___x_155_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_154_, v_a_145_, v_a_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v_a_156_; lean_object* v___f_157_; lean_object* v___x_158_; 
v_a_156_ = lean_ctor_get(v___x_155_, 0);
lean_inc(v_a_156_);
lean_dec_ref_known(v___x_155_, 1);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___lam__1___boxed), 12, 2);
lean_closure_set(v___f_157_, 0, v_a_156_);
lean_closure_set(v___f_157_, 1, v_atLocal_142_);
v___x_158_ = l_Lean_Elab_Tactic_withLocation(v_loc_141_, v___f_157_, v_atTarget_143_, v_failed_144_, v_a_145_, v_a_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
return v___x_158_;
}
else
{
lean_object* v_a_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_166_; 
lean_dec_ref(v_failed_144_);
lean_dec_ref(v_atTarget_143_);
lean_dec_ref(v_atLocal_142_);
v_a_159_ = lean_ctor_get(v___x_155_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_166_ == 0)
{
v___x_161_ = v___x_155_;
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_a_159_);
lean_dec(v___x_155_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_164_; 
if (v_isShared_162_ == 0)
{
v___x_164_ = v___x_161_;
goto v_reusejp_163_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_a_159_);
v___x_164_ = v_reuseFailAlloc_165_;
goto v_reusejp_163_;
}
v_reusejp_163_:
{
return v___x_164_;
}
}
}
}
else
{
lean_object* v___x_167_; 
v___x_167_ = l_Lean_Elab_Tactic_withLocation(v_loc_141_, v_atLocal_142_, v_atTarget_143_, v_failed_144_, v_a_145_, v_a_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
return v___x_167_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation___boxed(lean_object* v_loc_168_, lean_object* v_atLocal_169_, lean_object* v_atTarget_170_, lean_object* v_failed_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation(v_loc_168_, v_atLocal_169_, v_atTarget_170_, v_failed_171_, v_a_172_, v_a_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_, v_a_178_, v_a_179_);
lean_dec(v_a_179_);
lean_dec_ref(v_a_178_);
lean_dec(v_a_177_);
lean_dec_ref(v_a_176_);
lean_dec(v_a_175_);
lean_dec_ref(v_a_174_);
lean_dec(v_a_173_);
lean_dec_ref(v_a_172_);
lean_dec(v_loc_168_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1(lean_object* v_00_u03b1_182_, lean_object* v_msg_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(v_msg_183_, v___y_188_, v___y_189_, v___y_190_, v___y_191_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___boxed(lean_object* v_00_u03b1_194_, lean_object* v_msg_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1(v_00_u03b1_194_, v_msg_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0(lean_object* v_x_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_box(0);
v___x_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_217_, 0, v___x_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0___boxed(lean_object* v_x_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__0(v_x_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec(v___y_224_);
lean_dec_ref(v___y_223_);
lean_dec(v___y_222_);
lean_dec_ref(v___y_221_);
lean_dec(v___y_220_);
lean_dec_ref(v___y_219_);
lean_dec(v_x_218_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1(lean_object* v_atLocal_229_, lean_object* v_val_230_, lean_object* v_fvarId_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v___x_241_; 
lean_inc(v___y_239_);
lean_inc_ref(v___y_238_);
lean_inc(v___y_237_);
lean_inc_ref(v___y_236_);
lean_inc(v___y_235_);
lean_inc_ref(v___y_234_);
lean_inc(v___y_233_);
lean_inc_ref(v___y_232_);
v___x_241_ = lean_apply_10(v_atLocal_229_, v_fvarId_231_, v___y_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_, lean_box(0));
if (lean_obj_tag(v___x_241_) == 0)
{
lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_252_; 
v_a_242_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_252_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_252_ == 0)
{
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_252_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_252_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_250_; 
v___x_246_ = lean_st_ref_take(v_val_230_);
v___x_247_ = lean_array_push(v___x_246_, v_a_242_);
v___x_248_ = lean_st_ref_set(v_val_230_, v___x_247_);
if (v_isShared_245_ == 0)
{
lean_ctor_set(v___x_244_, 0, v___x_248_);
v___x_250_ = v___x_244_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v___x_248_);
v___x_250_ = v_reuseFailAlloc_251_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
return v___x_250_;
}
}
}
else
{
lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_260_; 
v_a_253_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_260_ == 0)
{
v___x_255_ = v___x_241_;
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_241_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_258_; 
if (v_isShared_256_ == 0)
{
v___x_258_ = v___x_255_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v_a_253_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1___boxed(lean_object* v_atLocal_261_, lean_object* v_val_262_, lean_object* v_fvarId_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1(v_atLocal_261_, v_val_262_, v_fvarId_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v_val_262_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2(lean_object* v_atTarget_274_, lean_object* v_val_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_apply_9(v_atTarget_274_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, lean_box(0));
if (lean_obj_tag(v___x_285_) == 0)
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_296_; 
v_a_286_ = lean_ctor_get(v___x_285_, 0);
v_isSharedCheck_296_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_296_ == 0)
{
v___x_288_ = v___x_285_;
v_isShared_289_ = v_isSharedCheck_296_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_285_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_296_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_294_; 
v___x_290_ = lean_st_ref_take(v_val_275_);
v___x_291_ = lean_array_push(v___x_290_, v_a_286_);
v___x_292_ = lean_st_ref_set(v_val_275_, v___x_291_);
if (v_isShared_289_ == 0)
{
lean_ctor_set(v___x_288_, 0, v___x_292_);
v___x_294_ = v___x_288_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v___x_292_);
v___x_294_ = v_reuseFailAlloc_295_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
return v___x_294_;
}
}
}
else
{
lean_object* v_a_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_304_; 
v_a_297_ = lean_ctor_get(v___x_285_, 0);
v_isSharedCheck_304_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_304_ == 0)
{
v___x_299_ = v___x_285_;
v_isShared_300_ = v_isSharedCheck_304_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_a_297_);
lean_dec(v___x_285_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_304_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
lean_object* v___x_302_; 
if (v_isShared_300_ == 0)
{
v___x_302_ = v___x_299_;
goto v_reusejp_301_;
}
else
{
lean_object* v_reuseFailAlloc_303_; 
v_reuseFailAlloc_303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_303_, 0, v_a_297_);
v___x_302_ = v_reuseFailAlloc_303_;
goto v_reusejp_301_;
}
v_reusejp_301_:
{
return v___x_302_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2___boxed(lean_object* v_atTarget_305_, lean_object* v_val_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2(v_atTarget_305_, v_val_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
lean_dec(v_val_306_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(lean_object* v_loc_320_, lean_object* v_atLocal_321_, lean_object* v_atTarget_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___f_334_; lean_object* v___f_335_; lean_object* v___f_336_; lean_object* v___x_337_; 
v___x_332_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__0));
v___x_333_ = lean_st_mk_ref(v___x_332_);
v___f_334_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___closed__1));
lean_inc_n(v___x_333_, 2);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__1___boxed), 12, 2);
lean_closure_set(v___f_335_, 0, v_atLocal_321_);
lean_closure_set(v___f_335_, 1, v___x_333_);
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___lam__2___boxed), 11, 2);
lean_closure_set(v___f_336_, 0, v_atTarget_322_);
lean_closure_set(v___f_336_, 1, v___x_333_);
v___x_337_ = lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation(v_loc_320_, v___f_335_, v___f_336_, v___f_334_, v_a_323_, v_a_324_, v_a_325_, v_a_326_, v_a_327_, v_a_328_, v_a_329_, v_a_330_);
if (lean_obj_tag(v___x_337_) == 0)
{
lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_345_; 
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_337_);
if (v_isSharedCheck_345_ == 0)
{
lean_object* v_unused_346_; 
v_unused_346_ = lean_ctor_get(v___x_337_, 0);
lean_dec(v_unused_346_);
v___x_339_ = v___x_337_;
v_isShared_340_ = v_isSharedCheck_345_;
goto v_resetjp_338_;
}
else
{
lean_dec(v___x_337_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_345_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_341_; lean_object* v___x_343_; 
v___x_341_ = lean_st_ref_get(v___x_333_);
lean_dec(v___x_333_);
if (v_isShared_340_ == 0)
{
lean_ctor_set(v___x_339_, 0, v___x_341_);
v___x_343_ = v___x_339_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v___x_341_);
v___x_343_ = v_reuseFailAlloc_344_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
return v___x_343_;
}
}
}
else
{
lean_object* v_a_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_354_; 
lean_dec(v___x_333_);
v_a_347_ = lean_ctor_get(v___x_337_, 0);
v_isSharedCheck_354_ = !lean_is_exclusive(v___x_337_);
if (v_isSharedCheck_354_ == 0)
{
v___x_349_ = v___x_337_;
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_a_347_);
lean_dec(v___x_337_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_352_; 
if (v_isShared_350_ == 0)
{
v___x_352_ = v___x_349_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_a_347_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg___boxed(lean_object* v_loc_355_, lean_object* v_atLocal_356_, lean_object* v_atTarget_357_, lean_object* v_a_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(v_loc_355_, v_atLocal_356_, v_atTarget_357_, v_a_358_, v_a_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_, v_a_364_, v_a_365_);
lean_dec(v_a_365_);
lean_dec_ref(v_a_364_);
lean_dec(v_a_363_);
lean_dec_ref(v_a_362_);
lean_dec(v_a_361_);
lean_dec_ref(v_a_360_);
lean_dec(v_a_359_);
lean_dec_ref(v_a_358_);
lean_dec(v_loc_355_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation(lean_object* v_00_u03b1_368_, lean_object* v_loc_369_, lean_object* v_atLocal_370_, lean_object* v_atTarget_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(v_loc_369_, v_atLocal_370_, v_atTarget_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_, v_a_378_, v_a_379_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___boxed(lean_object* v_00_u03b1_382_, lean_object* v_loc_383_, lean_object* v_atLocal_384_, lean_object* v_atTarget_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation(v_00_u03b1_382_, v_loc_383_, v_atLocal_384_, v_atTarget_385_, v_a_386_, v_a_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_);
lean_dec(v_a_393_);
lean_dec_ref(v_a_392_);
lean_dec(v_a_391_);
lean_dec_ref(v_a_390_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
lean_dec(v_a_387_);
lean_dec_ref(v_a_386_);
lean_dec(v_loc_383_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx(uint8_t v_x_396_){
_start:
{
switch(v_x_396_)
{
case 0:
{
lean_object* v___x_397_; 
v___x_397_ = lean_unsigned_to_nat(0u);
return v___x_397_;
}
case 1:
{
lean_object* v___x_398_; 
v___x_398_ = lean_unsigned_to_nat(1u);
return v___x_398_;
}
default: 
{
lean_object* v___x_399_; 
v___x_399_ = lean_unsigned_to_nat(2u);
return v___x_399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx___boxed(lean_object* v_x_400_){
_start:
{
uint8_t v_x_boxed_401_; lean_object* v_res_402_; 
v_x_boxed_401_ = lean_unbox(v_x_400_);
v_res_402_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx(v_x_boxed_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___redArg(lean_object* v_k_403_){
_start:
{
lean_inc(v_k_403_);
return v_k_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___redArg___boxed(lean_object* v_k_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___redArg(v_k_404_);
lean_dec(v_k_404_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim(lean_object* v_motive_406_, lean_object* v_ctorIdx_407_, uint8_t v_t_408_, lean_object* v_h_409_, lean_object* v_k_410_){
_start:
{
lean_inc(v_k_410_);
return v_k_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim___boxed(lean_object* v_motive_411_, lean_object* v_ctorIdx_412_, lean_object* v_t_413_, lean_object* v_h_414_, lean_object* v_k_415_){
_start:
{
uint8_t v_t_boxed_416_; lean_object* v_res_417_; 
v_t_boxed_416_ = lean_unbox(v_t_413_);
v_res_417_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorElim(v_motive_411_, v_ctorIdx_412_, v_t_boxed_416_, v_h_414_, v_k_415_);
lean_dec(v_k_415_);
lean_dec(v_ctorIdx_412_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___redArg(lean_object* v_silent_418_){
_start:
{
lean_inc(v_silent_418_);
return v_silent_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___redArg___boxed(lean_object* v_silent_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___redArg(v_silent_419_);
lean_dec(v_silent_419_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim(lean_object* v_motive_421_, uint8_t v_t_422_, lean_object* v_h_423_, lean_object* v_silent_424_){
_start:
{
lean_inc(v_silent_424_);
return v_silent_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim___boxed(lean_object* v_motive_425_, lean_object* v_t_426_, lean_object* v_h_427_, lean_object* v_silent_428_){
_start:
{
uint8_t v_t_boxed_429_; lean_object* v_res_430_; 
v_t_boxed_429_ = lean_unbox(v_t_426_);
v_res_430_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_silent_elim(v_motive_425_, v_t_boxed_429_, v_h_427_, v_silent_428_);
lean_dec(v_silent_428_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___redArg(lean_object* v_warning_431_){
_start:
{
lean_inc(v_warning_431_);
return v_warning_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___redArg___boxed(lean_object* v_warning_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___redArg(v_warning_432_);
lean_dec(v_warning_432_);
return v_res_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim(lean_object* v_motive_434_, uint8_t v_t_435_, lean_object* v_h_436_, lean_object* v_warning_437_){
_start:
{
lean_inc(v_warning_437_);
return v_warning_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim___boxed(lean_object* v_motive_438_, lean_object* v_t_439_, lean_object* v_h_440_, lean_object* v_warning_441_){
_start:
{
uint8_t v_t_boxed_442_; lean_object* v_res_443_; 
v_t_boxed_442_ = lean_unbox(v_t_439_);
v_res_443_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_warning_elim(v_motive_438_, v_t_boxed_442_, v_h_440_, v_warning_441_);
lean_dec(v_warning_441_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___redArg(lean_object* v_error_444_){
_start:
{
lean_inc(v_error_444_);
return v_error_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___redArg___boxed(lean_object* v_error_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___redArg(v_error_445_);
lean_dec(v_error_445_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim(lean_object* v_motive_447_, uint8_t v_t_448_, lean_object* v_h_449_, lean_object* v_error_450_){
_start:
{
lean_inc(v_error_450_);
return v_error_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim___boxed(lean_object* v_motive_451_, lean_object* v_t_452_, lean_object* v_h_453_, lean_object* v_error_454_){
_start:
{
uint8_t v_t_boxed_455_; lean_object* v_res_456_; 
v_t_boxed_455_ = lean_unbox(v_t_452_);
v_res_456_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_error_elim(v_motive_451_, v_t_boxed_455_, v_h_453_, v_error_454_);
lean_dec(v_error_454_);
return v_res_456_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq(uint8_t v_x_457_, uint8_t v_y_458_){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; uint8_t v___x_461_; 
v___x_459_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx(v_x_457_);
v___x_460_ = lp_mathlib_Mathlib_Tactic_BehaviorIfUnchanged_ctorIdx(v_y_458_);
v___x_461_ = lean_nat_dec_eq(v___x_459_, v___x_460_);
lean_dec(v___x_460_);
lean_dec(v___x_459_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq___boxed(lean_object* v_x_462_, lean_object* v_y_463_){
_start:
{
uint8_t v_x_17__boxed_464_; uint8_t v_y_18__boxed_465_; uint8_t v_res_466_; lean_object* v_r_467_; 
v_x_17__boxed_464_ = lean_unbox(v_x_462_);
v_y_18__boxed_465_ = lean_unbox(v_y_463_);
v_res_466_ = lp_mathlib_Mathlib_Tactic_instBEqBehaviorIfUnchanged_beq(v_x_17__boxed_464_, v_y_18__boxed_465_);
v_r_467_ = lean_box(v_res_466_);
return v_r_467_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged_default(void){
_start:
{
uint8_t v___x_470_; 
v___x_470_ = 0;
return v___x_470_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged(void){
_start:
{
uint8_t v___x_471_; 
v___x_471_ = 0;
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6(void){
_start:
{
lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_481_ = lean_unsigned_to_nat(2u);
v___x_482_ = lean_nat_to_int(v___x_481_);
return v___x_482_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = lean_unsigned_to_nat(1u);
v___x_484_ = lean_nat_to_int(v___x_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr(uint8_t v_x_485_, lean_object* v_prec_486_){
_start:
{
lean_object* v___y_488_; lean_object* v___y_495_; lean_object* v___y_502_; 
switch(v_x_485_)
{
case 0:
{
lean_object* v___x_508_; uint8_t v___x_509_; 
v___x_508_ = lean_unsigned_to_nat(1024u);
v___x_509_ = lean_nat_dec_le(v___x_508_, v_prec_486_);
if (v___x_509_ == 0)
{
lean_object* v___x_510_; 
v___x_510_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6);
v___y_488_ = v___x_510_;
goto v___jp_487_;
}
else
{
lean_object* v___x_511_; 
v___x_511_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7);
v___y_488_ = v___x_511_;
goto v___jp_487_;
}
}
case 1:
{
lean_object* v___x_512_; uint8_t v___x_513_; 
v___x_512_ = lean_unsigned_to_nat(1024u);
v___x_513_ = lean_nat_dec_le(v___x_512_, v_prec_486_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; 
v___x_514_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6);
v___y_495_ = v___x_514_;
goto v___jp_494_;
}
else
{
lean_object* v___x_515_; 
v___x_515_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7);
v___y_495_ = v___x_515_;
goto v___jp_494_;
}
}
default: 
{
lean_object* v___x_516_; uint8_t v___x_517_; 
v___x_516_ = lean_unsigned_to_nat(1024u);
v___x_517_ = lean_nat_dec_le(v___x_516_, v_prec_486_);
if (v___x_517_ == 0)
{
lean_object* v___x_518_; 
v___x_518_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__6);
v___y_502_ = v___x_518_;
goto v___jp_501_;
}
else
{
lean_object* v___x_519_; 
v___x_519_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7, &lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__7);
v___y_502_ = v___x_519_;
goto v___jp_501_;
}
}
}
v___jp_487_:
{
lean_object* v___x_489_; lean_object* v___x_490_; uint8_t v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_489_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__1));
lean_inc(v___y_488_);
v___x_490_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_490_, 0, v___y_488_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = 0;
v___x_492_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_492_, 0, v___x_490_);
lean_ctor_set_uint8(v___x_492_, sizeof(void*)*1, v___x_491_);
v___x_493_ = l_Repr_addAppParen(v___x_492_, v_prec_486_);
return v___x_493_;
}
v___jp_494_:
{
lean_object* v___x_496_; lean_object* v___x_497_; uint8_t v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__3));
lean_inc(v___y_495_);
v___x_497_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_497_, 0, v___y_495_);
lean_ctor_set(v___x_497_, 1, v___x_496_);
v___x_498_ = 0;
v___x_499_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_499_, 0, v___x_497_);
lean_ctor_set_uint8(v___x_499_, sizeof(void*)*1, v___x_498_);
v___x_500_ = l_Repr_addAppParen(v___x_499_, v_prec_486_);
return v___x_500_;
}
v___jp_501_:
{
lean_object* v___x_503_; lean_object* v___x_504_; uint8_t v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___closed__5));
lean_inc(v___y_502_);
v___x_504_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_504_, 0, v___y_502_);
lean_ctor_set(v___x_504_, 1, v___x_503_);
v___x_505_ = 0;
v___x_506_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_506_, 0, v___x_504_);
lean_ctor_set_uint8(v___x_506_, sizeof(void*)*1, v___x_505_);
v___x_507_ = l_Repr_addAppParen(v___x_506_, v_prec_486_);
return v___x_507_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr___boxed(lean_object* v_x_520_, lean_object* v_prec_521_){
_start:
{
uint8_t v_x_177__boxed_522_; lean_object* v_res_523_; 
v_x_177__boxed_522_ = lean_unbox(v_x_520_);
v_res_523_ = lp_mathlib_Mathlib_Tactic_instReprBehaviorIfUnchanged_repr(v_x_177__boxed_522_, v_prec_521_);
lean_dec(v_prec_521_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(lean_object* v_e_526_, lean_object* v___y_527_){
_start:
{
uint8_t v___x_529_; 
v___x_529_ = l_Lean_Expr_hasMVar(v_e_526_);
if (v___x_529_ == 0)
{
lean_object* v___x_530_; 
v___x_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_530_, 0, v_e_526_);
return v___x_530_;
}
else
{
lean_object* v___x_531_; lean_object* v_mctx_532_; lean_object* v___x_533_; lean_object* v_fst_534_; lean_object* v_snd_535_; lean_object* v___x_536_; lean_object* v_cache_537_; lean_object* v_zetaDeltaFVarIds_538_; lean_object* v_postponed_539_; lean_object* v_diag_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_549_; 
v___x_531_ = lean_st_ref_get(v___y_527_);
v_mctx_532_ = lean_ctor_get(v___x_531_, 0);
lean_inc_ref(v_mctx_532_);
lean_dec(v___x_531_);
v___x_533_ = l_Lean_instantiateMVarsCore(v_mctx_532_, v_e_526_);
v_fst_534_ = lean_ctor_get(v___x_533_, 0);
lean_inc(v_fst_534_);
v_snd_535_ = lean_ctor_get(v___x_533_, 1);
lean_inc(v_snd_535_);
lean_dec_ref(v___x_533_);
v___x_536_ = lean_st_ref_take(v___y_527_);
v_cache_537_ = lean_ctor_get(v___x_536_, 1);
v_zetaDeltaFVarIds_538_ = lean_ctor_get(v___x_536_, 2);
v_postponed_539_ = lean_ctor_get(v___x_536_, 3);
v_diag_540_ = lean_ctor_get(v___x_536_, 4);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_536_);
if (v_isSharedCheck_549_ == 0)
{
lean_object* v_unused_550_; 
v_unused_550_ = lean_ctor_get(v___x_536_, 0);
lean_dec(v_unused_550_);
v___x_542_ = v___x_536_;
v_isShared_543_ = v_isSharedCheck_549_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_diag_540_);
lean_inc(v_postponed_539_);
lean_inc(v_zetaDeltaFVarIds_538_);
lean_inc(v_cache_537_);
lean_dec(v___x_536_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_549_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_545_; 
if (v_isShared_543_ == 0)
{
lean_ctor_set(v___x_542_, 0, v_snd_535_);
v___x_545_ = v___x_542_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v_snd_535_);
lean_ctor_set(v_reuseFailAlloc_548_, 1, v_cache_537_);
lean_ctor_set(v_reuseFailAlloc_548_, 2, v_zetaDeltaFVarIds_538_);
lean_ctor_set(v_reuseFailAlloc_548_, 3, v_postponed_539_);
lean_ctor_set(v_reuseFailAlloc_548_, 4, v_diag_540_);
v___x_545_ = v_reuseFailAlloc_548_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_546_ = lean_st_ref_set(v___y_527_, v___x_545_);
v___x_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_547_, 0, v_fst_534_);
return v___x_547_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg___boxed(lean_object* v_e_551_, lean_object* v___y_552_, lean_object* v___y_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(v_e_551_, v___y_552_);
lean_dec(v___y_552_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0(lean_object* v_e_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_){
_start:
{
lean_object* v___x_562_; 
v___x_562_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(v_e_555_, v___y_558_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___boxed(lean_object* v_e_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0(v_e_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec_ref(v___y_564_);
return v_res_570_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9(lean_object* v_opts_571_, lean_object* v_opt_572_){
_start:
{
lean_object* v_name_573_; lean_object* v_defValue_574_; lean_object* v_map_575_; lean_object* v___x_576_; 
v_name_573_ = lean_ctor_get(v_opt_572_, 0);
v_defValue_574_ = lean_ctor_get(v_opt_572_, 1);
v_map_575_ = lean_ctor_get(v_opts_571_, 0);
v___x_576_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_575_, v_name_573_);
if (lean_obj_tag(v___x_576_) == 0)
{
uint8_t v___x_577_; 
v___x_577_ = lean_unbox(v_defValue_574_);
return v___x_577_;
}
else
{
lean_object* v_val_578_; 
v_val_578_ = lean_ctor_get(v___x_576_, 0);
lean_inc(v_val_578_);
lean_dec_ref_known(v___x_576_, 1);
if (lean_obj_tag(v_val_578_) == 1)
{
uint8_t v_v_579_; 
v_v_579_ = lean_ctor_get_uint8(v_val_578_, 0);
lean_dec_ref_known(v_val_578_, 0);
return v_v_579_;
}
else
{
uint8_t v___x_580_; 
lean_dec(v_val_578_);
v___x_580_ = lean_unbox(v_defValue_574_);
return v___x_580_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9___boxed(lean_object* v_opts_581_, lean_object* v_opt_582_){
_start:
{
uint8_t v_res_583_; lean_object* v_r_584_; 
v_res_583_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9(v_opts_581_, v_opt_582_);
lean_dec_ref(v_opt_582_);
lean_dec_ref(v_opts_581_);
v_r_584_ = lean_box(v_res_583_);
return v_r_584_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0(uint8_t v___y_593_, uint8_t v_suppressElabErrors_594_, lean_object* v_x_595_){
_start:
{
if (lean_obj_tag(v_x_595_) == 1)
{
lean_object* v_pre_596_; 
v_pre_596_ = lean_ctor_get(v_x_595_, 0);
switch(lean_obj_tag(v_pre_596_))
{
case 1:
{
lean_object* v_pre_597_; 
v_pre_597_ = lean_ctor_get(v_pre_596_, 0);
switch(lean_obj_tag(v_pre_597_))
{
case 0:
{
lean_object* v_str_598_; lean_object* v_str_599_; lean_object* v___x_600_; uint8_t v___x_601_; 
v_str_598_ = lean_ctor_get(v_x_595_, 1);
v_str_599_ = lean_ctor_get(v_pre_596_, 1);
v___x_600_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__0));
v___x_601_ = lean_string_dec_eq(v_str_599_, v___x_600_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; uint8_t v___x_603_; 
v___x_602_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__1));
v___x_603_ = lean_string_dec_eq(v_str_599_, v___x_602_);
if (v___x_603_ == 0)
{
return v___y_593_;
}
else
{
lean_object* v___x_604_; uint8_t v___x_605_; 
v___x_604_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__2));
v___x_605_ = lean_string_dec_eq(v_str_598_, v___x_604_);
if (v___x_605_ == 0)
{
return v___y_593_;
}
else
{
return v_suppressElabErrors_594_;
}
}
}
else
{
lean_object* v___x_606_; uint8_t v___x_607_; 
v___x_606_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__3));
v___x_607_ = lean_string_dec_eq(v_str_598_, v___x_606_);
if (v___x_607_ == 0)
{
return v___y_593_;
}
else
{
return v_suppressElabErrors_594_;
}
}
}
case 1:
{
lean_object* v_pre_608_; 
v_pre_608_ = lean_ctor_get(v_pre_597_, 0);
if (lean_obj_tag(v_pre_608_) == 0)
{
lean_object* v_str_609_; lean_object* v_str_610_; lean_object* v_str_611_; lean_object* v___x_612_; uint8_t v___x_613_; 
v_str_609_ = lean_ctor_get(v_x_595_, 1);
v_str_610_ = lean_ctor_get(v_pre_596_, 1);
v_str_611_ = lean_ctor_get(v_pre_597_, 1);
v___x_612_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__4));
v___x_613_ = lean_string_dec_eq(v_str_611_, v___x_612_);
if (v___x_613_ == 0)
{
return v___y_593_;
}
else
{
lean_object* v___x_614_; uint8_t v___x_615_; 
v___x_614_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__5));
v___x_615_ = lean_string_dec_eq(v_str_610_, v___x_614_);
if (v___x_615_ == 0)
{
return v___y_593_;
}
else
{
lean_object* v___x_616_; uint8_t v___x_617_; 
v___x_616_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__6));
v___x_617_ = lean_string_dec_eq(v_str_609_, v___x_616_);
if (v___x_617_ == 0)
{
return v___y_593_;
}
else
{
return v_suppressElabErrors_594_;
}
}
}
}
else
{
return v___y_593_;
}
}
default: 
{
return v___y_593_;
}
}
}
case 0:
{
lean_object* v_str_618_; lean_object* v___x_619_; uint8_t v___x_620_; 
v_str_618_ = lean_ctor_get(v_x_595_, 1);
v___x_619_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___closed__7));
v___x_620_ = lean_string_dec_eq(v_str_618_, v___x_619_);
if (v___x_620_ == 0)
{
return v___y_593_;
}
else
{
return v_suppressElabErrors_594_;
}
}
default: 
{
return v___y_593_;
}
}
}
else
{
return v___y_593_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v___y_621_, lean_object* v_suppressElabErrors_622_, lean_object* v_x_623_){
_start:
{
uint8_t v___y_9891__boxed_624_; uint8_t v_suppressElabErrors_boxed_625_; uint8_t v_res_626_; lean_object* v_r_627_; 
v___y_9891__boxed_624_ = lean_unbox(v___y_621_);
v_suppressElabErrors_boxed_625_ = lean_unbox(v_suppressElabErrors_622_);
v_res_626_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0(v___y_9891__boxed_624_, v_suppressElabErrors_boxed_625_, v_x_623_);
lean_dec(v_x_623_);
v_r_627_ = lean_box(v_res_626_);
return v_r_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg(lean_object* v_ref_629_, lean_object* v_msgData_630_, uint8_t v_severity_631_, uint8_t v_isSilent_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; uint8_t v___y_644_; uint8_t v___y_645_; lean_object* v___y_646_; lean_object* v___y_647_; lean_object* v___y_675_; lean_object* v___y_676_; lean_object* v___y_677_; uint8_t v___y_678_; lean_object* v___y_679_; uint8_t v___y_680_; uint8_t v___y_681_; lean_object* v___y_682_; lean_object* v___y_700_; lean_object* v___y_701_; lean_object* v___y_702_; uint8_t v___y_703_; lean_object* v___y_704_; uint8_t v___y_705_; uint8_t v___y_706_; lean_object* v___y_707_; lean_object* v___y_711_; lean_object* v___y_712_; uint8_t v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; uint8_t v___y_716_; uint8_t v___y_717_; uint8_t v___x_722_; lean_object* v___y_724_; uint8_t v___y_725_; lean_object* v___y_726_; lean_object* v___y_727_; lean_object* v___y_728_; uint8_t v___y_729_; uint8_t v___y_730_; uint8_t v___y_732_; uint8_t v___x_747_; 
v___x_722_ = 2;
v___x_747_ = l_Lean_instBEqMessageSeverity_beq(v_severity_631_, v___x_722_);
if (v___x_747_ == 0)
{
v___y_732_ = v___x_747_;
goto v___jp_731_;
}
else
{
uint8_t v___x_748_; 
lean_inc_ref(v_msgData_630_);
v___x_748_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_630_);
v___y_732_ = v___x_748_;
goto v___jp_731_;
}
v___jp_638_:
{
lean_object* v___x_648_; lean_object* v_currNamespace_649_; lean_object* v_openDecls_650_; lean_object* v_env_651_; lean_object* v_nextMacroScope_652_; lean_object* v_ngen_653_; lean_object* v_auxDeclNGen_654_; lean_object* v_traceState_655_; lean_object* v_cache_656_; lean_object* v_messages_657_; lean_object* v_infoState_658_; lean_object* v_snapshotTasks_659_; lean_object* v___x_661_; uint8_t v_isShared_662_; uint8_t v_isSharedCheck_673_; 
v___x_648_ = lean_st_ref_take(v___y_647_);
v_currNamespace_649_ = lean_ctor_get(v___y_646_, 6);
v_openDecls_650_ = lean_ctor_get(v___y_646_, 7);
v_env_651_ = lean_ctor_get(v___x_648_, 0);
v_nextMacroScope_652_ = lean_ctor_get(v___x_648_, 1);
v_ngen_653_ = lean_ctor_get(v___x_648_, 2);
v_auxDeclNGen_654_ = lean_ctor_get(v___x_648_, 3);
v_traceState_655_ = lean_ctor_get(v___x_648_, 4);
v_cache_656_ = lean_ctor_get(v___x_648_, 5);
v_messages_657_ = lean_ctor_get(v___x_648_, 6);
v_infoState_658_ = lean_ctor_get(v___x_648_, 7);
v_snapshotTasks_659_ = lean_ctor_get(v___x_648_, 8);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_673_ == 0)
{
v___x_661_ = v___x_648_;
v_isShared_662_ = v_isSharedCheck_673_;
goto v_resetjp_660_;
}
else
{
lean_inc(v_snapshotTasks_659_);
lean_inc(v_infoState_658_);
lean_inc(v_messages_657_);
lean_inc(v_cache_656_);
lean_inc(v_traceState_655_);
lean_inc(v_auxDeclNGen_654_);
lean_inc(v_ngen_653_);
lean_inc(v_nextMacroScope_652_);
lean_inc(v_env_651_);
lean_dec(v___x_648_);
v___x_661_ = lean_box(0);
v_isShared_662_ = v_isSharedCheck_673_;
goto v_resetjp_660_;
}
v_resetjp_660_:
{
lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_668_; 
lean_inc(v_openDecls_650_);
lean_inc(v_currNamespace_649_);
v___x_663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_663_, 0, v_currNamespace_649_);
lean_ctor_set(v___x_663_, 1, v_openDecls_650_);
v___x_664_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_664_, 0, v___x_663_);
lean_ctor_set(v___x_664_, 1, v___y_643_);
lean_inc_ref(v___y_639_);
lean_inc_ref(v___y_642_);
v___x_665_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_665_, 0, v___y_642_);
lean_ctor_set(v___x_665_, 1, v___y_640_);
lean_ctor_set(v___x_665_, 2, v___y_641_);
lean_ctor_set(v___x_665_, 3, v___y_639_);
lean_ctor_set(v___x_665_, 4, v___x_664_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*5, v___y_645_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*5 + 1, v___y_644_);
lean_ctor_set_uint8(v___x_665_, sizeof(void*)*5 + 2, v_isSilent_632_);
v___x_666_ = l_Lean_MessageLog_add(v___x_665_, v_messages_657_);
if (v_isShared_662_ == 0)
{
lean_ctor_set(v___x_661_, 6, v___x_666_);
v___x_668_ = v___x_661_;
goto v_reusejp_667_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_env_651_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_nextMacroScope_652_);
lean_ctor_set(v_reuseFailAlloc_672_, 2, v_ngen_653_);
lean_ctor_set(v_reuseFailAlloc_672_, 3, v_auxDeclNGen_654_);
lean_ctor_set(v_reuseFailAlloc_672_, 4, v_traceState_655_);
lean_ctor_set(v_reuseFailAlloc_672_, 5, v_cache_656_);
lean_ctor_set(v_reuseFailAlloc_672_, 6, v___x_666_);
lean_ctor_set(v_reuseFailAlloc_672_, 7, v_infoState_658_);
lean_ctor_set(v_reuseFailAlloc_672_, 8, v_snapshotTasks_659_);
v___x_668_ = v_reuseFailAlloc_672_;
goto v_reusejp_667_;
}
v_reusejp_667_:
{
lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_669_ = lean_st_ref_set(v___y_647_, v___x_668_);
v___x_670_ = lean_box(0);
v___x_671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
return v___x_671_;
}
}
}
v___jp_674_:
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_698_; 
v___x_683_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_630_);
v___x_684_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(v___x_683_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
v_a_685_ = lean_ctor_get(v___x_684_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_684_);
if (v_isSharedCheck_698_ == 0)
{
v___x_687_ = v___x_684_;
v_isShared_688_ = v_isSharedCheck_698_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_684_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_698_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; 
lean_inc_ref_n(v___y_676_, 2);
v___x_689_ = l_Lean_FileMap_toPosition(v___y_676_, v___y_677_);
lean_dec(v___y_677_);
v___x_690_ = l_Lean_FileMap_toPosition(v___y_676_, v___y_682_);
lean_dec(v___y_682_);
v___x_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_691_, 0, v___x_690_);
v___x_692_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___closed__0));
if (v___y_678_ == 0)
{
lean_del_object(v___x_687_);
lean_dec_ref(v___y_675_);
v___y_639_ = v___x_692_;
v___y_640_ = v___x_689_;
v___y_641_ = v___x_691_;
v___y_642_ = v___y_679_;
v___y_643_ = v_a_685_;
v___y_644_ = v___y_680_;
v___y_645_ = v___y_681_;
v___y_646_ = v___y_635_;
v___y_647_ = v___y_636_;
goto v___jp_638_;
}
else
{
uint8_t v___x_693_; 
lean_inc(v_a_685_);
v___x_693_ = l_Lean_MessageData_hasTag(v___y_675_, v_a_685_);
if (v___x_693_ == 0)
{
lean_object* v___x_694_; lean_object* v___x_696_; 
lean_dec_ref_known(v___x_691_, 1);
lean_dec_ref(v___x_689_);
lean_dec(v_a_685_);
v___x_694_ = lean_box(0);
if (v_isShared_688_ == 0)
{
lean_ctor_set(v___x_687_, 0, v___x_694_);
v___x_696_ = v___x_687_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_694_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
else
{
lean_del_object(v___x_687_);
v___y_639_ = v___x_692_;
v___y_640_ = v___x_689_;
v___y_641_ = v___x_691_;
v___y_642_ = v___y_679_;
v___y_643_ = v_a_685_;
v___y_644_ = v___y_680_;
v___y_645_ = v___y_681_;
v___y_646_ = v___y_635_;
v___y_647_ = v___y_636_;
goto v___jp_638_;
}
}
}
}
v___jp_699_:
{
lean_object* v___x_708_; 
v___x_708_ = l_Lean_Syntax_getTailPos_x3f(v___y_702_, v___y_706_);
lean_dec(v___y_702_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_inc(v___y_707_);
v___y_675_ = v___y_700_;
v___y_676_ = v___y_701_;
v___y_677_ = v___y_707_;
v___y_678_ = v___y_703_;
v___y_679_ = v___y_704_;
v___y_680_ = v___y_705_;
v___y_681_ = v___y_706_;
v___y_682_ = v___y_707_;
goto v___jp_674_;
}
else
{
lean_object* v_val_709_; 
v_val_709_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_val_709_);
lean_dec_ref_known(v___x_708_, 1);
v___y_675_ = v___y_700_;
v___y_676_ = v___y_701_;
v___y_677_ = v___y_707_;
v___y_678_ = v___y_703_;
v___y_679_ = v___y_704_;
v___y_680_ = v___y_705_;
v___y_681_ = v___y_706_;
v___y_682_ = v_val_709_;
goto v___jp_674_;
}
}
v___jp_710_:
{
lean_object* v_ref_718_; lean_object* v___x_719_; 
v_ref_718_ = l_Lean_replaceRef(v_ref_629_, v___y_715_);
v___x_719_ = l_Lean_Syntax_getPos_x3f(v_ref_718_, v___y_716_);
if (lean_obj_tag(v___x_719_) == 0)
{
lean_object* v___x_720_; 
v___x_720_ = lean_unsigned_to_nat(0u);
v___y_700_ = v___y_711_;
v___y_701_ = v___y_712_;
v___y_702_ = v_ref_718_;
v___y_703_ = v___y_713_;
v___y_704_ = v___y_714_;
v___y_705_ = v___y_717_;
v___y_706_ = v___y_716_;
v___y_707_ = v___x_720_;
goto v___jp_699_;
}
else
{
lean_object* v_val_721_; 
v_val_721_ = lean_ctor_get(v___x_719_, 0);
lean_inc(v_val_721_);
lean_dec_ref_known(v___x_719_, 1);
v___y_700_ = v___y_711_;
v___y_701_ = v___y_712_;
v___y_702_ = v_ref_718_;
v___y_703_ = v___y_713_;
v___y_704_ = v___y_714_;
v___y_705_ = v___y_717_;
v___y_706_ = v___y_716_;
v___y_707_ = v_val_721_;
goto v___jp_699_;
}
}
v___jp_723_:
{
if (v___y_730_ == 0)
{
v___y_711_ = v___y_727_;
v___y_712_ = v___y_724_;
v___y_713_ = v___y_725_;
v___y_714_ = v___y_726_;
v___y_715_ = v___y_728_;
v___y_716_ = v___y_729_;
v___y_717_ = v_severity_631_;
goto v___jp_710_;
}
else
{
v___y_711_ = v___y_727_;
v___y_712_ = v___y_724_;
v___y_713_ = v___y_725_;
v___y_714_ = v___y_726_;
v___y_715_ = v___y_728_;
v___y_716_ = v___y_729_;
v___y_717_ = v___x_722_;
goto v___jp_710_;
}
}
v___jp_731_:
{
if (v___y_732_ == 0)
{
lean_object* v_fileName_733_; lean_object* v_fileMap_734_; lean_object* v_options_735_; lean_object* v_ref_736_; uint8_t v_suppressElabErrors_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___f_740_; uint8_t v___x_741_; uint8_t v___x_742_; 
v_fileName_733_ = lean_ctor_get(v___y_635_, 0);
v_fileMap_734_ = lean_ctor_get(v___y_635_, 1);
v_options_735_ = lean_ctor_get(v___y_635_, 2);
v_ref_736_ = lean_ctor_get(v___y_635_, 5);
v_suppressElabErrors_737_ = lean_ctor_get_uint8(v___y_635_, sizeof(void*)*14 + 1);
v___x_738_ = lean_box(v___y_732_);
v___x_739_ = lean_box(v_suppressElabErrors_737_);
v___f_740_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_740_, 0, v___x_738_);
lean_closure_set(v___f_740_, 1, v___x_739_);
v___x_741_ = 1;
v___x_742_ = l_Lean_instBEqMessageSeverity_beq(v_severity_631_, v___x_741_);
if (v___x_742_ == 0)
{
v___y_724_ = v_fileMap_734_;
v___y_725_ = v_suppressElabErrors_737_;
v___y_726_ = v_fileName_733_;
v___y_727_ = v___f_740_;
v___y_728_ = v_ref_736_;
v___y_729_ = v___y_732_;
v___y_730_ = v___x_742_;
goto v___jp_723_;
}
else
{
lean_object* v___x_743_; uint8_t v___x_744_; 
v___x_743_ = l_Lean_warningAsError;
v___x_744_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5_spec__9(v_options_735_, v___x_743_);
v___y_724_ = v_fileMap_734_;
v___y_725_ = v_suppressElabErrors_737_;
v___y_726_ = v_fileName_733_;
v___y_727_ = v___f_740_;
v___y_728_ = v_ref_736_;
v___y_729_ = v___y_732_;
v___y_730_ = v___x_744_;
goto v___jp_723_;
}
}
else
{
lean_object* v___x_745_; lean_object* v___x_746_; 
lean_dec_ref(v_msgData_630_);
v___x_745_ = lean_box(0);
v___x_746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_746_, 0, v___x_745_);
return v___x_746_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_ref_749_, lean_object* v_msgData_750_, lean_object* v_severity_751_, lean_object* v_isSilent_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_){
_start:
{
uint8_t v_severity_boxed_758_; uint8_t v_isSilent_boxed_759_; lean_object* v_res_760_; 
v_severity_boxed_758_ = lean_unbox(v_severity_751_);
v_isSilent_boxed_759_ = lean_unbox(v_isSilent_752_);
v_res_760_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg(v_ref_749_, v_msgData_750_, v_severity_boxed_758_, v_isSilent_boxed_759_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v_ref_749_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3(lean_object* v_msgData_761_, uint8_t v_severity_762_, uint8_t v_isSilent_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v_ref_770_; lean_object* v___x_771_; 
v_ref_770_ = lean_ctor_get(v___y_767_, 5);
v___x_771_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg(v_ref_770_, v_msgData_761_, v_severity_762_, v_isSilent_763_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
return v___x_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3___boxed(lean_object* v_msgData_772_, lean_object* v_severity_773_, lean_object* v_isSilent_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
uint8_t v_severity_boxed_781_; uint8_t v_isSilent_boxed_782_; lean_object* v_res_783_; 
v_severity_boxed_781_ = lean_unbox(v_severity_773_);
v_isSilent_boxed_782_ = lean_unbox(v_isSilent_774_);
v_res_783_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3(v_msgData_772_, v_severity_boxed_781_, v_isSilent_boxed_782_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
lean_dec_ref(v___y_775_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2(lean_object* v_msgData_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
uint8_t v___x_791_; uint8_t v___x_792_; lean_object* v___x_793_; 
v___x_791_ = 1;
v___x_792_ = 0;
v___x_793_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3(v_msgData_784_, v___x_791_, v___x_792_, v___y_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2___boxed(lean_object* v_msgData_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2(v_msgData_794_, v___y_795_, v___y_796_, v___y_797_, v___y_798_, v___y_799_);
lean_dec(v___y_799_);
lean_dec_ref(v___y_798_);
lean_dec(v___y_797_);
lean_dec_ref(v___y_796_);
lean_dec_ref(v___y_795_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(lean_object* v_msg_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_){
_start:
{
lean_object* v_ref_808_; lean_object* v___x_809_; lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_818_; 
v_ref_808_ = lean_ctor_get(v___y_805_, 5);
v___x_809_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1_spec__2(v_msg_802_, v___y_803_, v___y_804_, v___y_805_, v___y_806_);
v_a_810_ = lean_ctor_get(v___x_809_, 0);
v_isSharedCheck_818_ = !lean_is_exclusive(v___x_809_);
if (v_isSharedCheck_818_ == 0)
{
v___x_812_ = v___x_809_;
v_isShared_813_ = v_isSharedCheck_818_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_809_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_818_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_814_; lean_object* v___x_816_; 
lean_inc(v_ref_808_);
v___x_814_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_814_, 0, v_ref_808_);
lean_ctor_set(v___x_814_, 1, v_a_810_);
if (v_isShared_813_ == 0)
{
lean_ctor_set_tag(v___x_812_, 1);
lean_ctor_set(v___x_812_, 0, v___x_814_);
v___x_816_ = v___x_812_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v___x_814_);
v___x_816_ = v_reuseFailAlloc_817_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
return v___x_816_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg___boxed(lean_object* v_msg_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(v_msg_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_);
lean_dec(v___y_823_);
lean_dec_ref(v___y_822_);
lean_dec(v___y_821_);
lean_dec_ref(v___y_820_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7___redArg(lean_object* v_x_826_, lean_object* v_x_827_, lean_object* v_x_828_, lean_object* v_x_829_){
_start:
{
lean_object* v_ks_830_; lean_object* v_vs_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_855_; 
v_ks_830_ = lean_ctor_get(v_x_826_, 0);
v_vs_831_ = lean_ctor_get(v_x_826_, 1);
v_isSharedCheck_855_ = !lean_is_exclusive(v_x_826_);
if (v_isSharedCheck_855_ == 0)
{
v___x_833_ = v_x_826_;
v_isShared_834_ = v_isSharedCheck_855_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_vs_831_);
lean_inc(v_ks_830_);
lean_dec(v_x_826_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_855_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___x_835_; uint8_t v___x_836_; 
v___x_835_ = lean_array_get_size(v_ks_830_);
v___x_836_ = lean_nat_dec_lt(v_x_827_, v___x_835_);
if (v___x_836_ == 0)
{
lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_840_; 
lean_dec(v_x_827_);
v___x_837_ = lean_array_push(v_ks_830_, v_x_828_);
v___x_838_ = lean_array_push(v_vs_831_, v_x_829_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 1, v___x_838_);
lean_ctor_set(v___x_833_, 0, v___x_837_);
v___x_840_ = v___x_833_;
goto v_reusejp_839_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v___x_837_);
lean_ctor_set(v_reuseFailAlloc_841_, 1, v___x_838_);
v___x_840_ = v_reuseFailAlloc_841_;
goto v_reusejp_839_;
}
v_reusejp_839_:
{
return v___x_840_;
}
}
else
{
lean_object* v_k_x27_842_; uint8_t v___x_843_; 
v_k_x27_842_ = lean_array_fget_borrowed(v_ks_830_, v_x_827_);
v___x_843_ = l_Lean_instBEqMVarId_beq(v_x_828_, v_k_x27_842_);
if (v___x_843_ == 0)
{
lean_object* v___x_845_; 
if (v_isShared_834_ == 0)
{
v___x_845_ = v___x_833_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_ks_830_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v_vs_831_);
v___x_845_ = v_reuseFailAlloc_849_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
lean_object* v___x_846_; lean_object* v___x_847_; 
v___x_846_ = lean_unsigned_to_nat(1u);
v___x_847_ = lean_nat_add(v_x_827_, v___x_846_);
lean_dec(v_x_827_);
v_x_826_ = v___x_845_;
v_x_827_ = v___x_847_;
goto _start;
}
}
else
{
lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_853_; 
v___x_850_ = lean_array_fset(v_ks_830_, v_x_827_, v_x_828_);
v___x_851_ = lean_array_fset(v_vs_831_, v_x_827_, v_x_829_);
lean_dec(v_x_827_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 1, v___x_851_);
lean_ctor_set(v___x_833_, 0, v___x_850_);
v___x_853_ = v___x_833_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_854_; 
v_reuseFailAlloc_854_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_854_, 0, v___x_850_);
lean_ctor_set(v_reuseFailAlloc_854_, 1, v___x_851_);
v___x_853_ = v_reuseFailAlloc_854_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
return v___x_853_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5___redArg(lean_object* v_n_856_, lean_object* v_k_857_, lean_object* v_v_858_){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_859_ = lean_unsigned_to_nat(0u);
v___x_860_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7___redArg(v_n_856_, v___x_859_, v_k_857_, v_v_858_);
return v___x_860_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_861_; 
v___x_861_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(lean_object* v_x_862_, size_t v_x_863_, size_t v_x_864_, lean_object* v_x_865_, lean_object* v_x_866_){
_start:
{
if (lean_obj_tag(v_x_862_) == 0)
{
lean_object* v_es_867_; size_t v___x_868_; size_t v___x_869_; lean_object* v_j_870_; lean_object* v___x_871_; uint8_t v___x_872_; 
v_es_867_ = lean_ctor_get(v_x_862_, 0);
v___x_868_ = ((size_t)31ULL);
v___x_869_ = lean_usize_land(v_x_863_, v___x_868_);
v_j_870_ = lean_usize_to_nat(v___x_869_);
v___x_871_ = lean_array_get_size(v_es_867_);
v___x_872_ = lean_nat_dec_lt(v_j_870_, v___x_871_);
if (v___x_872_ == 0)
{
lean_dec(v_j_870_);
lean_dec(v_x_866_);
lean_dec(v_x_865_);
return v_x_862_;
}
else
{
lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_911_; 
lean_inc_ref(v_es_867_);
v_isSharedCheck_911_ = !lean_is_exclusive(v_x_862_);
if (v_isSharedCheck_911_ == 0)
{
lean_object* v_unused_912_; 
v_unused_912_ = lean_ctor_get(v_x_862_, 0);
lean_dec(v_unused_912_);
v___x_874_ = v_x_862_;
v_isShared_875_ = v_isSharedCheck_911_;
goto v_resetjp_873_;
}
else
{
lean_dec(v_x_862_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_911_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v_v_876_; lean_object* v___x_877_; lean_object* v_xs_x27_878_; lean_object* v___y_880_; 
v_v_876_ = lean_array_fget(v_es_867_, v_j_870_);
v___x_877_ = lean_box(0);
v_xs_x27_878_ = lean_array_fset(v_es_867_, v_j_870_, v___x_877_);
switch(lean_obj_tag(v_v_876_))
{
case 0:
{
lean_object* v_key_885_; lean_object* v_val_886_; lean_object* v___x_888_; uint8_t v_isShared_889_; uint8_t v_isSharedCheck_896_; 
v_key_885_ = lean_ctor_get(v_v_876_, 0);
v_val_886_ = lean_ctor_get(v_v_876_, 1);
v_isSharedCheck_896_ = !lean_is_exclusive(v_v_876_);
if (v_isSharedCheck_896_ == 0)
{
v___x_888_ = v_v_876_;
v_isShared_889_ = v_isSharedCheck_896_;
goto v_resetjp_887_;
}
else
{
lean_inc(v_val_886_);
lean_inc(v_key_885_);
lean_dec(v_v_876_);
v___x_888_ = lean_box(0);
v_isShared_889_ = v_isSharedCheck_896_;
goto v_resetjp_887_;
}
v_resetjp_887_:
{
uint8_t v___x_890_; 
v___x_890_ = l_Lean_instBEqMVarId_beq(v_x_865_, v_key_885_);
if (v___x_890_ == 0)
{
lean_object* v___x_891_; lean_object* v___x_892_; 
lean_del_object(v___x_888_);
v___x_891_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_885_, v_val_886_, v_x_865_, v_x_866_);
v___x_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
v___y_880_ = v___x_892_;
goto v___jp_879_;
}
else
{
lean_object* v___x_894_; 
lean_dec(v_val_886_);
lean_dec(v_key_885_);
if (v_isShared_889_ == 0)
{
lean_ctor_set(v___x_888_, 1, v_x_866_);
lean_ctor_set(v___x_888_, 0, v_x_865_);
v___x_894_ = v___x_888_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_x_865_);
lean_ctor_set(v_reuseFailAlloc_895_, 1, v_x_866_);
v___x_894_ = v_reuseFailAlloc_895_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
v___y_880_ = v___x_894_;
goto v___jp_879_;
}
}
}
}
case 1:
{
lean_object* v_node_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_909_; 
v_node_897_ = lean_ctor_get(v_v_876_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v_v_876_);
if (v_isSharedCheck_909_ == 0)
{
v___x_899_ = v_v_876_;
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_node_897_);
lean_dec(v_v_876_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
size_t v___x_901_; size_t v___x_902_; size_t v___x_903_; size_t v___x_904_; lean_object* v___x_905_; lean_object* v___x_907_; 
v___x_901_ = ((size_t)5ULL);
v___x_902_ = lean_usize_shift_right(v_x_863_, v___x_901_);
v___x_903_ = ((size_t)1ULL);
v___x_904_ = lean_usize_add(v_x_864_, v___x_903_);
v___x_905_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(v_node_897_, v___x_902_, v___x_904_, v_x_865_, v_x_866_);
if (v_isShared_900_ == 0)
{
lean_ctor_set(v___x_899_, 0, v___x_905_);
v___x_907_ = v___x_899_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v___x_905_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
v___y_880_ = v___x_907_;
goto v___jp_879_;
}
}
}
default: 
{
lean_object* v___x_910_; 
v___x_910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_910_, 0, v_x_865_);
lean_ctor_set(v___x_910_, 1, v_x_866_);
v___y_880_ = v___x_910_;
goto v___jp_879_;
}
}
v___jp_879_:
{
lean_object* v___x_881_; lean_object* v___x_883_; 
v___x_881_ = lean_array_fset(v_xs_x27_878_, v_j_870_, v___y_880_);
lean_dec(v_j_870_);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v___x_881_);
v___x_883_ = v___x_874_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___x_881_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
}
else
{
lean_object* v_ks_913_; lean_object* v_vs_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_934_; 
v_ks_913_ = lean_ctor_get(v_x_862_, 0);
v_vs_914_ = lean_ctor_get(v_x_862_, 1);
v_isSharedCheck_934_ = !lean_is_exclusive(v_x_862_);
if (v_isSharedCheck_934_ == 0)
{
v___x_916_ = v_x_862_;
v_isShared_917_ = v_isSharedCheck_934_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_vs_914_);
lean_inc(v_ks_913_);
lean_dec(v_x_862_);
v___x_916_ = lean_box(0);
v_isShared_917_ = v_isSharedCheck_934_;
goto v_resetjp_915_;
}
v_resetjp_915_:
{
lean_object* v___x_919_; 
if (v_isShared_917_ == 0)
{
v___x_919_ = v___x_916_;
goto v_reusejp_918_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_ks_913_);
lean_ctor_set(v_reuseFailAlloc_933_, 1, v_vs_914_);
v___x_919_ = v_reuseFailAlloc_933_;
goto v_reusejp_918_;
}
v_reusejp_918_:
{
lean_object* v_newNode_920_; uint8_t v___y_922_; size_t v___x_928_; uint8_t v___x_929_; 
v_newNode_920_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5___redArg(v___x_919_, v_x_865_, v_x_866_);
v___x_928_ = ((size_t)7ULL);
v___x_929_ = lean_usize_dec_le(v___x_928_, v_x_864_);
if (v___x_929_ == 0)
{
lean_object* v___x_930_; lean_object* v___x_931_; uint8_t v___x_932_; 
v___x_930_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_920_);
v___x_931_ = lean_unsigned_to_nat(4u);
v___x_932_ = lean_nat_dec_lt(v___x_930_, v___x_931_);
lean_dec(v___x_930_);
v___y_922_ = v___x_932_;
goto v___jp_921_;
}
else
{
v___y_922_ = v___x_929_;
goto v___jp_921_;
}
v___jp_921_:
{
if (v___y_922_ == 0)
{
lean_object* v_ks_923_; lean_object* v_vs_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v_ks_923_ = lean_ctor_get(v_newNode_920_, 0);
lean_inc_ref(v_ks_923_);
v_vs_924_ = lean_ctor_get(v_newNode_920_, 1);
lean_inc_ref(v_vs_924_);
lean_dec_ref(v_newNode_920_);
v___x_925_ = lean_unsigned_to_nat(0u);
v___x_926_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___closed__0);
v___x_927_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg(v_x_864_, v_ks_923_, v_vs_924_, v___x_925_, v___x_926_);
lean_dec_ref(v_vs_924_);
lean_dec_ref(v_ks_923_);
return v___x_927_;
}
else
{
return v_newNode_920_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg(size_t v_depth_935_, lean_object* v_keys_936_, lean_object* v_vals_937_, lean_object* v_i_938_, lean_object* v_entries_939_){
_start:
{
lean_object* v___x_940_; uint8_t v___x_941_; 
v___x_940_ = lean_array_get_size(v_keys_936_);
v___x_941_ = lean_nat_dec_lt(v_i_938_, v___x_940_);
if (v___x_941_ == 0)
{
lean_dec(v_i_938_);
return v_entries_939_;
}
else
{
lean_object* v_k_942_; lean_object* v_v_943_; uint64_t v___x_944_; size_t v_h_945_; size_t v___x_946_; lean_object* v___x_947_; size_t v___x_948_; size_t v___x_949_; size_t v___x_950_; size_t v_h_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v_k_942_ = lean_array_fget_borrowed(v_keys_936_, v_i_938_);
v_v_943_ = lean_array_fget_borrowed(v_vals_937_, v_i_938_);
v___x_944_ = l_Lean_instHashableMVarId_hash(v_k_942_);
v_h_945_ = lean_uint64_to_usize(v___x_944_);
v___x_946_ = ((size_t)5ULL);
v___x_947_ = lean_unsigned_to_nat(1u);
v___x_948_ = ((size_t)1ULL);
v___x_949_ = lean_usize_sub(v_depth_935_, v___x_948_);
v___x_950_ = lean_usize_mul(v___x_946_, v___x_949_);
v_h_951_ = lean_usize_shift_right(v_h_945_, v___x_950_);
v___x_952_ = lean_nat_add(v_i_938_, v___x_947_);
lean_dec(v_i_938_);
lean_inc(v_v_943_);
lean_inc(v_k_942_);
v___x_953_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(v_entries_939_, v_h_951_, v_depth_935_, v_k_942_, v_v_943_);
v_i_938_ = v___x_952_;
v_entries_939_ = v___x_953_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg___boxed(lean_object* v_depth_955_, lean_object* v_keys_956_, lean_object* v_vals_957_, lean_object* v_i_958_, lean_object* v_entries_959_){
_start:
{
size_t v_depth_boxed_960_; lean_object* v_res_961_; 
v_depth_boxed_960_ = lean_unbox_usize(v_depth_955_);
lean_dec(v_depth_955_);
v_res_961_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg(v_depth_boxed_960_, v_keys_956_, v_vals_957_, v_i_958_, v_entries_959_);
lean_dec_ref(v_vals_957_);
lean_dec_ref(v_keys_956_);
return v_res_961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_962_, lean_object* v_x_963_, lean_object* v_x_964_, lean_object* v_x_965_, lean_object* v_x_966_){
_start:
{
size_t v_x_10327__boxed_967_; size_t v_x_10328__boxed_968_; lean_object* v_res_969_; 
v_x_10327__boxed_967_ = lean_unbox_usize(v_x_963_);
lean_dec(v_x_963_);
v_x_10328__boxed_968_ = lean_unbox_usize(v_x_964_);
lean_dec(v_x_964_);
v_res_969_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(v_x_962_, v_x_10327__boxed_967_, v_x_10328__boxed_968_, v_x_965_, v_x_966_);
return v_res_969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1___redArg(lean_object* v_x_970_, lean_object* v_x_971_, lean_object* v_x_972_){
_start:
{
uint64_t v___x_973_; size_t v___x_974_; size_t v___x_975_; lean_object* v___x_976_; 
v___x_973_ = l_Lean_instHashableMVarId_hash(v_x_971_);
v___x_974_ = lean_uint64_to_usize(v___x_973_);
v___x_975_ = ((size_t)1ULL);
v___x_976_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(v_x_970_, v___x_974_, v___x_975_, v_x_971_, v_x_972_);
return v___x_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg(lean_object* v_mvarId_977_, lean_object* v_val_978_, lean_object* v___y_979_){
_start:
{
lean_object* v___x_981_; lean_object* v_mctx_982_; lean_object* v_cache_983_; lean_object* v_zetaDeltaFVarIds_984_; lean_object* v_postponed_985_; lean_object* v_diag_986_; lean_object* v___x_988_; uint8_t v_isShared_989_; uint8_t v_isSharedCheck_1014_; 
v___x_981_ = lean_st_ref_take(v___y_979_);
v_mctx_982_ = lean_ctor_get(v___x_981_, 0);
v_cache_983_ = lean_ctor_get(v___x_981_, 1);
v_zetaDeltaFVarIds_984_ = lean_ctor_get(v___x_981_, 2);
v_postponed_985_ = lean_ctor_get(v___x_981_, 3);
v_diag_986_ = lean_ctor_get(v___x_981_, 4);
v_isSharedCheck_1014_ = !lean_is_exclusive(v___x_981_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_988_ = v___x_981_;
v_isShared_989_ = v_isSharedCheck_1014_;
goto v_resetjp_987_;
}
else
{
lean_inc(v_diag_986_);
lean_inc(v_postponed_985_);
lean_inc(v_zetaDeltaFVarIds_984_);
lean_inc(v_cache_983_);
lean_inc(v_mctx_982_);
lean_dec(v___x_981_);
v___x_988_ = lean_box(0);
v_isShared_989_ = v_isSharedCheck_1014_;
goto v_resetjp_987_;
}
v_resetjp_987_:
{
lean_object* v_depth_990_; lean_object* v_levelAssignDepth_991_; lean_object* v_lmvarCounter_992_; lean_object* v_mvarCounter_993_; lean_object* v_lDecls_994_; lean_object* v_decls_995_; lean_object* v_userNames_996_; lean_object* v_lAssignment_997_; lean_object* v_eAssignment_998_; lean_object* v_dAssignment_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1013_; 
v_depth_990_ = lean_ctor_get(v_mctx_982_, 0);
v_levelAssignDepth_991_ = lean_ctor_get(v_mctx_982_, 1);
v_lmvarCounter_992_ = lean_ctor_get(v_mctx_982_, 2);
v_mvarCounter_993_ = lean_ctor_get(v_mctx_982_, 3);
v_lDecls_994_ = lean_ctor_get(v_mctx_982_, 4);
v_decls_995_ = lean_ctor_get(v_mctx_982_, 5);
v_userNames_996_ = lean_ctor_get(v_mctx_982_, 6);
v_lAssignment_997_ = lean_ctor_get(v_mctx_982_, 7);
v_eAssignment_998_ = lean_ctor_get(v_mctx_982_, 8);
v_dAssignment_999_ = lean_ctor_get(v_mctx_982_, 9);
v_isSharedCheck_1013_ = !lean_is_exclusive(v_mctx_982_);
if (v_isSharedCheck_1013_ == 0)
{
v___x_1001_ = v_mctx_982_;
v_isShared_1002_ = v_isSharedCheck_1013_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_dAssignment_999_);
lean_inc(v_eAssignment_998_);
lean_inc(v_lAssignment_997_);
lean_inc(v_userNames_996_);
lean_inc(v_decls_995_);
lean_inc(v_lDecls_994_);
lean_inc(v_mvarCounter_993_);
lean_inc(v_lmvarCounter_992_);
lean_inc(v_levelAssignDepth_991_);
lean_inc(v_depth_990_);
lean_dec(v_mctx_982_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1013_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
lean_object* v___x_1003_; lean_object* v___x_1005_; 
v___x_1003_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1___redArg(v_eAssignment_998_, v_mvarId_977_, v_val_978_);
if (v_isShared_1002_ == 0)
{
lean_ctor_set(v___x_1001_, 8, v___x_1003_);
v___x_1005_ = v___x_1001_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1012_; 
v_reuseFailAlloc_1012_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1012_, 0, v_depth_990_);
lean_ctor_set(v_reuseFailAlloc_1012_, 1, v_levelAssignDepth_991_);
lean_ctor_set(v_reuseFailAlloc_1012_, 2, v_lmvarCounter_992_);
lean_ctor_set(v_reuseFailAlloc_1012_, 3, v_mvarCounter_993_);
lean_ctor_set(v_reuseFailAlloc_1012_, 4, v_lDecls_994_);
lean_ctor_set(v_reuseFailAlloc_1012_, 5, v_decls_995_);
lean_ctor_set(v_reuseFailAlloc_1012_, 6, v_userNames_996_);
lean_ctor_set(v_reuseFailAlloc_1012_, 7, v_lAssignment_997_);
lean_ctor_set(v_reuseFailAlloc_1012_, 8, v___x_1003_);
lean_ctor_set(v_reuseFailAlloc_1012_, 9, v_dAssignment_999_);
v___x_1005_ = v_reuseFailAlloc_1012_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
lean_object* v___x_1007_; 
if (v_isShared_989_ == 0)
{
lean_ctor_set(v___x_988_, 0, v___x_1005_);
v___x_1007_ = v___x_988_;
goto v_reusejp_1006_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_1005_);
lean_ctor_set(v_reuseFailAlloc_1011_, 1, v_cache_983_);
lean_ctor_set(v_reuseFailAlloc_1011_, 2, v_zetaDeltaFVarIds_984_);
lean_ctor_set(v_reuseFailAlloc_1011_, 3, v_postponed_985_);
lean_ctor_set(v_reuseFailAlloc_1011_, 4, v_diag_986_);
v___x_1007_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1006_;
}
v_reusejp_1006_:
{
lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1008_ = lean_st_ref_set(v___y_979_, v___x_1007_);
v___x_1009_ = lean_box(0);
v___x_1010_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1009_);
return v___x_1010_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg___boxed(lean_object* v_mvarId_1015_, lean_object* v_val_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_){
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg(v_mvarId_1015_, v_val_1016_, v___y_1017_);
lean_dec(v___y_1017_);
return v_res_1019_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1(void){
_start:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; 
v___x_1021_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__0));
v___x_1022_ = l_Lean_stringToMessageData(v___x_1021_);
return v___x_1022_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3(void){
_start:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__2));
v___x_1025_ = l_Lean_stringToMessageData(v___x_1024_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget(lean_object* v_m_1026_, lean_object* v_proc_1027_, uint8_t v_ifUnchanged_1028_, lean_object* v_goal_1029_, lean_object* v_a_1030_, lean_object* v_a_1031_, lean_object* v_a_1032_, lean_object* v_a_1033_, lean_object* v_a_1034_){
_start:
{
lean_object* v___x_1036_; 
lean_inc(v_goal_1029_);
v___x_1036_ = l_Lean_MVarId_getType(v_goal_1029_, v_a_1031_, v_a_1032_, v_a_1033_, v_a_1034_);
if (lean_obj_tag(v___x_1036_) == 0)
{
lean_object* v_a_1037_; lean_object* v___x_1038_; lean_object* v_a_1039_; lean_object* v___x_1041_; uint8_t v_isShared_1042_; uint8_t v_isSharedCheck_1152_; 
v_a_1037_ = lean_ctor_get(v___x_1036_, 0);
lean_inc(v_a_1037_);
lean_dec_ref_known(v___x_1036_, 1);
v___x_1038_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(v_a_1037_, v_a_1032_);
v_a_1039_ = lean_ctor_get(v___x_1038_, 0);
v_isSharedCheck_1152_ = !lean_is_exclusive(v___x_1038_);
if (v_isSharedCheck_1152_ == 0)
{
v___x_1041_ = v___x_1038_;
v_isShared_1042_ = v_isSharedCheck_1152_;
goto v_resetjp_1040_;
}
else
{
lean_inc(v_a_1039_);
lean_dec(v___x_1038_);
v___x_1041_ = lean_box(0);
v_isShared_1042_ = v_isSharedCheck_1152_;
goto v_resetjp_1040_;
}
v_resetjp_1040_:
{
lean_object* v___x_1043_; 
lean_inc(v_a_1034_);
lean_inc_ref(v_a_1033_);
lean_inc(v_a_1032_);
lean_inc_ref(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1039_);
v___x_1043_ = lean_apply_7(v_m_1026_, v_a_1039_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, v_a_1034_, lean_box(0));
if (lean_obj_tag(v___x_1043_) == 0)
{
lean_object* v_a_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1143_; 
v_a_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1143_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_a_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1143_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v_expr_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; uint8_t v___x_1051_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v___y_1056_; lean_object* v___y_1057_; 
v_expr_1048_ = lean_ctor_get(v_a_1044_, 0);
lean_inc(v_a_1039_);
v___x_1049_ = l_Lean_Expr_cleanupAnnotations(v_a_1039_);
lean_inc_ref(v_expr_1048_);
v___x_1050_ = l_Lean_Expr_cleanupAnnotations(v_expr_1048_);
v___x_1051_ = lean_expr_eqv(v___x_1049_, v___x_1050_);
lean_dec_ref(v___x_1050_);
lean_dec_ref(v___x_1049_);
if (v___x_1051_ == 0)
{
lean_dec_ref(v_proc_1027_);
v___y_1053_ = v_a_1030_;
v___y_1054_ = v_a_1031_;
v___y_1055_ = v_a_1032_;
v___y_1056_ = v_a_1033_;
v___y_1057_ = v_a_1034_;
goto v___jp_1052_;
}
else
{
switch(v_ifUnchanged_1028_)
{
case 0:
{
lean_dec_ref(v_proc_1027_);
v___y_1053_ = v_a_1030_;
v___y_1054_ = v_a_1031_;
v___y_1055_ = v_a_1032_;
v___y_1056_ = v_a_1033_;
v___y_1057_ = v_a_1034_;
goto v___jp_1052_;
}
case 1:
{
lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; 
v___x_1115_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1);
v___x_1116_ = l_Lean_stringToMessageData(v_proc_1027_);
v___x_1117_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1115_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
v___x_1118_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3);
v___x_1119_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1117_);
lean_ctor_set(v___x_1119_, 1, v___x_1118_);
v___x_1120_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2(v___x_1119_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, v_a_1034_);
if (lean_obj_tag(v___x_1120_) == 0)
{
lean_dec_ref_known(v___x_1120_, 1);
v___y_1053_ = v_a_1030_;
v___y_1054_ = v_a_1031_;
v___y_1055_ = v_a_1032_;
v___y_1056_ = v_a_1033_;
v___y_1057_ = v_a_1034_;
goto v___jp_1052_;
}
else
{
lean_object* v_a_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1128_; 
lean_del_object(v___x_1046_);
lean_dec(v_a_1044_);
lean_del_object(v___x_1041_);
lean_dec(v_a_1039_);
lean_dec(v_goal_1029_);
v_a_1121_ = lean_ctor_get(v___x_1120_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_1120_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1123_ = v___x_1120_;
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_a_1121_);
lean_dec(v___x_1120_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1126_; 
if (v_isShared_1124_ == 0)
{
v___x_1126_ = v___x_1123_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_a_1121_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
return v___x_1126_;
}
}
}
}
default: 
{
lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
lean_del_object(v___x_1046_);
lean_dec(v_a_1044_);
lean_del_object(v___x_1041_);
lean_dec(v_a_1039_);
lean_dec(v_goal_1029_);
v___x_1129_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1);
v___x_1130_ = l_Lean_stringToMessageData(v_proc_1027_);
v___x_1131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1131_, 0, v___x_1129_);
lean_ctor_set(v___x_1131_, 1, v___x_1130_);
v___x_1132_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__3);
v___x_1133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1131_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
v___x_1134_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(v___x_1133_, v_a_1031_, v_a_1032_, v_a_1033_, v_a_1034_);
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1137_ = v___x_1134_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1134_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1135_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
}
v___jp_1052_:
{
uint8_t v___x_1058_; 
lean_inc_ref(v_expr_1048_);
v___x_1058_ = l_Lean_Expr_isTrue(v_expr_1048_);
if (v___x_1058_ == 0)
{
if (v___x_1051_ == 0)
{
lean_object* v___x_1059_; 
lean_del_object(v___x_1046_);
v___x_1059_ = l_Lean_Meta_applySimpResultToTarget(v_goal_1029_, v_a_1039_, v_a_1044_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_);
lean_dec(v_a_1039_);
if (lean_obj_tag(v___x_1059_) == 0)
{
lean_object* v_a_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1070_; 
v_a_1060_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1062_ = v___x_1059_;
v_isShared_1063_ = v_isSharedCheck_1070_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_a_1060_);
lean_dec(v___x_1059_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1070_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1065_; 
if (v_isShared_1042_ == 0)
{
lean_ctor_set_tag(v___x_1041_, 1);
lean_ctor_set(v___x_1041_, 0, v_a_1060_);
v___x_1065_ = v___x_1041_;
goto v_reusejp_1064_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_a_1060_);
v___x_1065_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1064_;
}
v_reusejp_1064_:
{
lean_object* v___x_1067_; 
if (v_isShared_1063_ == 0)
{
lean_ctor_set(v___x_1062_, 0, v___x_1065_);
v___x_1067_ = v___x_1062_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v___x_1065_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
}
}
else
{
lean_object* v_a_1071_; lean_object* v___x_1073_; uint8_t v_isShared_1074_; uint8_t v_isSharedCheck_1078_; 
lean_del_object(v___x_1041_);
v_a_1071_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1073_ = v___x_1059_;
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
else
{
lean_inc(v_a_1071_);
lean_dec(v___x_1059_);
v___x_1073_ = lean_box(0);
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
v_resetjp_1072_:
{
lean_object* v___x_1076_; 
if (v_isShared_1074_ == 0)
{
v___x_1076_ = v___x_1073_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1071_);
v___x_1076_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1075_;
}
v_reusejp_1075_:
{
return v___x_1076_;
}
}
}
}
else
{
lean_object* v___x_1080_; 
lean_dec(v_a_1044_);
lean_dec(v_a_1039_);
if (v_isShared_1042_ == 0)
{
lean_ctor_set_tag(v___x_1041_, 1);
lean_ctor_set(v___x_1041_, 0, v_goal_1029_);
v___x_1080_ = v___x_1041_;
goto v_reusejp_1079_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_goal_1029_);
v___x_1080_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1079_;
}
v_reusejp_1079_:
{
lean_object* v___x_1082_; 
if (v_isShared_1047_ == 0)
{
lean_ctor_set(v___x_1046_, 0, v___x_1080_);
v___x_1082_ = v___x_1046_;
goto v_reusejp_1081_;
}
else
{
lean_object* v_reuseFailAlloc_1083_; 
v_reuseFailAlloc_1083_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1083_, 0, v___x_1080_);
v___x_1082_ = v_reuseFailAlloc_1083_;
goto v_reusejp_1081_;
}
v_reusejp_1081_:
{
return v___x_1082_;
}
}
}
}
else
{
lean_object* v___x_1085_; 
lean_del_object(v___x_1046_);
lean_del_object(v___x_1041_);
lean_dec(v_a_1039_);
v___x_1085_ = l_Lean_Meta_Simp_Result_getProof(v_a_1044_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_);
if (lean_obj_tag(v___x_1085_) == 0)
{
lean_object* v_a_1086_; lean_object* v___x_1087_; 
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
lean_inc(v_a_1086_);
lean_dec_ref_known(v___x_1085_, 1);
v___x_1087_ = l_Lean_Meta_mkOfEqTrue(v_a_1086_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_);
if (lean_obj_tag(v___x_1087_) == 0)
{
lean_object* v_a_1088_; lean_object* v___x_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1097_; 
v_a_1088_ = lean_ctor_get(v___x_1087_, 0);
lean_inc(v_a_1088_);
lean_dec_ref_known(v___x_1087_, 1);
v___x_1089_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg(v_goal_1029_, v_a_1088_, v___y_1055_);
v_isSharedCheck_1097_ = !lean_is_exclusive(v___x_1089_);
if (v_isSharedCheck_1097_ == 0)
{
lean_object* v_unused_1098_; 
v_unused_1098_ = lean_ctor_get(v___x_1089_, 0);
lean_dec(v_unused_1098_);
v___x_1091_ = v___x_1089_;
v_isShared_1092_ = v_isSharedCheck_1097_;
goto v_resetjp_1090_;
}
else
{
lean_dec(v___x_1089_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1097_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1093_; lean_object* v___x_1095_; 
v___x_1093_ = lean_box(0);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 0, v___x_1093_);
v___x_1095_ = v___x_1091_;
goto v_reusejp_1094_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v___x_1093_);
v___x_1095_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1094_;
}
v_reusejp_1094_:
{
return v___x_1095_;
}
}
}
else
{
lean_object* v_a_1099_; lean_object* v___x_1101_; uint8_t v_isShared_1102_; uint8_t v_isSharedCheck_1106_; 
lean_dec(v_goal_1029_);
v_a_1099_ = lean_ctor_get(v___x_1087_, 0);
v_isSharedCheck_1106_ = !lean_is_exclusive(v___x_1087_);
if (v_isSharedCheck_1106_ == 0)
{
v___x_1101_ = v___x_1087_;
v_isShared_1102_ = v_isSharedCheck_1106_;
goto v_resetjp_1100_;
}
else
{
lean_inc(v_a_1099_);
lean_dec(v___x_1087_);
v___x_1101_ = lean_box(0);
v_isShared_1102_ = v_isSharedCheck_1106_;
goto v_resetjp_1100_;
}
v_resetjp_1100_:
{
lean_object* v___x_1104_; 
if (v_isShared_1102_ == 0)
{
v___x_1104_ = v___x_1101_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1105_; 
v_reuseFailAlloc_1105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1105_, 0, v_a_1099_);
v___x_1104_ = v_reuseFailAlloc_1105_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
return v___x_1104_;
}
}
}
}
else
{
lean_object* v_a_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1114_; 
lean_dec(v_goal_1029_);
v_a_1107_ = lean_ctor_get(v___x_1085_, 0);
v_isSharedCheck_1114_ = !lean_is_exclusive(v___x_1085_);
if (v_isSharedCheck_1114_ == 0)
{
v___x_1109_ = v___x_1085_;
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_a_1107_);
lean_dec(v___x_1085_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1114_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v___x_1112_; 
if (v_isShared_1110_ == 0)
{
v___x_1112_ = v___x_1109_;
goto v_reusejp_1111_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v_a_1107_);
v___x_1112_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1111_;
}
v_reusejp_1111_:
{
return v___x_1112_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1144_; lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1151_; 
lean_del_object(v___x_1041_);
lean_dec(v_a_1039_);
lean_dec(v_goal_1029_);
lean_dec_ref(v_proc_1027_);
v_a_1144_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1151_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1151_ == 0)
{
v___x_1146_ = v___x_1043_;
v_isShared_1147_ = v_isSharedCheck_1151_;
goto v_resetjp_1145_;
}
else
{
lean_inc(v_a_1144_);
lean_dec(v___x_1043_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1151_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v___x_1149_; 
if (v_isShared_1147_ == 0)
{
v___x_1149_ = v___x_1146_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1150_; 
v_reuseFailAlloc_1150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1150_, 0, v_a_1144_);
v___x_1149_ = v_reuseFailAlloc_1150_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
return v___x_1149_;
}
}
}
}
}
else
{
lean_object* v_a_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1160_; 
lean_dec(v_goal_1029_);
lean_dec_ref(v_proc_1027_);
lean_dec_ref(v_m_1026_);
v_a_1153_ = lean_ctor_get(v___x_1036_, 0);
v_isSharedCheck_1160_ = !lean_is_exclusive(v___x_1036_);
if (v_isSharedCheck_1160_ == 0)
{
v___x_1155_ = v___x_1036_;
v_isShared_1156_ = v_isSharedCheck_1160_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_a_1153_);
lean_dec(v___x_1036_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1160_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___x_1158_; 
if (v_isShared_1156_ == 0)
{
v___x_1158_ = v___x_1155_;
goto v_reusejp_1157_;
}
else
{
lean_object* v_reuseFailAlloc_1159_; 
v_reuseFailAlloc_1159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1159_, 0, v_a_1153_);
v___x_1158_ = v_reuseFailAlloc_1159_;
goto v_reusejp_1157_;
}
v_reusejp_1157_:
{
return v___x_1158_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtTarget___boxed(lean_object* v_m_1161_, lean_object* v_proc_1162_, lean_object* v_ifUnchanged_1163_, lean_object* v_goal_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_, lean_object* v_a_1170_){
_start:
{
uint8_t v_ifUnchanged_boxed_1171_; lean_object* v_res_1172_; 
v_ifUnchanged_boxed_1171_ = lean_unbox(v_ifUnchanged_1163_);
v_res_1172_ = lp_mathlib_Mathlib_Tactic_transformAtTarget(v_m_1161_, v_proc_1162_, v_ifUnchanged_boxed_1171_, v_goal_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_);
lean_dec(v_a_1169_);
lean_dec_ref(v_a_1168_);
lean_dec(v_a_1167_);
lean_dec_ref(v_a_1166_);
lean_dec_ref(v_a_1165_);
return v_res_1172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1(lean_object* v_mvarId_1173_, lean_object* v_val_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_){
_start:
{
lean_object* v___x_1181_; 
v___x_1181_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___redArg(v_mvarId_1173_, v_val_1174_, v___y_1177_);
return v___x_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1___boxed(lean_object* v_mvarId_1182_, lean_object* v_val_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1(v_mvarId_1182_, v_val_1183_, v___y_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
lean_dec(v___y_1186_);
lean_dec_ref(v___y_1185_);
lean_dec_ref(v___y_1184_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3(lean_object* v_00_u03b1_1191_, lean_object* v_msg_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_){
_start:
{
lean_object* v___x_1199_; 
v___x_1199_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(v_msg_1192_, v___y_1194_, v___y_1195_, v___y_1196_, v___y_1197_);
return v___x_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___boxed(lean_object* v_00_u03b1_1200_, lean_object* v_msg_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v_res_1208_; 
v_res_1208_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3(v_00_u03b1_1200_, v_msg_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
lean_dec_ref(v___y_1202_);
return v_res_1208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1(lean_object* v_00_u03b2_1209_, lean_object* v_x_1210_, lean_object* v_x_1211_, lean_object* v_x_1212_){
_start:
{
lean_object* v___x_1213_; 
v___x_1213_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1___redArg(v_x_1210_, v_x_1211_, v_x_1212_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_1214_, lean_object* v_x_1215_, size_t v_x_1216_, size_t v_x_1217_, lean_object* v_x_1218_, lean_object* v_x_1219_){
_start:
{
lean_object* v___x_1220_; 
v___x_1220_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___redArg(v_x_1215_, v_x_1216_, v_x_1217_, v_x_1218_, v_x_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1221_, lean_object* v_x_1222_, lean_object* v_x_1223_, lean_object* v_x_1224_, lean_object* v_x_1225_, lean_object* v_x_1226_){
_start:
{
size_t v_x_10869__boxed_1227_; size_t v_x_10870__boxed_1228_; lean_object* v_res_1229_; 
v_x_10869__boxed_1227_ = lean_unbox_usize(v_x_1223_);
lean_dec(v_x_1223_);
v_x_10870__boxed_1228_ = lean_unbox_usize(v_x_1224_);
lean_dec(v_x_1224_);
v_res_1229_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2(v_00_u03b2_1221_, v_x_1222_, v_x_10869__boxed_1227_, v_x_10870__boxed_1228_, v_x_1225_, v_x_1226_);
return v_res_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5(lean_object* v_ref_1230_, lean_object* v_msgData_1231_, uint8_t v_severity_1232_, uint8_t v_isSilent_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___x_1240_; 
v___x_1240_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___redArg(v_ref_1230_, v_msgData_1231_, v_severity_1232_, v_isSilent_1233_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
return v___x_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5___boxed(lean_object* v_ref_1241_, lean_object* v_msgData_1242_, lean_object* v_severity_1243_, lean_object* v_isSilent_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
uint8_t v_severity_boxed_1251_; uint8_t v_isSilent_boxed_1252_; lean_object* v_res_1253_; 
v_severity_boxed_1251_ = lean_unbox(v_severity_1243_);
v_isSilent_boxed_1252_ = lean_unbox(v_isSilent_1244_);
v_res_1253_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2_spec__3_spec__5(v_ref_1241_, v_msgData_1242_, v_severity_boxed_1251_, v_isSilent_boxed_1252_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
lean_dec(v___y_1247_);
lean_dec_ref(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v_ref_1241_);
return v_res_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_1254_, lean_object* v_n_1255_, lean_object* v_k_1256_, lean_object* v_v_1257_){
_start:
{
lean_object* v___x_1258_; 
v___x_1258_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5___redArg(v_n_1255_, v_k_1256_, v_v_1257_);
return v___x_1258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6(lean_object* v_00_u03b2_1259_, size_t v_depth_1260_, lean_object* v_keys_1261_, lean_object* v_vals_1262_, lean_object* v_heq_1263_, lean_object* v_i_1264_, lean_object* v_entries_1265_){
_start:
{
lean_object* v___x_1266_; 
v___x_1266_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___redArg(v_depth_1260_, v_keys_1261_, v_vals_1262_, v_i_1264_, v_entries_1265_);
return v___x_1266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6___boxed(lean_object* v_00_u03b2_1267_, lean_object* v_depth_1268_, lean_object* v_keys_1269_, lean_object* v_vals_1270_, lean_object* v_heq_1271_, lean_object* v_i_1272_, lean_object* v_entries_1273_){
_start:
{
size_t v_depth_boxed_1274_; lean_object* v_res_1275_; 
v_depth_boxed_1274_ = lean_unbox_usize(v_depth_1268_);
lean_dec(v_depth_1268_);
v_res_1275_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__6(v_00_u03b2_1267_, v_depth_boxed_1274_, v_keys_1269_, v_vals_1270_, v_heq_1271_, v_i_1272_, v_entries_1273_);
lean_dec_ref(v_vals_1270_);
lean_dec_ref(v_keys_1269_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7(lean_object* v_00_u03b2_1276_, lean_object* v_x_1277_, lean_object* v_x_1278_, lean_object* v_x_1279_, lean_object* v_x_1280_){
_start:
{
lean_object* v___x_1281_; 
v___x_1281_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_transformAtTarget_spec__1_spec__1_spec__2_spec__5_spec__7___redArg(v_x_1277_, v_x_1278_, v_x_1279_, v_x_1280_);
return v___x_1281_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1(void){
_start:
{
lean_object* v___x_1283_; lean_object* v___x_1284_; 
v___x_1283_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__0));
v___x_1284_ = l_Lean_stringToMessageData(v___x_1283_);
return v___x_1284_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3(void){
_start:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; 
v___x_1286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__2));
v___x_1287_ = l_Lean_stringToMessageData(v___x_1286_);
return v___x_1287_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5(void){
_start:
{
lean_object* v___x_1289_; lean_object* v___x_1290_; 
v___x_1289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__4));
v___x_1290_ = l_Lean_stringToMessageData(v___x_1289_);
return v___x_1290_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7(void){
_start:
{
lean_object* v___x_1292_; lean_object* v___x_1293_; 
v___x_1292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__6));
v___x_1293_ = l_Lean_stringToMessageData(v___x_1292_);
return v___x_1293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl(lean_object* v_m_1294_, lean_object* v_proc_1295_, uint8_t v_ifUnchanged_1296_, uint8_t v_mayCloseGoal_1297_, lean_object* v_fvarId_1298_, lean_object* v_goal_1299_, lean_object* v_a_1300_, lean_object* v_a_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_){
_start:
{
lean_object* v___y_1307_; lean_object* v___y_1308_; lean_object* v___y_1309_; lean_object* v___y_1310_; lean_object* v___y_1311_; lean_object* v___y_1343_; lean_object* v___y_1344_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1347_; lean_object* v___x_1420_; 
lean_inc(v_fvarId_1298_);
v___x_1420_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_1298_, v_a_1301_, v_a_1303_, v_a_1304_);
if (lean_obj_tag(v___x_1420_) == 0)
{
lean_object* v_a_1421_; uint8_t v___x_1422_; 
v_a_1421_ = lean_ctor_get(v___x_1420_, 0);
lean_inc(v_a_1421_);
lean_dec_ref_known(v___x_1420_, 1);
v___x_1422_ = l_Lean_LocalDecl_isImplementationDetail(v_a_1421_);
lean_dec(v_a_1421_);
if (v___x_1422_ == 0)
{
v___y_1343_ = v_a_1300_;
v___y_1344_ = v_a_1301_;
v___y_1345_ = v_a_1302_;
v___y_1346_ = v_a_1303_;
v___y_1347_ = v_a_1304_;
goto v___jp_1342_;
}
else
{
lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v_a_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1441_; 
lean_dec(v_goal_1299_);
lean_dec_ref(v_m_1294_);
v___x_1423_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3, &lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__3);
v___x_1424_ = l_Lean_stringToMessageData(v_proc_1295_);
v___x_1425_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1423_);
lean_ctor_set(v___x_1425_, 1, v___x_1424_);
v___x_1426_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5, &lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__5);
v___x_1427_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1427_, 0, v___x_1425_);
lean_ctor_set(v___x_1427_, 1, v___x_1426_);
v___x_1428_ = l_Lean_Expr_fvar___override(v_fvarId_1298_);
v___x_1429_ = l_Lean_MessageData_ofExpr(v___x_1428_);
v___x_1430_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1427_);
lean_ctor_set(v___x_1430_, 1, v___x_1429_);
v___x_1431_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7, &lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__7);
v___x_1432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1432_, 0, v___x_1430_);
lean_ctor_set(v___x_1432_, 1, v___x_1431_);
v___x_1433_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(v___x_1432_, v_a_1301_, v_a_1302_, v_a_1303_, v_a_1304_);
v_a_1434_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1441_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1441_ == 0)
{
v___x_1436_ = v___x_1433_;
v_isShared_1437_ = v_isSharedCheck_1441_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_a_1434_);
lean_dec(v___x_1433_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1441_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1439_; 
if (v_isShared_1437_ == 0)
{
v___x_1439_ = v___x_1436_;
goto v_reusejp_1438_;
}
else
{
lean_object* v_reuseFailAlloc_1440_; 
v_reuseFailAlloc_1440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1440_, 0, v_a_1434_);
v___x_1439_ = v_reuseFailAlloc_1440_;
goto v_reusejp_1438_;
}
v_reusejp_1438_:
{
return v___x_1439_;
}
}
}
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec(v_goal_1299_);
lean_dec(v_fvarId_1298_);
lean_dec_ref(v_proc_1295_);
lean_dec_ref(v_m_1294_);
v_a_1442_ = lean_ctor_get(v___x_1420_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1420_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1420_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1420_);
v___x_1444_ = lean_box(0);
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
v_resetjp_1443_:
{
lean_object* v___x_1447_; 
if (v_isShared_1445_ == 0)
{
v___x_1447_ = v___x_1444_;
goto v_reusejp_1446_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v_a_1442_);
v___x_1447_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1446_;
}
v_reusejp_1446_:
{
return v___x_1447_;
}
}
}
v___jp_1306_:
{
lean_object* v___x_1312_; 
v___x_1312_ = l_Lean_Meta_applySimpResultToLocalDecl(v_goal_1299_, v_fvarId_1298_, v___y_1307_, v_mayCloseGoal_1297_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_);
if (lean_obj_tag(v___x_1312_) == 0)
{
lean_object* v_a_1313_; lean_object* v___x_1315_; uint8_t v_isShared_1316_; uint8_t v_isSharedCheck_1333_; 
v_a_1313_ = lean_ctor_get(v___x_1312_, 0);
v_isSharedCheck_1333_ = !lean_is_exclusive(v___x_1312_);
if (v_isSharedCheck_1333_ == 0)
{
v___x_1315_ = v___x_1312_;
v_isShared_1316_ = v_isSharedCheck_1333_;
goto v_resetjp_1314_;
}
else
{
lean_inc(v_a_1313_);
lean_dec(v___x_1312_);
v___x_1315_ = lean_box(0);
v_isShared_1316_ = v_isSharedCheck_1333_;
goto v_resetjp_1314_;
}
v_resetjp_1314_:
{
if (lean_obj_tag(v_a_1313_) == 0)
{
lean_object* v___x_1317_; lean_object* v___x_1319_; 
v___x_1317_ = lean_box(0);
if (v_isShared_1316_ == 0)
{
lean_ctor_set(v___x_1315_, 0, v___x_1317_);
v___x_1319_ = v___x_1315_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v___x_1317_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
else
{
lean_object* v_val_1321_; lean_object* v___x_1323_; uint8_t v_isShared_1324_; uint8_t v_isSharedCheck_1332_; 
v_val_1321_ = lean_ctor_get(v_a_1313_, 0);
v_isSharedCheck_1332_ = !lean_is_exclusive(v_a_1313_);
if (v_isSharedCheck_1332_ == 0)
{
v___x_1323_ = v_a_1313_;
v_isShared_1324_ = v_isSharedCheck_1332_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_val_1321_);
lean_dec(v_a_1313_);
v___x_1323_ = lean_box(0);
v_isShared_1324_ = v_isSharedCheck_1332_;
goto v_resetjp_1322_;
}
v_resetjp_1322_:
{
lean_object* v_snd_1325_; lean_object* v___x_1327_; 
v_snd_1325_ = lean_ctor_get(v_val_1321_, 1);
lean_inc(v_snd_1325_);
lean_dec(v_val_1321_);
if (v_isShared_1324_ == 0)
{
lean_ctor_set(v___x_1323_, 0, v_snd_1325_);
v___x_1327_ = v___x_1323_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1331_; 
v_reuseFailAlloc_1331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1331_, 0, v_snd_1325_);
v___x_1327_ = v_reuseFailAlloc_1331_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
lean_object* v___x_1329_; 
if (v_isShared_1316_ == 0)
{
lean_ctor_set(v___x_1315_, 0, v___x_1327_);
v___x_1329_ = v___x_1315_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1330_; 
v_reuseFailAlloc_1330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1330_, 0, v___x_1327_);
v___x_1329_ = v_reuseFailAlloc_1330_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
return v___x_1329_;
}
}
}
}
}
}
else
{
lean_object* v_a_1334_; lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1341_; 
v_a_1334_ = lean_ctor_get(v___x_1312_, 0);
v_isSharedCheck_1341_ = !lean_is_exclusive(v___x_1312_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1336_ = v___x_1312_;
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
else
{
lean_inc(v_a_1334_);
lean_dec(v___x_1312_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v___x_1339_; 
if (v_isShared_1337_ == 0)
{
v___x_1339_ = v___x_1336_;
goto v_reusejp_1338_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v_a_1334_);
v___x_1339_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1338_;
}
v_reusejp_1338_:
{
return v___x_1339_;
}
}
}
}
v___jp_1342_:
{
lean_object* v___x_1348_; 
lean_inc(v_fvarId_1298_);
v___x_1348_ = l_Lean_FVarId_getType___redArg(v_fvarId_1298_, v___y_1344_, v___y_1346_, v___y_1347_);
if (lean_obj_tag(v___x_1348_) == 0)
{
lean_object* v_a_1349_; lean_object* v___x_1350_; lean_object* v_a_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1411_; 
v_a_1349_ = lean_ctor_get(v___x_1348_, 0);
lean_inc(v_a_1349_);
lean_dec_ref_known(v___x_1348_, 1);
v___x_1350_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_transformAtTarget_spec__0___redArg(v_a_1349_, v___y_1345_);
v_a_1351_ = lean_ctor_get(v___x_1350_, 0);
v_isSharedCheck_1411_ = !lean_is_exclusive(v___x_1350_);
if (v_isSharedCheck_1411_ == 0)
{
v___x_1353_ = v___x_1350_;
v_isShared_1354_ = v_isSharedCheck_1411_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_a_1351_);
lean_dec(v___x_1350_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1411_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v_simpTheorems_1355_; lean_object* v___x_1357_; 
v_simpTheorems_1355_ = lean_ctor_get(v___y_1343_, 6);
lean_inc(v_fvarId_1298_);
if (v_isShared_1354_ == 0)
{
lean_ctor_set_tag(v___x_1353_, 1);
lean_ctor_set(v___x_1353_, 0, v_fvarId_1298_);
v___x_1357_ = v___x_1353_;
goto v_reusejp_1356_;
}
else
{
lean_object* v_reuseFailAlloc_1410_; 
v_reuseFailAlloc_1410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1410_, 0, v_fvarId_1298_);
v___x_1357_ = v_reuseFailAlloc_1410_;
goto v_reusejp_1356_;
}
v_reusejp_1356_:
{
lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; 
lean_inc_ref(v_simpTheorems_1355_);
v___x_1358_ = l_Lean_Meta_SimpTheoremsArray_eraseTheorem(v_simpTheorems_1355_, v___x_1357_);
lean_inc_ref(v___y_1343_);
v___x_1359_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v___y_1343_, v___x_1358_);
lean_inc(v___y_1347_);
lean_inc_ref(v___y_1346_);
lean_inc(v___y_1345_);
lean_inc_ref(v___y_1344_);
lean_inc(v_a_1351_);
v___x_1360_ = lean_apply_7(v_m_1294_, v_a_1351_, v___x_1359_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_, lean_box(0));
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_object* v_a_1361_; lean_object* v_expr_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; uint8_t v___x_1365_; 
v_a_1361_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_a_1361_);
lean_dec_ref_known(v___x_1360_, 1);
v_expr_1362_ = lean_ctor_get(v_a_1361_, 0);
v___x_1363_ = l_Lean_Expr_cleanupAnnotations(v_a_1351_);
lean_inc_ref(v_expr_1362_);
v___x_1364_ = l_Lean_Expr_cleanupAnnotations(v_expr_1362_);
v___x_1365_ = lean_expr_eqv(v___x_1363_, v___x_1364_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
if (v___x_1365_ == 0)
{
lean_dec_ref(v_proc_1295_);
v___y_1307_ = v_a_1361_;
v___y_1308_ = v___y_1344_;
v___y_1309_ = v___y_1345_;
v___y_1310_ = v___y_1346_;
v___y_1311_ = v___y_1347_;
goto v___jp_1306_;
}
else
{
switch(v_ifUnchanged_1296_)
{
case 0:
{
lean_dec_ref(v_proc_1295_);
v___y_1307_ = v_a_1361_;
v___y_1308_ = v___y_1344_;
v___y_1309_ = v___y_1345_;
v___y_1310_ = v___y_1346_;
v___y_1311_ = v___y_1347_;
goto v___jp_1306_;
}
case 1:
{
lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___x_1366_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1);
v___x_1367_ = l_Lean_stringToMessageData(v_proc_1295_);
v___x_1368_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1368_, 0, v___x_1366_);
lean_ctor_set(v___x_1368_, 1, v___x_1367_);
v___x_1369_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1);
v___x_1370_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1370_, 0, v___x_1368_);
lean_ctor_set(v___x_1370_, 1, v___x_1369_);
lean_inc(v_fvarId_1298_);
v___x_1371_ = l_Lean_Expr_fvar___override(v_fvarId_1298_);
v___x_1372_ = l_Lean_MessageData_ofExpr(v___x_1371_);
v___x_1373_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1370_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
v___x_1374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1374_, 0, v___x_1373_);
lean_ctor_set(v___x_1374_, 1, v___x_1366_);
v___x_1375_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_transformAtTarget_spec__2(v___x_1374_, v___y_1343_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_);
if (lean_obj_tag(v___x_1375_) == 0)
{
lean_dec_ref_known(v___x_1375_, 1);
v___y_1307_ = v_a_1361_;
v___y_1308_ = v___y_1344_;
v___y_1309_ = v___y_1345_;
v___y_1310_ = v___y_1346_;
v___y_1311_ = v___y_1347_;
goto v___jp_1306_;
}
else
{
lean_object* v_a_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1383_; 
lean_dec(v_a_1361_);
lean_dec(v_goal_1299_);
lean_dec(v_fvarId_1298_);
v_a_1376_ = lean_ctor_get(v___x_1375_, 0);
v_isSharedCheck_1383_ = !lean_is_exclusive(v___x_1375_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1378_ = v___x_1375_;
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_a_1376_);
lean_dec(v___x_1375_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1381_; 
if (v_isShared_1379_ == 0)
{
v___x_1381_ = v___x_1378_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v_a_1376_);
v___x_1381_ = v_reuseFailAlloc_1382_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
return v___x_1381_;
}
}
}
}
default: 
{
lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v_a_1394_; lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1401_; 
lean_dec(v_a_1361_);
lean_dec(v_goal_1299_);
v___x_1384_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1);
v___x_1385_ = l_Lean_stringToMessageData(v_proc_1295_);
v___x_1386_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1386_, 0, v___x_1384_);
lean_ctor_set(v___x_1386_, 1, v___x_1385_);
v___x_1387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___closed__1);
v___x_1388_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1388_, 0, v___x_1386_);
lean_ctor_set(v___x_1388_, 1, v___x_1387_);
v___x_1389_ = l_Lean_Expr_fvar___override(v_fvarId_1298_);
v___x_1390_ = l_Lean_MessageData_ofExpr(v___x_1389_);
v___x_1391_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1391_, 0, v___x_1388_);
lean_ctor_set(v___x_1391_, 1, v___x_1390_);
v___x_1392_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1392_, 0, v___x_1391_);
lean_ctor_set(v___x_1392_, 1, v___x_1384_);
v___x_1393_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_transformAtTarget_spec__3___redArg(v___x_1392_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_);
v_a_1394_ = lean_ctor_get(v___x_1393_, 0);
v_isSharedCheck_1401_ = !lean_is_exclusive(v___x_1393_);
if (v_isSharedCheck_1401_ == 0)
{
v___x_1396_ = v___x_1393_;
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
else
{
lean_inc(v_a_1394_);
lean_dec(v___x_1393_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v___x_1399_; 
if (v_isShared_1397_ == 0)
{
v___x_1399_ = v___x_1396_;
goto v_reusejp_1398_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v_a_1394_);
v___x_1399_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1398_;
}
v_reusejp_1398_:
{
return v___x_1399_;
}
}
}
}
}
}
else
{
lean_object* v_a_1402_; lean_object* v___x_1404_; uint8_t v_isShared_1405_; uint8_t v_isSharedCheck_1409_; 
lean_dec(v_a_1351_);
lean_dec(v_goal_1299_);
lean_dec(v_fvarId_1298_);
lean_dec_ref(v_proc_1295_);
v_a_1402_ = lean_ctor_get(v___x_1360_, 0);
v_isSharedCheck_1409_ = !lean_is_exclusive(v___x_1360_);
if (v_isSharedCheck_1409_ == 0)
{
v___x_1404_ = v___x_1360_;
v_isShared_1405_ = v_isSharedCheck_1409_;
goto v_resetjp_1403_;
}
else
{
lean_inc(v_a_1402_);
lean_dec(v___x_1360_);
v___x_1404_ = lean_box(0);
v_isShared_1405_ = v_isSharedCheck_1409_;
goto v_resetjp_1403_;
}
v_resetjp_1403_:
{
lean_object* v___x_1407_; 
if (v_isShared_1405_ == 0)
{
v___x_1407_ = v___x_1404_;
goto v_reusejp_1406_;
}
else
{
lean_object* v_reuseFailAlloc_1408_; 
v_reuseFailAlloc_1408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1408_, 0, v_a_1402_);
v___x_1407_ = v_reuseFailAlloc_1408_;
goto v_reusejp_1406_;
}
v_reusejp_1406_:
{
return v___x_1407_;
}
}
}
}
}
}
else
{
lean_object* v_a_1412_; lean_object* v___x_1414_; uint8_t v_isShared_1415_; uint8_t v_isSharedCheck_1419_; 
lean_dec(v_goal_1299_);
lean_dec(v_fvarId_1298_);
lean_dec_ref(v_proc_1295_);
lean_dec_ref(v_m_1294_);
v_a_1412_ = lean_ctor_get(v___x_1348_, 0);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1348_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1414_ = v___x_1348_;
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
else
{
lean_inc(v_a_1412_);
lean_dec(v___x_1348_);
v___x_1414_ = lean_box(0);
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
v_resetjp_1413_:
{
lean_object* v___x_1417_; 
if (v_isShared_1415_ == 0)
{
v___x_1417_ = v___x_1414_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_a_1412_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocalDecl___boxed(lean_object* v_m_1450_, lean_object* v_proc_1451_, lean_object* v_ifUnchanged_1452_, lean_object* v_mayCloseGoal_1453_, lean_object* v_fvarId_1454_, lean_object* v_goal_1455_, lean_object* v_a_1456_, lean_object* v_a_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_, lean_object* v_a_1460_, lean_object* v_a_1461_){
_start:
{
uint8_t v_ifUnchanged_boxed_1462_; uint8_t v_mayCloseGoal_boxed_1463_; lean_object* v_res_1464_; 
v_ifUnchanged_boxed_1462_ = lean_unbox(v_ifUnchanged_1452_);
v_mayCloseGoal_boxed_1463_ = lean_unbox(v_mayCloseGoal_1453_);
v_res_1464_ = lp_mathlib_Mathlib_Tactic_transformAtLocalDecl(v_m_1450_, v_proc_1451_, v_ifUnchanged_boxed_1462_, v_mayCloseGoal_boxed_1463_, v_fvarId_1454_, v_goal_1455_, v_a_1456_, v_a_1457_, v_a_1458_, v_a_1459_, v_a_1460_);
lean_dec(v_a_1460_);
lean_dec_ref(v_a_1459_);
lean_dec(v_a_1458_);
lean_dec_ref(v_a_1457_);
lean_dec_ref(v_a_1456_);
return v_res_1464_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1466_; lean_object* v___x_1467_; 
v___x_1466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__0));
v___x_1467_ = l_Lean_stringToMessageData(v___x_1466_);
return v___x_1467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0(lean_object* v_proc_1468_, lean_object* v_x_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_){
_start:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; 
v___x_1479_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtTarget___closed__1);
v___x_1480_ = l_Lean_stringToMessageData(v_proc_1468_);
v___x_1481_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1481_, 0, v___x_1479_);
lean_ctor_set(v___x_1481_, 1, v___x_1480_);
v___x_1482_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___closed__1);
v___x_1483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1483_, 0, v___x_1481_);
lean_ctor_set(v___x_1483_, 1, v___x_1482_);
v___x_1484_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_withNondepPropLocation_spec__1___redArg(v___x_1483_, v___y_1474_, v___y_1475_, v___y_1476_, v___y_1477_);
return v___x_1484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___boxed(lean_object* v_proc_1485_, lean_object* v_x_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_){
_start:
{
lean_object* v_res_1496_; 
v_res_1496_ = lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0(v_proc_1485_, v_x_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
lean_dec(v___y_1494_);
lean_dec_ref(v___y_1493_);
lean_dec(v___y_1492_);
lean_dec_ref(v___y_1491_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
lean_dec(v___y_1488_);
lean_dec_ref(v___y_1487_);
lean_dec(v_x_1486_);
return v_res_1496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1(lean_object* v_m_1497_, lean_object* v_proc_1498_, uint8_t v_ifUnchanged_1499_, uint8_t v_mayCloseGoalFromHyp_1500_, lean_object* v___y_1501_, lean_object* v_ctx_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_){
_start:
{
lean_object* v___x_1512_; 
v___x_1512_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1504_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
if (lean_obj_tag(v___x_1512_) == 0)
{
lean_object* v_a_1513_; lean_object* v___x_1514_; 
v_a_1513_ = lean_ctor_get(v___x_1512_, 0);
lean_inc(v_a_1513_);
lean_dec_ref_known(v___x_1512_, 1);
v___x_1514_ = lp_mathlib_Mathlib_Tactic_transformAtLocalDecl(v_m_1497_, v_proc_1498_, v_ifUnchanged_1499_, v_mayCloseGoalFromHyp_1500_, v___y_1501_, v_a_1513_, v_ctx_1502_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
if (lean_obj_tag(v___x_1514_) == 0)
{
lean_object* v_a_1515_; 
v_a_1515_ = lean_ctor_get(v___x_1514_, 0);
lean_inc(v_a_1515_);
lean_dec_ref_known(v___x_1514_, 1);
if (lean_obj_tag(v_a_1515_) == 1)
{
lean_object* v_val_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v_val_1516_ = lean_ctor_get(v_a_1515_, 0);
lean_inc(v_val_1516_);
lean_dec_ref_known(v_a_1515_, 1);
v___x_1517_ = lean_box(0);
v___x_1518_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1518_, 0, v_val_1516_);
lean_ctor_set(v___x_1518_, 1, v___x_1517_);
v___x_1519_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1518_, v___y_1504_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
return v___x_1519_;
}
else
{
lean_object* v___x_1520_; lean_object* v___x_1521_; 
lean_dec(v_a_1515_);
v___x_1520_ = lean_box(0);
v___x_1521_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1520_, v___y_1504_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
return v___x_1521_;
}
}
else
{
lean_object* v_a_1522_; lean_object* v___x_1524_; uint8_t v_isShared_1525_; uint8_t v_isSharedCheck_1529_; 
v_a_1522_ = lean_ctor_get(v___x_1514_, 0);
v_isSharedCheck_1529_ = !lean_is_exclusive(v___x_1514_);
if (v_isSharedCheck_1529_ == 0)
{
v___x_1524_ = v___x_1514_;
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
else
{
lean_inc(v_a_1522_);
lean_dec(v___x_1514_);
v___x_1524_ = lean_box(0);
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
v_resetjp_1523_:
{
lean_object* v___x_1527_; 
if (v_isShared_1525_ == 0)
{
v___x_1527_ = v___x_1524_;
goto v_reusejp_1526_;
}
else
{
lean_object* v_reuseFailAlloc_1528_; 
v_reuseFailAlloc_1528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1528_, 0, v_a_1522_);
v___x_1527_ = v_reuseFailAlloc_1528_;
goto v_reusejp_1526_;
}
v_reusejp_1526_:
{
return v___x_1527_;
}
}
}
}
else
{
lean_object* v_a_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1537_; 
lean_dec(v___y_1501_);
lean_dec_ref(v_proc_1498_);
lean_dec_ref(v_m_1497_);
v_a_1530_ = lean_ctor_get(v___x_1512_, 0);
v_isSharedCheck_1537_ = !lean_is_exclusive(v___x_1512_);
if (v_isSharedCheck_1537_ == 0)
{
v___x_1532_ = v___x_1512_;
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_a_1530_);
lean_dec(v___x_1512_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___x_1535_; 
if (v_isShared_1533_ == 0)
{
v___x_1535_ = v___x_1532_;
goto v_reusejp_1534_;
}
else
{
lean_object* v_reuseFailAlloc_1536_; 
v_reuseFailAlloc_1536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1536_, 0, v_a_1530_);
v___x_1535_ = v_reuseFailAlloc_1536_;
goto v_reusejp_1534_;
}
v_reusejp_1534_:
{
return v___x_1535_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1___boxed(lean_object* v_m_1538_, lean_object* v_proc_1539_, lean_object* v_ifUnchanged_1540_, lean_object* v_mayCloseGoalFromHyp_1541_, lean_object* v___y_1542_, lean_object* v_ctx_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_){
_start:
{
uint8_t v_ifUnchanged_boxed_1553_; uint8_t v_mayCloseGoalFromHyp_boxed_1554_; lean_object* v_res_1555_; 
v_ifUnchanged_boxed_1553_ = lean_unbox(v_ifUnchanged_1540_);
v_mayCloseGoalFromHyp_boxed_1554_ = lean_unbox(v_mayCloseGoalFromHyp_1541_);
v_res_1555_ = lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1(v_m_1538_, v_proc_1539_, v_ifUnchanged_boxed_1553_, v_mayCloseGoalFromHyp_boxed_1554_, v___y_1542_, v_ctx_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_, v___y_1551_);
lean_dec(v___y_1551_);
lean_dec_ref(v___y_1550_);
lean_dec(v___y_1549_);
lean_dec_ref(v___y_1548_);
lean_dec(v___y_1547_);
lean_dec_ref(v___y_1546_);
lean_dec(v___y_1545_);
lean_dec_ref(v___y_1544_);
lean_dec_ref(v_ctx_1543_);
return v_res_1555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2(lean_object* v_m_1556_, lean_object* v_proc_1557_, uint8_t v_ifUnchanged_1558_, uint8_t v_mayCloseGoalFromHyp_1559_, lean_object* v_ctx_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_){
_start:
{
lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___f_1573_; lean_object* v___x_1574_; 
v___x_1571_ = lean_box(v_ifUnchanged_1558_);
v___x_1572_ = lean_box(v_mayCloseGoalFromHyp_1559_);
v___f_1573_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__1___boxed), 15, 6);
lean_closure_set(v___f_1573_, 0, v_m_1556_);
lean_closure_set(v___f_1573_, 1, v_proc_1557_);
lean_closure_set(v___f_1573_, 2, v___x_1571_);
lean_closure_set(v___f_1573_, 3, v___x_1572_);
lean_closure_set(v___f_1573_, 4, v___y_1561_);
lean_closure_set(v___f_1573_, 5, v_ctx_1560_);
v___x_1574_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1573_, v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_);
return v___x_1574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2___boxed(lean_object* v_m_1575_, lean_object* v_proc_1576_, lean_object* v_ifUnchanged_1577_, lean_object* v_mayCloseGoalFromHyp_1578_, lean_object* v_ctx_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_){
_start:
{
uint8_t v_ifUnchanged_boxed_1590_; uint8_t v_mayCloseGoalFromHyp_boxed_1591_; lean_object* v_res_1592_; 
v_ifUnchanged_boxed_1590_ = lean_unbox(v_ifUnchanged_1577_);
v_mayCloseGoalFromHyp_boxed_1591_ = lean_unbox(v_mayCloseGoalFromHyp_1578_);
v_res_1592_ = lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2(v_m_1575_, v_proc_1576_, v_ifUnchanged_boxed_1590_, v_mayCloseGoalFromHyp_boxed_1591_, v_ctx_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
return v_res_1592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3(lean_object* v_m_1593_, lean_object* v_proc_1594_, uint8_t v_ifUnchanged_1595_, lean_object* v_ctx_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_){
_start:
{
lean_object* v___x_1606_; 
v___x_1606_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1598_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_);
if (lean_obj_tag(v___x_1606_) == 0)
{
lean_object* v_a_1607_; lean_object* v___x_1608_; 
v_a_1607_ = lean_ctor_get(v___x_1606_, 0);
lean_inc(v_a_1607_);
lean_dec_ref_known(v___x_1606_, 1);
v___x_1608_ = lp_mathlib_Mathlib_Tactic_transformAtTarget(v_m_1593_, v_proc_1594_, v_ifUnchanged_1595_, v_a_1607_, v_ctx_1596_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v_a_1609_; 
v_a_1609_ = lean_ctor_get(v___x_1608_, 0);
lean_inc(v_a_1609_);
lean_dec_ref_known(v___x_1608_, 1);
if (lean_obj_tag(v_a_1609_) == 1)
{
lean_object* v_val_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
v_val_1610_ = lean_ctor_get(v_a_1609_, 0);
lean_inc(v_val_1610_);
lean_dec_ref_known(v_a_1609_, 1);
v___x_1611_ = lean_box(0);
v___x_1612_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1612_, 0, v_val_1610_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
v___x_1613_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1612_, v___y_1598_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_);
return v___x_1613_;
}
else
{
lean_object* v___x_1614_; lean_object* v___x_1615_; 
lean_dec(v_a_1609_);
v___x_1614_ = lean_box(0);
v___x_1615_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1614_, v___y_1598_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_);
return v___x_1615_;
}
}
else
{
lean_object* v_a_1616_; lean_object* v___x_1618_; uint8_t v_isShared_1619_; uint8_t v_isSharedCheck_1623_; 
v_a_1616_ = lean_ctor_get(v___x_1608_, 0);
v_isSharedCheck_1623_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1623_ == 0)
{
v___x_1618_ = v___x_1608_;
v_isShared_1619_ = v_isSharedCheck_1623_;
goto v_resetjp_1617_;
}
else
{
lean_inc(v_a_1616_);
lean_dec(v___x_1608_);
v___x_1618_ = lean_box(0);
v_isShared_1619_ = v_isSharedCheck_1623_;
goto v_resetjp_1617_;
}
v_resetjp_1617_:
{
lean_object* v___x_1621_; 
if (v_isShared_1619_ == 0)
{
v___x_1621_ = v___x_1618_;
goto v_reusejp_1620_;
}
else
{
lean_object* v_reuseFailAlloc_1622_; 
v_reuseFailAlloc_1622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1622_, 0, v_a_1616_);
v___x_1621_ = v_reuseFailAlloc_1622_;
goto v_reusejp_1620_;
}
v_reusejp_1620_:
{
return v___x_1621_;
}
}
}
}
else
{
lean_object* v_a_1624_; lean_object* v___x_1626_; uint8_t v_isShared_1627_; uint8_t v_isSharedCheck_1631_; 
lean_dec_ref(v_proc_1594_);
lean_dec_ref(v_m_1593_);
v_a_1624_ = lean_ctor_get(v___x_1606_, 0);
v_isSharedCheck_1631_ = !lean_is_exclusive(v___x_1606_);
if (v_isSharedCheck_1631_ == 0)
{
v___x_1626_ = v___x_1606_;
v_isShared_1627_ = v_isSharedCheck_1631_;
goto v_resetjp_1625_;
}
else
{
lean_inc(v_a_1624_);
lean_dec(v___x_1606_);
v___x_1626_ = lean_box(0);
v_isShared_1627_ = v_isSharedCheck_1631_;
goto v_resetjp_1625_;
}
v_resetjp_1625_:
{
lean_object* v___x_1629_; 
if (v_isShared_1627_ == 0)
{
v___x_1629_ = v___x_1626_;
goto v_reusejp_1628_;
}
else
{
lean_object* v_reuseFailAlloc_1630_; 
v_reuseFailAlloc_1630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1630_, 0, v_a_1624_);
v___x_1629_ = v_reuseFailAlloc_1630_;
goto v_reusejp_1628_;
}
v_reusejp_1628_:
{
return v___x_1629_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3___boxed(lean_object* v_m_1632_, lean_object* v_proc_1633_, lean_object* v_ifUnchanged_1634_, lean_object* v_ctx_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_){
_start:
{
uint8_t v_ifUnchanged_boxed_1645_; lean_object* v_res_1646_; 
v_ifUnchanged_boxed_1645_ = lean_unbox(v_ifUnchanged_1634_);
v_res_1646_ = lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3(v_m_1632_, v_proc_1633_, v_ifUnchanged_boxed_1645_, v_ctx_1635_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
lean_dec(v___y_1643_);
lean_dec_ref(v___y_1642_);
lean_dec(v___y_1641_);
lean_dec_ref(v___y_1640_);
lean_dec(v___y_1639_);
lean_dec_ref(v___y_1638_);
lean_dec(v___y_1637_);
lean_dec_ref(v___y_1636_);
lean_dec_ref(v_ctx_1635_);
return v_res_1646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4(lean_object* v___f_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_){
_start:
{
lean_object* v___x_1657_; 
v___x_1657_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1647_, v___y_1648_, v___y_1649_, v___y_1650_, v___y_1651_, v___y_1652_, v___y_1653_, v___y_1654_, v___y_1655_);
return v___x_1657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4___boxed(lean_object* v___f_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_){
_start:
{
lean_object* v_res_1668_; 
v_res_1668_ = lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4(v___f_1658_, v___y_1659_, v___y_1660_, v___y_1661_, v___y_1662_, v___y_1663_, v___y_1664_, v___y_1665_, v___y_1666_);
lean_dec(v___y_1666_);
lean_dec_ref(v___y_1665_);
lean_dec(v___y_1664_);
lean_dec_ref(v___y_1663_);
lean_dec(v___y_1662_);
lean_dec_ref(v___y_1661_);
lean_dec(v___y_1660_);
lean_dec_ref(v___y_1659_);
return v_res_1668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation(lean_object* v_m_1669_, lean_object* v_proc_1670_, lean_object* v_loc_1671_, uint8_t v_ifUnchanged_1672_, uint8_t v_mayCloseGoalFromHyp_1673_, lean_object* v_ctx_1674_, lean_object* v_a_1675_, lean_object* v_a_1676_, lean_object* v_a_1677_, lean_object* v_a_1678_, lean_object* v_a_1679_, lean_object* v_a_1680_, lean_object* v_a_1681_, lean_object* v_a_1682_){
_start:
{
lean_object* v___f_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___f_1687_; lean_object* v___x_1688_; lean_object* v___f_1689_; lean_object* v___f_1690_; lean_object* v___x_1691_; 
lean_inc_ref_n(v_proc_1670_, 2);
v___f_1684_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1684_, 0, v_proc_1670_);
v___x_1685_ = lean_box(v_ifUnchanged_1672_);
v___x_1686_ = lean_box(v_mayCloseGoalFromHyp_1673_);
lean_inc_ref(v_ctx_1674_);
lean_inc_ref(v_m_1669_);
v___f_1687_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2___boxed), 15, 5);
lean_closure_set(v___f_1687_, 0, v_m_1669_);
lean_closure_set(v___f_1687_, 1, v_proc_1670_);
lean_closure_set(v___f_1687_, 2, v___x_1685_);
lean_closure_set(v___f_1687_, 3, v___x_1686_);
lean_closure_set(v___f_1687_, 4, v_ctx_1674_);
v___x_1688_ = lean_box(v_ifUnchanged_1672_);
v___f_1689_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3___boxed), 13, 4);
lean_closure_set(v___f_1689_, 0, v_m_1669_);
lean_closure_set(v___f_1689_, 1, v_proc_1670_);
lean_closure_set(v___f_1689_, 2, v___x_1688_);
lean_closure_set(v___f_1689_, 3, v_ctx_1674_);
v___f_1690_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4___boxed), 10, 1);
lean_closure_set(v___f_1690_, 0, v___f_1689_);
v___x_1691_ = l_Lean_Elab_Tactic_withLocation(v_loc_1671_, v___f_1687_, v___f_1690_, v___f_1684_, v_a_1675_, v_a_1676_, v_a_1677_, v_a_1678_, v_a_1679_, v_a_1680_, v_a_1681_, v_a_1682_);
return v___x_1691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtLocation___boxed(lean_object* v_m_1692_, lean_object* v_proc_1693_, lean_object* v_loc_1694_, lean_object* v_ifUnchanged_1695_, lean_object* v_mayCloseGoalFromHyp_1696_, lean_object* v_ctx_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_, lean_object* v_a_1700_, lean_object* v_a_1701_, lean_object* v_a_1702_, lean_object* v_a_1703_, lean_object* v_a_1704_, lean_object* v_a_1705_, lean_object* v_a_1706_){
_start:
{
uint8_t v_ifUnchanged_boxed_1707_; uint8_t v_mayCloseGoalFromHyp_boxed_1708_; lean_object* v_res_1709_; 
v_ifUnchanged_boxed_1707_ = lean_unbox(v_ifUnchanged_1695_);
v_mayCloseGoalFromHyp_boxed_1708_ = lean_unbox(v_mayCloseGoalFromHyp_1696_);
v_res_1709_ = lp_mathlib_Mathlib_Tactic_transformAtLocation(v_m_1692_, v_proc_1693_, v_loc_1694_, v_ifUnchanged_boxed_1707_, v_mayCloseGoalFromHyp_boxed_1708_, v_ctx_1697_, v_a_1698_, v_a_1699_, v_a_1700_, v_a_1701_, v_a_1702_, v_a_1703_, v_a_1704_, v_a_1705_);
lean_dec(v_a_1705_);
lean_dec_ref(v_a_1704_);
lean_dec(v_a_1703_);
lean_dec_ref(v_a_1702_);
lean_dec(v_a_1701_);
lean_dec_ref(v_a_1700_);
lean_dec(v_a_1699_);
lean_dec_ref(v_a_1698_);
lean_dec(v_loc_1694_);
return v_res_1709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(lean_object* v_m_1710_, lean_object* v_proc_1711_, lean_object* v_loc_1712_, uint8_t v_ifUnchanged_1713_, uint8_t v_mayCloseGoalFromHyp_1714_, lean_object* v_ctx_1715_, lean_object* v_a_1716_, lean_object* v_a_1717_, lean_object* v_a_1718_, lean_object* v_a_1719_, lean_object* v_a_1720_, lean_object* v_a_1721_, lean_object* v_a_1722_, lean_object* v_a_1723_){
_start:
{
lean_object* v___f_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___f_1728_; lean_object* v___x_1729_; lean_object* v___f_1730_; lean_object* v___f_1731_; lean_object* v___x_1732_; 
lean_inc_ref_n(v_proc_1711_, 2);
v___f_1725_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1725_, 0, v_proc_1711_);
v___x_1726_ = lean_box(v_ifUnchanged_1713_);
v___x_1727_ = lean_box(v_mayCloseGoalFromHyp_1714_);
lean_inc_ref(v_ctx_1715_);
lean_inc_ref(v_m_1710_);
v___f_1728_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__2___boxed), 15, 5);
lean_closure_set(v___f_1728_, 0, v_m_1710_);
lean_closure_set(v___f_1728_, 1, v_proc_1711_);
lean_closure_set(v___f_1728_, 2, v___x_1726_);
lean_closure_set(v___f_1728_, 3, v___x_1727_);
lean_closure_set(v___f_1728_, 4, v_ctx_1715_);
v___x_1729_ = lean_box(v_ifUnchanged_1713_);
v___f_1730_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__3___boxed), 13, 4);
lean_closure_set(v___f_1730_, 0, v_m_1710_);
lean_closure_set(v___f_1730_, 1, v_proc_1711_);
lean_closure_set(v___f_1730_, 2, v___x_1729_);
lean_closure_set(v___f_1730_, 3, v_ctx_1715_);
v___f_1731_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_transformAtLocation___lam__4___boxed), 10, 1);
lean_closure_set(v___f_1731_, 0, v___f_1730_);
v___x_1732_ = lp_mathlib_Lean_Elab_Tactic_withNondepPropLocation(v_loc_1712_, v___f_1728_, v___f_1731_, v___f_1725_, v_a_1716_, v_a_1717_, v_a_1718_, v_a_1719_, v_a_1720_, v_a_1721_, v_a_1722_, v_a_1723_);
return v___x_1732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation___boxed(lean_object* v_m_1733_, lean_object* v_proc_1734_, lean_object* v_loc_1735_, lean_object* v_ifUnchanged_1736_, lean_object* v_mayCloseGoalFromHyp_1737_, lean_object* v_ctx_1738_, lean_object* v_a_1739_, lean_object* v_a_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_, lean_object* v_a_1747_){
_start:
{
uint8_t v_ifUnchanged_boxed_1748_; uint8_t v_mayCloseGoalFromHyp_boxed_1749_; lean_object* v_res_1750_; 
v_ifUnchanged_boxed_1748_ = lean_unbox(v_ifUnchanged_1736_);
v_mayCloseGoalFromHyp_boxed_1749_ = lean_unbox(v_mayCloseGoalFromHyp_1737_);
v_res_1750_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v_m_1733_, v_proc_1734_, v_loc_1735_, v_ifUnchanged_boxed_1748_, v_mayCloseGoalFromHyp_boxed_1749_, v_ctx_1738_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
lean_dec(v_a_1746_);
lean_dec_ref(v_a_1745_);
lean_dec(v_a_1744_);
lean_dec_ref(v_a_1743_);
lean_dec(v_a_1742_);
lean_dec_ref(v_a_1741_);
lean_dec(v_a_1740_);
lean_dec_ref(v_a_1739_);
lean_dec(v_loc_1735_);
return v_res_1750_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin) {
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
res = runtime_initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged_default = _init_lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged_default();
lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged = _init_lp_mathlib_Mathlib_Tactic_instInhabitedBehaviorIfUnchanged();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Main(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_AtLocation(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_AtLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_AtLocation(builtin);
}
#ifdef __cplusplus
}
#endif
