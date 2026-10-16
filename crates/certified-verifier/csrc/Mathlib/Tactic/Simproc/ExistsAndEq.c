// Lean compiler output
// Module: Mathlib.Tactic.Simproc.ExistsAndEq
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Qq public import Qq public import Qq.Typ
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
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_constLevels_x21(lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_replaceFVar(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_substCore(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Meta_FVarSubst_apply(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_replaceFVars(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_AbstractMVarsResult_numMVars(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instBEqGoTo_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo = (const lean_object*)&lp_mathlib_ExistsAndEq_instBEqGoTo___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instInhabitedGoTo_default;
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instInhabitedGoTo;
static const lean_string_object lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2;
static lean_once_cell_t lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3;
static lean_once_cell_t lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instInhabitedVarQ;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instInhabitedHypQ;
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_eqDetermines(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_eqDetermines___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1_value;
static const lean_string_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3_value;
static const lean_string_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__6 = (const lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Mathlib.Tactic.Simproc.ExistsAndEq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "_private.Mathlib.Tactic.Simproc.ExistsAndEq.0.ExistsAndEq.findEq.go"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "path is empty, but `P` is not an equality: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 72, .m_capacity = 72, .m_length = 71, .m_data = "some side of equality must be `a`, and the other must not depend on `a`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "path starts with andLeft, but `P` is not a conjunction"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "not implemented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "path starts with `existsBody`, but `P` is not `Exists`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_partition_loop___at___00ExistsAndEq_Path_forResult_spec__0(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_ExistsAndEq_Path_forResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_ExistsAndEq_Path_forResult___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_Path_forResult___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_Path_forResult(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ExistsAndEq_destruct___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "path starts with `andLeft`, but `P` is not a conjunction"};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___closed__1_value;
static const lean_string_object lp_mathlib_ExistsAndEq_destruct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "ExistsAndEq.destruct"};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___closed__0_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_destruct___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_destruct___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__0___boxed(lean_object**);
static const lean_string_object lp_mathlib_ExistsAndEq_destruct___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "path starts with `andRight`, but `P` is not a conjunction"};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___closed__3 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___closed__3_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_destruct___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_destruct___closed__4;
static const lean_string_object lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 29, 251, 16, 162, 14, 18, 105)}};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__2___boxed(lean_object**);
static lean_once_cell_t lp_mathlib_ExistsAndEq_destruct___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_destruct___closed__5;
static const lean_string_object lp_mathlib_ExistsAndEq_destruct___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "path starts with `existsBody`, but `exs` is empty"};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___closed__6 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___closed__6_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_destruct___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_destruct___closed__7;
static lean_once_cell_t lp_mathlib_ExistsAndEq_destruct___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_destruct___closed__8;
static const lean_ctor_object lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1_value_aux_0),((lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 192, 106, 76, 156, 25, 23, 244)}};
static const lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "ExistsAndEq.construct"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__0_value;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "path is empty, but the goal is not an equation"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__2;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__3 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__3_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_construct___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__3_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__4 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__4_value;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "path starts with `andLeft`, but the goal is not a conjunction"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__5 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__5_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__6;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "path starts with `andLeft`, but `leaves` is empty"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__7 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__7_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__8;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__9 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__9_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_construct___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_ExistsAndEq_construct___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__10_value_aux_0),((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__9_value),LEAN_SCALAR_PTR_LITERAL(58, 46, 244, 208, 18, 71, 77, 162)}};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__10 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__10_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__11;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "path starts with `andRight`, but the goal is not a conjunction"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__12 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__12_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__13;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "path starts with `andRight`, but `leaves` is empty"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__14 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__14_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__15;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__16;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__17;
static const lean_string_object lp_mathlib_ExistsAndEq_construct___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "path starts with `existsBody`, but the goal is not `Exists`"};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__18 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__18_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_construct___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_construct___closed__19;
static const lean_ctor_object lp_mathlib_ExistsAndEq_construct___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_mathlib_ExistsAndEq_construct___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__20_value_aux_0),((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__9_value),LEAN_SCALAR_PTR_LITERAL(74, 55, 158, 60, 144, 34, 77, 172)}};
static const lean_object* lp_mathlib_ExistsAndEq_construct___closed__20 = (const lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "ExistsAndEq.mkBeforeToAfter"};
static const lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "the end of the path is not an equation"};
static const lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5___boxed(lean_object**);
static const lean_string_object lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__0_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2;
static lean_once_cell_t lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3;
static lean_once_cell_t lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "propext"};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 150, 49, 30, 125, 3, 39, 172)}};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2;
static const lean_string_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__3 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_ExistsAndEq_construct___closed__9_value),LEAN_SCALAR_PTR_LITERAL(176, 155, 85, 49, 105, 137, 67, 168)}};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "ExistsAndEq.existsAndEqCore"};
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__1 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ExistsAndEq_existsAndEq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ExistsAndEq_existsAndEqCore___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ExistsAndEq_existsAndEq___redArg___closed__0 = (const lean_object*)&lp_mathlib_ExistsAndEq_existsAndEq___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
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
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
default: 
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorIdx___boxed(lean_object* v_x_6_){
_start:
{
uint8_t v_x_boxed_7_; lean_object* v_res_8_; 
v_x_boxed_7_ = lean_unbox(v_x_6_);
v_res_8_ = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(v_x_boxed_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(lean_object* v_k_9_){
_start:
{
lean_inc(v_k_9_);
return v_k_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg___boxed(lean_object* v_k_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_ExistsAndEq_GoTo_ctorElim___redArg(v_k_10_);
lean_dec(v_k_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim(lean_object* v_motive_12_, lean_object* v_ctorIdx_13_, uint8_t v_t_14_, lean_object* v_h_15_, lean_object* v_k_16_){
_start:
{
lean_inc(v_k_16_);
return v_k_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_ctorElim___boxed(lean_object* v_motive_17_, lean_object* v_ctorIdx_18_, lean_object* v_t_19_, lean_object* v_h_20_, lean_object* v_k_21_){
_start:
{
uint8_t v_t_boxed_22_; lean_object* v_res_23_; 
v_t_boxed_22_ = lean_unbox(v_t_19_);
v_res_23_ = lp_mathlib_ExistsAndEq_GoTo_ctorElim(v_motive_17_, v_ctorIdx_18_, v_t_boxed_22_, v_h_20_, v_k_21_);
lean_dec(v_k_21_);
lean_dec(v_ctorIdx_18_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___redArg(lean_object* v_andLeft_24_){
_start:
{
lean_inc(v_andLeft_24_);
return v_andLeft_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___redArg___boxed(lean_object* v_andLeft_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___redArg(v_andLeft_25_);
lean_dec(v_andLeft_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim(lean_object* v_motive_27_, uint8_t v_t_28_, lean_object* v_h_29_, lean_object* v_andLeft_30_){
_start:
{
lean_inc(v_andLeft_30_);
return v_andLeft_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andLeft_elim___boxed(lean_object* v_motive_31_, lean_object* v_t_32_, lean_object* v_h_33_, lean_object* v_andLeft_34_){
_start:
{
uint8_t v_t_boxed_35_; lean_object* v_res_36_; 
v_t_boxed_35_ = lean_unbox(v_t_32_);
v_res_36_ = lp_mathlib_ExistsAndEq_GoTo_andLeft_elim(v_motive_31_, v_t_boxed_35_, v_h_33_, v_andLeft_34_);
lean_dec(v_andLeft_34_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___redArg(lean_object* v_andRight_37_){
_start:
{
lean_inc(v_andRight_37_);
return v_andRight_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___redArg___boxed(lean_object* v_andRight_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_ExistsAndEq_GoTo_andRight_elim___redArg(v_andRight_38_);
lean_dec(v_andRight_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim(lean_object* v_motive_40_, uint8_t v_t_41_, lean_object* v_h_42_, lean_object* v_andRight_43_){
_start:
{
lean_inc(v_andRight_43_);
return v_andRight_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_andRight_elim___boxed(lean_object* v_motive_44_, lean_object* v_t_45_, lean_object* v_h_46_, lean_object* v_andRight_47_){
_start:
{
uint8_t v_t_boxed_48_; lean_object* v_res_49_; 
v_t_boxed_48_ = lean_unbox(v_t_45_);
v_res_49_ = lp_mathlib_ExistsAndEq_GoTo_andRight_elim(v_motive_44_, v_t_boxed_48_, v_h_46_, v_andRight_47_);
lean_dec(v_andRight_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___redArg(lean_object* v_existsType_50_){
_start:
{
lean_inc(v_existsType_50_);
return v_existsType_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___redArg___boxed(lean_object* v_existsType_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_ExistsAndEq_GoTo_existsType_elim___redArg(v_existsType_51_);
lean_dec(v_existsType_51_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim(lean_object* v_motive_53_, uint8_t v_t_54_, lean_object* v_h_55_, lean_object* v_existsType_56_){
_start:
{
lean_inc(v_existsType_56_);
return v_existsType_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsType_elim___boxed(lean_object* v_motive_57_, lean_object* v_t_58_, lean_object* v_h_59_, lean_object* v_existsType_60_){
_start:
{
uint8_t v_t_boxed_61_; lean_object* v_res_62_; 
v_t_boxed_61_ = lean_unbox(v_t_58_);
v_res_62_ = lp_mathlib_ExistsAndEq_GoTo_existsType_elim(v_motive_57_, v_t_boxed_61_, v_h_59_, v_existsType_60_);
lean_dec(v_existsType_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___redArg(lean_object* v_existsBody_63_){
_start:
{
lean_inc(v_existsBody_63_);
return v_existsBody_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___redArg___boxed(lean_object* v_existsBody_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___redArg(v_existsBody_64_);
lean_dec(v_existsBody_64_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim(lean_object* v_motive_66_, uint8_t v_t_67_, lean_object* v_h_68_, lean_object* v_existsBody_69_){
_start:
{
lean_inc(v_existsBody_69_);
return v_existsBody_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_GoTo_existsBody_elim___boxed(lean_object* v_motive_70_, lean_object* v_t_71_, lean_object* v_h_72_, lean_object* v_existsBody_73_){
_start:
{
uint8_t v_t_boxed_74_; lean_object* v_res_75_; 
v_t_boxed_74_ = lean_unbox(v_t_71_);
v_res_75_ = lp_mathlib_ExistsAndEq_GoTo_existsBody_elim(v_motive_70_, v_t_boxed_74_, v_h_72_, v_existsBody_73_);
lean_dec(v_existsBody_73_);
return v_res_75_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_instBEqGoTo_beq(uint8_t v_x_76_, uint8_t v_y_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_78_ = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(v_x_76_);
v___x_79_ = lp_mathlib_ExistsAndEq_GoTo_ctorIdx(v_y_77_);
v___x_80_ = lean_nat_dec_eq(v___x_78_, v___x_79_);
lean_dec(v___x_79_);
lean_dec(v___x_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_instBEqGoTo_beq___boxed(lean_object* v_x_81_, lean_object* v_y_82_){
_start:
{
uint8_t v_x_17__boxed_83_; uint8_t v_y_18__boxed_84_; uint8_t v_res_85_; lean_object* v_r_86_; 
v_x_17__boxed_83_ = lean_unbox(v_x_81_);
v_y_18__boxed_84_ = lean_unbox(v_y_82_);
v_res_85_ = lp_mathlib_ExistsAndEq_instBEqGoTo_beq(v_x_17__boxed_83_, v_y_18__boxed_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
static uint8_t _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo_default(void){
_start:
{
uint8_t v___x_89_; 
v___x_89_ = 0;
return v___x_89_;
}
}
static uint8_t _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo(void){
_start:
{
uint8_t v___x_90_; 
v___x_90_ = 0;
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = lean_box(0);
v___x_95_ = ((lean_object*)(lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__1));
v___x_96_ = l_Lean_Expr_const___override(v___x_95_, v___x_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = lean_obj_once(&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2, &lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2_once, _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__2);
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_99_ = lean_obj_once(&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3, &lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3_once, _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3);
v___x_100_ = lean_box(0);
v___x_101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v___x_99_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ(void){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lean_obj_once(&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4, &lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4_once, _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__4);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_instInhabitedHypQ(void){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_obj_once(&lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3, &lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3_once, _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ___closed__3);
return v___x_103_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ExistsAndEq_eqDetermines(lean_object* v_a_104_, lean_object* v_x_105_, lean_object* v_y_106_){
_start:
{
uint8_t v___x_107_; 
v___x_107_ = lean_expr_eqv(v_a_104_, v_x_105_);
if (v___x_107_ == 0)
{
return v___x_107_;
}
else
{
lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_108_ = l_Lean_Expr_fvarId_x21(v_a_104_);
v___x_109_ = l_Lean_Expr_containsFVar(v_y_106_, v___x_108_);
lean_dec(v___x_108_);
if (v___x_109_ == 0)
{
return v___x_107_;
}
else
{
uint8_t v___x_110_; 
v___x_110_ = 0;
return v___x_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_eqDetermines___boxed(lean_object* v_a_111_, lean_object* v_x_112_, lean_object* v_y_113_){
_start:
{
uint8_t v_res_114_; lean_object* v_r_115_; 
v_res_114_ = lp_mathlib_ExistsAndEq_eqDetermines(v_a_111_, v_x_112_, v_y_113_);
lean_dec_ref(v_y_113_);
lean_dec_ref(v_x_112_);
lean_dec_ref(v_a_111_);
v_r_115_ = lean_box(v_res_114_);
return v_r_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg(lean_object* v_a_127_, lean_object* v_P_128_, lean_object* v_a_129_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_P_128_, v_a_129_);
if (lean_obj_tag(v___x_131_) == 0)
{
lean_object* v_a_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_238_; 
v_a_132_ = lean_ctor_get(v___x_131_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_131_);
if (v_isSharedCheck_238_ == 0)
{
v___x_134_ = v___x_131_;
v_isShared_135_ = v_isSharedCheck_238_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_a_132_);
lean_dec(v___x_131_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_238_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_144_; uint8_t v___x_145_; 
v___x_144_ = l_Lean_Expr_cleanupAnnotations(v_a_132_);
v___x_145_ = l_Lean_Expr_isApp(v___x_144_);
if (v___x_145_ == 0)
{
lean_dec_ref(v___x_144_);
goto v___jp_136_;
}
else
{
lean_object* v_arg_146_; lean_object* v___y_148_; lean_object* v___x_171_; uint8_t v___x_172_; 
v_arg_146_ = lean_ctor_get(v___x_144_, 1);
lean_inc_ref(v_arg_146_);
v___x_171_ = l_Lean_Expr_appFnCleanup___redArg(v___x_144_);
v___x_172_ = l_Lean_Expr_isApp(v___x_171_);
if (v___x_172_ == 0)
{
lean_dec_ref(v___x_171_);
lean_dec_ref(v_arg_146_);
goto v___jp_136_;
}
else
{
lean_object* v_arg_173_; lean_object* v___x_174_; lean_object* v___x_175_; uint8_t v___x_176_; 
v_arg_173_ = lean_ctor_get(v___x_171_, 1);
lean_inc_ref(v_arg_173_);
v___x_174_ = l_Lean_Expr_appFnCleanup___redArg(v___x_171_);
v___x_175_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_176_ = l_Lean_Expr_isConstOf(v___x_174_, v___x_175_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_177_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3));
v___x_178_ = l_Lean_Expr_isConstOf(v___x_174_, v___x_177_);
if (v___x_178_ == 0)
{
uint8_t v___x_179_; 
v___x_179_ = l_Lean_Expr_isApp(v___x_174_);
if (v___x_179_ == 0)
{
lean_dec_ref(v___x_174_);
lean_dec_ref(v_arg_173_);
lean_dec_ref(v_arg_146_);
goto v___jp_136_;
}
else
{
lean_object* v___x_180_; lean_object* v___x_181_; uint8_t v___x_182_; 
v___x_180_ = l_Lean_Expr_appFnCleanup___redArg(v___x_174_);
v___x_181_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5));
v___x_182_ = l_Lean_Expr_isConstOf(v___x_180_, v___x_181_);
lean_dec_ref(v___x_180_);
if (v___x_182_ == 0)
{
lean_dec_ref(v_arg_173_);
lean_dec_ref(v_arg_146_);
goto v___jp_136_;
}
else
{
uint8_t v___x_183_; 
lean_del_object(v___x_134_);
v___x_183_ = lp_mathlib_ExistsAndEq_eqDetermines(v_a_127_, v_arg_173_, v_arg_146_);
if (v___x_183_ == 0)
{
uint8_t v___x_184_; 
v___x_184_ = lp_mathlib_ExistsAndEq_eqDetermines(v_a_127_, v_arg_146_, v_arg_173_);
lean_dec_ref(v_arg_173_);
lean_dec_ref(v_arg_146_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_185_ = lean_box(0);
v___x_186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
return v___x_186_;
}
else
{
lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_187_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__6));
v___x_188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
return v___x_188_;
}
}
else
{
lean_object* v___x_189_; lean_object* v___x_190_; 
lean_dec_ref(v_arg_173_);
lean_dec_ref(v_arg_146_);
v___x_189_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__6));
v___x_190_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
return v___x_190_;
}
}
}
}
else
{
lean_object* v___x_191_; 
lean_dec_ref(v___x_174_);
lean_del_object(v___x_134_);
v___x_191_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v_a_127_, v_arg_173_, v_a_129_);
if (lean_obj_tag(v___x_191_) == 0)
{
lean_object* v_a_192_; 
v_a_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_a_192_);
if (lean_obj_tag(v_a_192_) == 0)
{
v___y_148_ = v___x_191_;
goto v___jp_147_;
}
else
{
lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_210_; 
lean_dec_ref(v_arg_146_);
v_isSharedCheck_210_ = !lean_is_exclusive(v___x_191_);
if (v_isSharedCheck_210_ == 0)
{
lean_object* v_unused_211_; 
v_unused_211_ = lean_ctor_get(v___x_191_, 0);
lean_dec(v_unused_211_);
v___x_194_ = v___x_191_;
v_isShared_195_ = v_isSharedCheck_210_;
goto v_resetjp_193_;
}
else
{
lean_dec(v___x_191_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_210_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v_val_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_209_; 
v_val_196_ = lean_ctor_get(v_a_192_, 0);
v_isSharedCheck_209_ = !lean_is_exclusive(v_a_192_);
if (v_isSharedCheck_209_ == 0)
{
v___x_198_ = v_a_192_;
v_isShared_199_ = v_isSharedCheck_209_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_val_196_);
lean_dec(v_a_192_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_209_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
uint8_t v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_204_; 
v___x_200_ = 0;
v___x_201_ = lean_box(v___x_200_);
v___x_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_val_196_);
if (v_isShared_199_ == 0)
{
lean_ctor_set(v___x_198_, 0, v___x_202_);
v___x_204_ = v___x_198_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_202_);
v___x_204_ = v_reuseFailAlloc_208_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
lean_object* v___x_206_; 
if (v_isShared_195_ == 0)
{
lean_ctor_set(v___x_194_, 0, v___x_204_);
v___x_206_ = v___x_194_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_204_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
}
else
{
v___y_148_ = v___x_191_;
goto v___jp_147_;
}
}
}
else
{
lean_object* v___x_212_; uint8_t v___x_213_; 
lean_dec_ref(v___x_174_);
lean_del_object(v___x_134_);
v___x_212_ = l_Lean_Expr_fvarId_x21(v_a_127_);
v___x_213_ = l_Lean_Expr_containsFVar(v_arg_173_, v___x_212_);
lean_dec(v___x_212_);
lean_dec_ref(v_arg_173_);
if (v___x_213_ == 0)
{
if (v___x_176_ == 0)
{
lean_dec_ref(v_arg_146_);
goto v___jp_141_;
}
else
{
if (lean_obj_tag(v_arg_146_) == 6)
{
lean_object* v_body_214_; lean_object* v___x_215_; 
v_body_214_ = lean_ctor_get(v_arg_146_, 2);
lean_inc_ref(v_body_214_);
lean_dec_ref_known(v_arg_146_, 3);
v___x_215_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v_a_127_, v_body_214_, v_a_129_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
lean_inc(v_a_216_);
if (lean_obj_tag(v_a_216_) == 0)
{
return v___x_215_;
}
else
{
lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_234_; 
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_234_ == 0)
{
lean_object* v_unused_235_; 
v_unused_235_ = lean_ctor_get(v___x_215_, 0);
lean_dec(v_unused_235_);
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_234_;
goto v_resetjp_217_;
}
else
{
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_234_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v_val_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_233_; 
v_val_220_ = lean_ctor_get(v_a_216_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v_a_216_);
if (v_isSharedCheck_233_ == 0)
{
v___x_222_ = v_a_216_;
v_isShared_223_ = v_isSharedCheck_233_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_val_220_);
lean_dec(v_a_216_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_233_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
uint8_t v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_228_; 
v___x_224_ = 3;
v___x_225_ = lean_box(v___x_224_);
v___x_226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
lean_ctor_set(v___x_226_, 1, v_val_220_);
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 0, v___x_226_);
v___x_228_ = v___x_222_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v___x_226_);
v___x_228_ = v_reuseFailAlloc_232_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
lean_object* v___x_230_; 
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_228_);
v___x_230_ = v___x_218_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v___x_228_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
}
}
else
{
return v___x_215_;
}
}
else
{
lean_object* v___x_236_; lean_object* v___x_237_; 
lean_dec_ref(v_arg_146_);
v___x_236_ = lean_box(0);
v___x_237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
return v___x_237_;
}
}
}
else
{
lean_dec_ref(v_arg_146_);
goto v___jp_141_;
}
}
}
v___jp_147_:
{
if (lean_obj_tag(v___y_148_) == 0)
{
lean_object* v_a_149_; 
v_a_149_ = lean_ctor_get(v___y_148_, 0);
if (lean_obj_tag(v_a_149_) == 0)
{
lean_object* v___x_150_; 
lean_dec_ref_known(v___y_148_, 1);
v___x_150_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v_a_127_, v_arg_146_, v_a_129_);
if (lean_obj_tag(v___x_150_) == 0)
{
lean_object* v_a_151_; 
v_a_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc(v_a_151_);
if (lean_obj_tag(v_a_151_) == 0)
{
return v___x_150_;
}
else
{
lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_169_; 
v_isSharedCheck_169_ = !lean_is_exclusive(v___x_150_);
if (v_isSharedCheck_169_ == 0)
{
lean_object* v_unused_170_; 
v_unused_170_ = lean_ctor_get(v___x_150_, 0);
lean_dec(v_unused_170_);
v___x_153_ = v___x_150_;
v_isShared_154_ = v_isSharedCheck_169_;
goto v_resetjp_152_;
}
else
{
lean_dec(v___x_150_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_169_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v_val_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_168_; 
v_val_155_ = lean_ctor_get(v_a_151_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v_a_151_);
if (v_isSharedCheck_168_ == 0)
{
v___x_157_ = v_a_151_;
v_isShared_158_ = v_isSharedCheck_168_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_val_155_);
lean_dec(v_a_151_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_168_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
uint8_t v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_163_; 
v___x_159_ = 1;
v___x_160_ = lean_box(v___x_159_);
v___x_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v_val_155_);
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v___x_161_);
v___x_163_ = v___x_157_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v___x_161_);
v___x_163_ = v_reuseFailAlloc_167_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
lean_object* v___x_165_; 
if (v_isShared_154_ == 0)
{
lean_ctor_set(v___x_153_, 0, v___x_163_);
v___x_165_ = v___x_153_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
}
}
else
{
return v___x_150_;
}
}
else
{
lean_dec_ref(v_arg_146_);
return v___y_148_;
}
}
else
{
lean_dec_ref(v_arg_146_);
return v___y_148_;
}
}
}
v___jp_136_:
{
lean_object* v___x_137_; lean_object* v___x_139_; 
v___x_137_ = lean_box(0);
if (v_isShared_135_ == 0)
{
lean_ctor_set(v___x_134_, 0, v___x_137_);
v___x_139_ = v___x_134_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v___x_137_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
v___jp_141_:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = lean_box(0);
v___x_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
return v___x_143_;
}
}
}
else
{
lean_object* v_a_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_246_; 
v_a_239_ = lean_ctor_get(v___x_131_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_131_);
if (v_isSharedCheck_246_ == 0)
{
v___x_241_ = v___x_131_;
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_a_239_);
lean_dec(v___x_131_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_244_; 
if (v_isShared_242_ == 0)
{
v___x_244_ = v___x_241_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_a_239_);
v___x_244_ = v_reuseFailAlloc_245_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
return v___x_244_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___redArg___boxed(lean_object* v_a_247_, lean_object* v_P_248_, lean_object* v_a_249_, lean_object* v_a_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v_a_247_, v_P_248_, v_a_249_);
lean_dec(v_a_249_);
lean_dec_ref(v_a_247_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath(lean_object* v_u_252_, lean_object* v_00_u03b1_253_, lean_object* v_a_254_, lean_object* v_P_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v_a_254_, v_P_255_, v_a_257_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEqPath___boxed(lean_object* v_u_262_, lean_object* v_00_u03b1_263_, lean_object* v_a_264_, lean_object* v_P_265_, lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_ExistsAndEq_findEqPath(v_u_262_, v_00_u03b1_263_, v_a_264_, v_P_265_, v_a_266_, v_a_267_, v_a_268_, v_a_269_);
lean_dec(v_a_269_);
lean_dec_ref(v_a_268_);
lean_dec(v_a_267_);
lean_dec_ref(v_a_266_);
lean_dec_ref(v_a_264_);
lean_dec_ref(v_00_u03b1_263_);
lean_dec(v_u_262_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(lean_object* v_l_272_, lean_object* v___y_273_){
_start:
{
lean_object* v___x_275_; lean_object* v_mctx_276_; lean_object* v___x_277_; lean_object* v_fst_278_; lean_object* v_snd_279_; lean_object* v___x_280_; lean_object* v_cache_281_; lean_object* v_zetaDeltaFVarIds_282_; lean_object* v_postponed_283_; lean_object* v_diag_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_293_; 
v___x_275_ = lean_st_ref_get(v___y_273_);
v_mctx_276_ = lean_ctor_get(v___x_275_, 0);
lean_inc_ref(v_mctx_276_);
lean_dec(v___x_275_);
v___x_277_ = lean_instantiate_level_mvars(v_mctx_276_, v_l_272_);
v_fst_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_fst_278_);
v_snd_279_ = lean_ctor_get(v___x_277_, 1);
lean_inc(v_snd_279_);
lean_dec_ref(v___x_277_);
v___x_280_ = lean_st_ref_take(v___y_273_);
v_cache_281_ = lean_ctor_get(v___x_280_, 1);
v_zetaDeltaFVarIds_282_ = lean_ctor_get(v___x_280_, 2);
v_postponed_283_ = lean_ctor_get(v___x_280_, 3);
v_diag_284_ = lean_ctor_get(v___x_280_, 4);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_280_);
if (v_isSharedCheck_293_ == 0)
{
lean_object* v_unused_294_; 
v_unused_294_ = lean_ctor_get(v___x_280_, 0);
lean_dec(v_unused_294_);
v___x_286_ = v___x_280_;
v_isShared_287_ = v_isSharedCheck_293_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_diag_284_);
lean_inc(v_postponed_283_);
lean_inc(v_zetaDeltaFVarIds_282_);
lean_inc(v_cache_281_);
lean_dec(v___x_280_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_293_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 0, v_fst_278_);
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_fst_278_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v_cache_281_);
lean_ctor_set(v_reuseFailAlloc_292_, 2, v_zetaDeltaFVarIds_282_);
lean_ctor_set(v_reuseFailAlloc_292_, 3, v_postponed_283_);
lean_ctor_set(v_reuseFailAlloc_292_, 4, v_diag_284_);
v___x_289_ = v_reuseFailAlloc_292_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_290_ = lean_st_ref_set(v___y_273_, v___x_289_);
v___x_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_291_, 0, v_snd_279_);
return v___x_291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg___boxed(lean_object* v_l_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(v_l_295_, v___y_296_);
lean_dec(v___y_296_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(lean_object* v_l_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(v_l_299_, v___y_301_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___boxed(lean_object* v_l_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0(v_l_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(lean_object* v_e_313_, lean_object* v___y_314_){
_start:
{
uint8_t v___x_316_; 
v___x_316_ = l_Lean_Expr_hasMVar(v_e_313_);
if (v___x_316_ == 0)
{
lean_object* v___x_317_; 
v___x_317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_317_, 0, v_e_313_);
return v___x_317_;
}
else
{
lean_object* v___x_318_; lean_object* v_mctx_319_; lean_object* v___x_320_; lean_object* v_fst_321_; lean_object* v_snd_322_; lean_object* v___x_323_; lean_object* v_cache_324_; lean_object* v_zetaDeltaFVarIds_325_; lean_object* v_postponed_326_; lean_object* v_diag_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_336_; 
v___x_318_ = lean_st_ref_get(v___y_314_);
v_mctx_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc_ref(v_mctx_319_);
lean_dec(v___x_318_);
v___x_320_ = l_Lean_instantiateMVarsCore(v_mctx_319_, v_e_313_);
v_fst_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc(v_fst_321_);
v_snd_322_ = lean_ctor_get(v___x_320_, 1);
lean_inc(v_snd_322_);
lean_dec_ref(v___x_320_);
v___x_323_ = lean_st_ref_take(v___y_314_);
v_cache_324_ = lean_ctor_get(v___x_323_, 1);
v_zetaDeltaFVarIds_325_ = lean_ctor_get(v___x_323_, 2);
v_postponed_326_ = lean_ctor_get(v___x_323_, 3);
v_diag_327_ = lean_ctor_get(v___x_323_, 4);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_336_ == 0)
{
lean_object* v_unused_337_; 
v_unused_337_ = lean_ctor_get(v___x_323_, 0);
lean_dec(v_unused_337_);
v___x_329_ = v___x_323_;
v_isShared_330_ = v_isSharedCheck_336_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_diag_327_);
lean_inc(v_postponed_326_);
lean_inc(v_zetaDeltaFVarIds_325_);
lean_inc(v_cache_324_);
lean_dec(v___x_323_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_336_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_332_; 
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 0, v_snd_322_);
v___x_332_ = v___x_329_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_snd_322_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v_cache_324_);
lean_ctor_set(v_reuseFailAlloc_335_, 2, v_zetaDeltaFVarIds_325_);
lean_ctor_set(v_reuseFailAlloc_335_, 3, v_postponed_326_);
lean_ctor_set(v_reuseFailAlloc_335_, 4, v_diag_327_);
v___x_332_ = v_reuseFailAlloc_335_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_333_ = lean_st_ref_set(v___y_314_, v___x_332_);
v___x_334_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_334_, 0, v_fst_321_);
return v___x_334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg___boxed(lean_object* v_e_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_e_338_, v___y_339_);
lean_dec(v___y_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(lean_object* v_e_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_e_342_, v___y_344_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___boxed(lean_object* v_e_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1(v_e_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(lean_object* v_k_356_, uint8_t v_allowLevelAssignments_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_357_, v_k_356_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_371_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_371_ == 0)
{
v___x_366_ = v___x_363_;
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_363_);
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
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_379_; 
v_a_372_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_379_ == 0)
{
v___x_374_ = v___x_363_;
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_a_372_);
lean_dec(v___x_363_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v___x_377_; 
if (v_isShared_375_ == 0)
{
v___x_377_ = v___x_374_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_a_372_);
v___x_377_ = v_reuseFailAlloc_378_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
return v___x_377_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg___boxed(lean_object* v_k_380_, lean_object* v_allowLevelAssignments_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_387_; lean_object* v_res_388_; 
v_allowLevelAssignments_boxed_387_ = lean_unbox(v_allowLevelAssignments_381_);
v_res_388_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v_k_380_, v_allowLevelAssignments_boxed_387_, v___y_382_, v___y_383_, v___y_384_, v___y_385_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(lean_object* v_00_u03b1_389_, lean_object* v_k_390_, uint8_t v_allowLevelAssignments_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v_k_390_, v_allowLevelAssignments_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___boxed(lean_object* v_00_u03b1_398_, lean_object* v_k_399_, lean_object* v_allowLevelAssignments_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_406_; lean_object* v_res_407_; 
v_allowLevelAssignments_boxed_406_ = lean_unbox(v_allowLevelAssignments_400_);
v_res_407_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2(v_00_u03b1_398_, v_k_399_, v_allowLevelAssignments_boxed_406_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(lean_object* v_msg_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_){
_start:
{
lean_object* v___f_415_; lean_object* v___x_9161__overap_416_; lean_object* v___x_417_; 
v___f_415_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0));
v___x_9161__overap_416_ = lean_panic_fn_borrowed(v___f_415_, v_msg_409_);
lean_inc(v___y_413_);
lean_inc_ref(v___y_412_);
lean_inc(v___y_411_);
lean_inc_ref(v___y_410_);
v___x_417_ = lean_apply_5(v___x_9161__overap_416_, v___y_410_, v___y_411_, v___y_412_, v___y_413_, lean_box(0));
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___boxed(lean_object* v_msg_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v_msg_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(lean_object* v_k_425_, lean_object* v_b_426_, lean_object* v_c_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_){
_start:
{
lean_object* v___x_433_; 
lean_inc(v___y_431_);
lean_inc_ref(v___y_430_);
lean_inc(v___y_429_);
lean_inc_ref(v___y_428_);
v___x_433_ = lean_apply_7(v_k_425_, v_b_426_, v_c_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, lean_box(0));
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed(lean_object* v_k_434_, lean_object* v_b_435_, lean_object* v_c_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0(v_k_434_, v_b_435_, v_c_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(lean_object* v_e_443_, lean_object* v_maxFVars_444_, lean_object* v_k_445_, uint8_t v_cleanupAnnotations_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v___f_452_; uint8_t v___x_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___f_452_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_452_, 0, v_k_445_);
v___x_453_ = 1;
v___x_454_ = 0;
v___x_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_455_, 0, v_maxFVars_444_);
v___x_456_ = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), v_e_443_, v___x_453_, v___x_454_, v___x_453_, v___x_454_, v___x_455_, v___f_452_, v_cleanupAnnotations_446_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
lean_dec_ref_known(v___x_455_, 1);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_464_; 
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_464_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_464_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v_a_457_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
else
{
lean_object* v_a_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_472_; 
v_a_465_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_472_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_472_ == 0)
{
v___x_467_ = v___x_456_;
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_a_465_);
lean_dec(v___x_456_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___x_470_; 
if (v_isShared_468_ == 0)
{
v___x_470_ = v___x_467_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_a_465_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg___boxed(lean_object* v_e_473_, lean_object* v_maxFVars_474_, lean_object* v_k_475_, lean_object* v_cleanupAnnotations_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_482_; lean_object* v_res_483_; 
v_cleanupAnnotations_boxed_482_ = lean_unbox(v_cleanupAnnotations_476_);
v_res_483_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(v_e_473_, v_maxFVars_474_, v_k_475_, v_cleanupAnnotations_boxed_482_, v___y_477_, v___y_478_, v___y_479_, v___y_480_);
lean_dec(v___y_480_);
lean_dec_ref(v___y_479_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
return v_res_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(lean_object* v_00_u03b1_484_, lean_object* v_e_485_, lean_object* v_maxFVars_486_, lean_object* v_k_487_, uint8_t v_cleanupAnnotations_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(v_e_485_, v_maxFVars_486_, v_k_487_, v_cleanupAnnotations_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___boxed(lean_object* v_00_u03b1_495_, lean_object* v_e_496_, lean_object* v_maxFVars_497_, lean_object* v_k_498_, lean_object* v_cleanupAnnotations_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_505_; lean_object* v_res_506_; 
v_cleanupAnnotations_boxed_505_ = lean_unbox(v_cleanupAnnotations_499_);
v_res_506_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4(v_00_u03b1_495_, v_e_496_, v_maxFVars_497_, v_k_498_, v_cleanupAnnotations_boxed_505_, v___y_500_, v___y_501_, v___y_502_, v___y_503_);
lean_dec(v___y_503_);
lean_dec_ref(v___y_502_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(lean_object* v___x_507_, uint8_t v___x_508_, lean_object* v___x_509_, lean_object* v_u_510_, lean_object* v_P_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v___x_517_; 
lean_inc(v___x_509_);
v___x_517_ = l_Lean_Meta_mkFreshExprMVar(v___x_507_, v___x_508_, v___x_509_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
if (lean_obj_tag(v___x_517_) == 0)
{
lean_object* v_a_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
v_a_518_ = lean_ctor_get(v___x_517_, 0);
lean_inc_n(v_a_518_, 2);
lean_dec_ref_known(v___x_517_, 1);
v___x_519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_519_, 0, v_a_518_);
lean_inc(v___x_509_);
lean_inc_ref(v___x_519_);
v___x_520_ = l_Lean_Meta_mkFreshExprMVar(v___x_519_, v___x_508_, v___x_509_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
if (lean_obj_tag(v___x_520_) == 0)
{
lean_object* v_a_521_; lean_object* v___x_522_; 
v_a_521_ = lean_ctor_get(v___x_520_, 0);
lean_inc(v_a_521_);
lean_dec_ref_known(v___x_520_, 1);
v___x_522_ = l_Lean_Meta_mkFreshExprMVar(v___x_519_, v___x_508_, v___x_509_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
if (lean_obj_tag(v___x_522_) == 0)
{
lean_object* v_a_523_; lean_object* v_keyedConfig_524_; uint8_t v_trackZetaDelta_525_; lean_object* v_zetaDeltaSet_526_; lean_object* v_lctx_527_; lean_object* v_localInstances_528_; lean_object* v_defEqCtx_x3f_529_; lean_object* v_synthPendingDepth_530_; lean_object* v_customCanUnfoldPredicate_x3f_531_; uint8_t v_univApprox_532_; uint8_t v_inTypeClassResolution_533_; uint8_t v_cacheInferType_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_611_; 
v_a_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_a_523_);
lean_dec_ref_known(v___x_522_, 1);
v_keyedConfig_524_ = lean_ctor_get(v___y_512_, 0);
v_trackZetaDelta_525_ = lean_ctor_get_uint8(v___y_512_, sizeof(void*)*7);
v_zetaDeltaSet_526_ = lean_ctor_get(v___y_512_, 1);
v_lctx_527_ = lean_ctor_get(v___y_512_, 2);
v_localInstances_528_ = lean_ctor_get(v___y_512_, 3);
v_defEqCtx_x3f_529_ = lean_ctor_get(v___y_512_, 4);
v_synthPendingDepth_530_ = lean_ctor_get(v___y_512_, 5);
v_customCanUnfoldPredicate_x3f_531_ = lean_ctor_get(v___y_512_, 6);
v_univApprox_532_ = lean_ctor_get_uint8(v___y_512_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_533_ = lean_ctor_get_uint8(v___y_512_, sizeof(void*)*7 + 2);
v_cacheInferType_534_ = lean_ctor_get_uint8(v___y_512_, sizeof(void*)*7 + 3);
v_isSharedCheck_611_ = !lean_is_exclusive(v___y_512_);
if (v_isSharedCheck_611_ == 0)
{
v___x_536_ = v___y_512_;
v_isShared_537_ = v_isSharedCheck_611_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_531_);
lean_inc(v_synthPendingDepth_530_);
lean_inc(v_defEqCtx_x3f_529_);
lean_inc(v_localInstances_528_);
lean_inc(v_lctx_527_);
lean_inc(v_zetaDeltaSet_526_);
lean_inc(v_keyedConfig_524_);
lean_dec(v___y_512_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_611_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; uint8_t v___x_545_; lean_object* v___x_546_; lean_object* v___x_548_; 
v___x_538_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5));
v___x_539_ = lean_box(0);
v___x_540_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_540_, 0, v_u_510_);
lean_ctor_set(v___x_540_, 1, v___x_539_);
v___x_541_ = l_Lean_Expr_const___override(v___x_538_, v___x_540_);
lean_inc(v_a_518_);
v___x_542_ = l_Lean_Expr_app___override(v___x_541_, v_a_518_);
lean_inc(v_a_521_);
v___x_543_ = l_Lean_Expr_app___override(v___x_542_, v_a_521_);
lean_inc(v_a_523_);
v___x_544_ = l_Lean_Expr_app___override(v___x_543_, v_a_523_);
v___x_545_ = 2;
v___x_546_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_545_, v_keyedConfig_524_);
if (v_isShared_537_ == 0)
{
lean_ctor_set(v___x_536_, 0, v___x_546_);
v___x_548_ = v___x_536_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_546_);
lean_ctor_set(v_reuseFailAlloc_610_, 1, v_zetaDeltaSet_526_);
lean_ctor_set(v_reuseFailAlloc_610_, 2, v_lctx_527_);
lean_ctor_set(v_reuseFailAlloc_610_, 3, v_localInstances_528_);
lean_ctor_set(v_reuseFailAlloc_610_, 4, v_defEqCtx_x3f_529_);
lean_ctor_set(v_reuseFailAlloc_610_, 5, v_synthPendingDepth_530_);
lean_ctor_set(v_reuseFailAlloc_610_, 6, v_customCanUnfoldPredicate_x3f_531_);
lean_ctor_set_uint8(v_reuseFailAlloc_610_, sizeof(void*)*7, v_trackZetaDelta_525_);
lean_ctor_set_uint8(v_reuseFailAlloc_610_, sizeof(void*)*7 + 1, v_univApprox_532_);
lean_ctor_set_uint8(v_reuseFailAlloc_610_, sizeof(void*)*7 + 2, v_inTypeClassResolution_533_);
lean_ctor_set_uint8(v_reuseFailAlloc_610_, sizeof(void*)*7 + 3, v_cacheInferType_534_);
v___x_548_ = v_reuseFailAlloc_610_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
lean_object* v___x_549_; 
v___x_549_ = l_Lean_Meta_isExprDefEq(v___x_544_, v_P_511_, v___x_548_, v___y_513_, v___y_514_, v___y_515_);
lean_dec_ref(v___x_548_);
if (lean_obj_tag(v___x_549_) == 0)
{
lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_601_; 
v_a_550_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_601_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_601_ == 0)
{
v___x_552_ = v___x_549_;
v_isShared_553_ = v_isSharedCheck_601_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_549_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_601_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
uint8_t v___x_554_; 
v___x_554_ = lean_unbox(v_a_550_);
if (v___x_554_ == 0)
{
lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_559_; 
v___x_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_555_, 0, v_a_523_);
lean_ctor_set(v___x_555_, 1, v_a_550_);
v___x_556_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_556_, 0, v_a_521_);
lean_ctor_set(v___x_556_, 1, v___x_555_);
v___x_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_557_, 0, v_a_518_);
lean_ctor_set(v___x_557_, 1, v___x_556_);
if (v_isShared_553_ == 0)
{
lean_ctor_set(v___x_552_, 0, v___x_557_);
v___x_559_ = v___x_552_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v___x_557_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
else
{
lean_object* v___x_561_; 
lean_del_object(v___x_552_);
v___x_561_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_518_, v___y_513_);
if (lean_obj_tag(v___x_561_) == 0)
{
lean_object* v_a_562_; lean_object* v___x_563_; 
v_a_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc(v_a_562_);
lean_dec_ref_known(v___x_561_, 1);
v___x_563_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_521_, v___y_513_);
if (lean_obj_tag(v___x_563_) == 0)
{
lean_object* v_a_564_; lean_object* v___x_565_; 
v_a_564_ = lean_ctor_get(v___x_563_, 0);
lean_inc(v_a_564_);
lean_dec_ref_known(v___x_563_, 1);
v___x_565_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_523_, v___y_513_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_576_; 
v_a_566_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_576_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_576_ == 0)
{
v___x_568_ = v___x_565_;
v_isShared_569_ = v_isSharedCheck_576_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_565_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_576_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_574_; 
v___x_570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_570_, 0, v_a_566_);
lean_ctor_set(v___x_570_, 1, v_a_550_);
v___x_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_571_, 0, v_a_564_);
lean_ctor_set(v___x_571_, 1, v___x_570_);
v___x_572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_572_, 0, v_a_562_);
lean_ctor_set(v___x_572_, 1, v___x_571_);
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 0, v___x_572_);
v___x_574_ = v___x_568_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v___x_572_);
v___x_574_ = v_reuseFailAlloc_575_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
return v___x_574_;
}
}
}
else
{
lean_object* v_a_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_584_; 
lean_dec(v_a_564_);
lean_dec(v_a_562_);
lean_dec(v_a_550_);
v_a_577_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_584_ == 0)
{
v___x_579_ = v___x_565_;
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_a_577_);
lean_dec(v___x_565_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_584_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v___x_582_; 
if (v_isShared_580_ == 0)
{
v___x_582_ = v___x_579_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v_a_577_);
v___x_582_ = v_reuseFailAlloc_583_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
return v___x_582_;
}
}
}
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
lean_dec(v_a_562_);
lean_dec(v_a_550_);
lean_dec(v_a_523_);
v_a_585_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_563_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_563_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
else
{
lean_object* v_a_593_; lean_object* v___x_595_; uint8_t v_isShared_596_; uint8_t v_isSharedCheck_600_; 
lean_dec(v_a_550_);
lean_dec(v_a_523_);
lean_dec(v_a_521_);
v_a_593_ = lean_ctor_get(v___x_561_, 0);
v_isSharedCheck_600_ = !lean_is_exclusive(v___x_561_);
if (v_isSharedCheck_600_ == 0)
{
v___x_595_ = v___x_561_;
v_isShared_596_ = v_isSharedCheck_600_;
goto v_resetjp_594_;
}
else
{
lean_inc(v_a_593_);
lean_dec(v___x_561_);
v___x_595_ = lean_box(0);
v_isShared_596_ = v_isSharedCheck_600_;
goto v_resetjp_594_;
}
v_resetjp_594_:
{
lean_object* v___x_598_; 
if (v_isShared_596_ == 0)
{
v___x_598_ = v___x_595_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v_a_593_);
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
}
else
{
lean_object* v_a_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_609_; 
lean_dec(v_a_523_);
lean_dec(v_a_521_);
lean_dec(v_a_518_);
v_a_602_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_609_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_609_ == 0)
{
v___x_604_ = v___x_549_;
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_a_602_);
lean_dec(v___x_549_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
lean_object* v___x_607_; 
if (v_isShared_605_ == 0)
{
v___x_607_ = v___x_604_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_a_602_);
v___x_607_ = v_reuseFailAlloc_608_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
return v___x_607_;
}
}
}
}
}
}
else
{
lean_object* v_a_612_; lean_object* v___x_614_; uint8_t v_isShared_615_; uint8_t v_isSharedCheck_619_; 
lean_dec(v_a_521_);
lean_dec(v_a_518_);
lean_dec_ref(v___y_512_);
lean_dec_ref(v_P_511_);
lean_dec(v_u_510_);
v_a_612_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_619_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_619_ == 0)
{
v___x_614_ = v___x_522_;
v_isShared_615_ = v_isSharedCheck_619_;
goto v_resetjp_613_;
}
else
{
lean_inc(v_a_612_);
lean_dec(v___x_522_);
v___x_614_ = lean_box(0);
v_isShared_615_ = v_isSharedCheck_619_;
goto v_resetjp_613_;
}
v_resetjp_613_:
{
lean_object* v___x_617_; 
if (v_isShared_615_ == 0)
{
v___x_617_ = v___x_614_;
goto v_reusejp_616_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v_a_612_);
v___x_617_ = v_reuseFailAlloc_618_;
goto v_reusejp_616_;
}
v_reusejp_616_:
{
return v___x_617_;
}
}
}
}
else
{
lean_object* v_a_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
lean_dec_ref_known(v___x_519_, 1);
lean_dec(v_a_518_);
lean_dec_ref(v___y_512_);
lean_dec_ref(v_P_511_);
lean_dec(v_u_510_);
lean_dec(v___x_509_);
v_a_620_ = lean_ctor_get(v___x_520_, 0);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_520_);
if (v_isSharedCheck_627_ == 0)
{
v___x_622_ = v___x_520_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_a_620_);
lean_dec(v___x_520_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v_a_620_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
}
else
{
lean_object* v_a_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_635_; 
lean_dec_ref(v___y_512_);
lean_dec_ref(v_P_511_);
lean_dec(v_u_510_);
lean_dec(v___x_509_);
v_a_628_ = lean_ctor_get(v___x_517_, 0);
v_isSharedCheck_635_ = !lean_is_exclusive(v___x_517_);
if (v_isSharedCheck_635_ == 0)
{
v___x_630_ = v___x_517_;
v_isShared_631_ = v_isSharedCheck_635_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_a_628_);
lean_dec(v___x_517_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_635_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v___x_633_; 
if (v_isShared_631_ == 0)
{
v___x_633_ = v___x_630_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_634_; 
v_reuseFailAlloc_634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_634_, 0, v_a_628_);
v___x_633_ = v_reuseFailAlloc_634_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
return v___x_633_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed(lean_object* v___x_636_, lean_object* v___x_637_, lean_object* v___x_638_, lean_object* v_u_639_, lean_object* v_P_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
uint8_t v___x_10096__boxed_646_; lean_object* v_res_647_; 
v___x_10096__boxed_646_ = lean_unbox(v___x_637_);
v_res_647_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0(v___x_636_, v___x_10096__boxed_646_, v___x_638_, v_u_639_, v_P_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
lean_dec(v___y_642_);
return v_res_647_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_648_ = lean_box(0);
v___x_649_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__3));
v___x_650_ = l_Lean_Expr_const___override(v___x_649_, v___x_648_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(lean_object* v___x_651_, uint8_t v___x_652_, lean_object* v___x_653_, lean_object* v_P_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v___x_660_; 
lean_inc(v___x_653_);
lean_inc(v___x_651_);
v___x_660_ = l_Lean_Meta_mkFreshExprMVar(v___x_651_, v___x_652_, v___x_653_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
if (lean_obj_tag(v___x_660_) == 0)
{
lean_object* v_a_661_; lean_object* v___x_662_; 
v_a_661_ = lean_ctor_get(v___x_660_, 0);
lean_inc(v_a_661_);
lean_dec_ref_known(v___x_660_, 1);
v___x_662_ = l_Lean_Meta_mkFreshExprMVar(v___x_651_, v___x_652_, v___x_653_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
if (lean_obj_tag(v___x_662_) == 0)
{
lean_object* v_a_663_; lean_object* v_keyedConfig_664_; uint8_t v_trackZetaDelta_665_; lean_object* v_zetaDeltaSet_666_; lean_object* v_lctx_667_; lean_object* v_localInstances_668_; lean_object* v_defEqCtx_x3f_669_; lean_object* v_synthPendingDepth_670_; lean_object* v_customCanUnfoldPredicate_x3f_671_; uint8_t v_univApprox_672_; uint8_t v_inTypeClassResolution_673_; uint8_t v_cacheInferType_674_; lean_object* v___x_676_; uint8_t v_isShared_677_; uint8_t v_isSharedCheck_735_; 
v_a_663_ = lean_ctor_get(v___x_662_, 0);
lean_inc(v_a_663_);
lean_dec_ref_known(v___x_662_, 1);
v_keyedConfig_664_ = lean_ctor_get(v___y_655_, 0);
v_trackZetaDelta_665_ = lean_ctor_get_uint8(v___y_655_, sizeof(void*)*7);
v_zetaDeltaSet_666_ = lean_ctor_get(v___y_655_, 1);
v_lctx_667_ = lean_ctor_get(v___y_655_, 2);
v_localInstances_668_ = lean_ctor_get(v___y_655_, 3);
v_defEqCtx_x3f_669_ = lean_ctor_get(v___y_655_, 4);
v_synthPendingDepth_670_ = lean_ctor_get(v___y_655_, 5);
v_customCanUnfoldPredicate_x3f_671_ = lean_ctor_get(v___y_655_, 6);
v_univApprox_672_ = lean_ctor_get_uint8(v___y_655_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_673_ = lean_ctor_get_uint8(v___y_655_, sizeof(void*)*7 + 2);
v_cacheInferType_674_ = lean_ctor_get_uint8(v___y_655_, sizeof(void*)*7 + 3);
v_isSharedCheck_735_ = !lean_is_exclusive(v___y_655_);
if (v_isSharedCheck_735_ == 0)
{
v___x_676_ = v___y_655_;
v_isShared_677_ = v_isSharedCheck_735_;
goto v_resetjp_675_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_671_);
lean_inc(v_synthPendingDepth_670_);
lean_inc(v_defEqCtx_x3f_669_);
lean_inc(v_localInstances_668_);
lean_inc(v_lctx_667_);
lean_inc(v_zetaDeltaSet_666_);
lean_inc(v_keyedConfig_664_);
lean_dec(v___y_655_);
v___x_676_ = lean_box(0);
v_isShared_677_ = v_isSharedCheck_735_;
goto v_resetjp_675_;
}
v_resetjp_675_:
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; uint8_t v___x_681_; lean_object* v___x_682_; lean_object* v___x_684_; 
v___x_678_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0);
lean_inc(v_a_661_);
v___x_679_ = l_Lean_Expr_app___override(v___x_678_, v_a_661_);
lean_inc(v_a_663_);
v___x_680_ = l_Lean_Expr_app___override(v___x_679_, v_a_663_);
v___x_681_ = 2;
v___x_682_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_681_, v_keyedConfig_664_);
if (v_isShared_677_ == 0)
{
lean_ctor_set(v___x_676_, 0, v___x_682_);
v___x_684_ = v___x_676_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_734_; 
v_reuseFailAlloc_734_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_734_, 0, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_734_, 1, v_zetaDeltaSet_666_);
lean_ctor_set(v_reuseFailAlloc_734_, 2, v_lctx_667_);
lean_ctor_set(v_reuseFailAlloc_734_, 3, v_localInstances_668_);
lean_ctor_set(v_reuseFailAlloc_734_, 4, v_defEqCtx_x3f_669_);
lean_ctor_set(v_reuseFailAlloc_734_, 5, v_synthPendingDepth_670_);
lean_ctor_set(v_reuseFailAlloc_734_, 6, v_customCanUnfoldPredicate_x3f_671_);
lean_ctor_set_uint8(v_reuseFailAlloc_734_, sizeof(void*)*7, v_trackZetaDelta_665_);
lean_ctor_set_uint8(v_reuseFailAlloc_734_, sizeof(void*)*7 + 1, v_univApprox_672_);
lean_ctor_set_uint8(v_reuseFailAlloc_734_, sizeof(void*)*7 + 2, v_inTypeClassResolution_673_);
lean_ctor_set_uint8(v_reuseFailAlloc_734_, sizeof(void*)*7 + 3, v_cacheInferType_674_);
v___x_684_ = v_reuseFailAlloc_734_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
lean_object* v___x_685_; 
v___x_685_ = l_Lean_Meta_isExprDefEq(v___x_680_, v_P_654_, v___x_684_, v___y_656_, v___y_657_, v___y_658_);
lean_dec_ref(v___x_684_);
if (lean_obj_tag(v___x_685_) == 0)
{
lean_object* v_a_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_725_; 
v_a_686_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_725_ == 0)
{
v___x_688_ = v___x_685_;
v_isShared_689_ = v_isSharedCheck_725_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_a_686_);
lean_dec(v___x_685_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_725_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
uint8_t v___x_690_; 
v___x_690_ = lean_unbox(v_a_686_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_694_; 
v___x_691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_691_, 0, v_a_663_);
lean_ctor_set(v___x_691_, 1, v_a_686_);
v___x_692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_692_, 0, v_a_661_);
lean_ctor_set(v___x_692_, 1, v___x_691_);
if (v_isShared_689_ == 0)
{
lean_ctor_set(v___x_688_, 0, v___x_692_);
v___x_694_ = v___x_688_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_692_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
return v___x_694_;
}
}
else
{
lean_object* v___x_696_; 
lean_del_object(v___x_688_);
v___x_696_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_661_, v___y_656_);
if (lean_obj_tag(v___x_696_) == 0)
{
lean_object* v_a_697_; lean_object* v___x_698_; 
v_a_697_ = lean_ctor_get(v___x_696_, 0);
lean_inc(v_a_697_);
lean_dec_ref_known(v___x_696_, 1);
v___x_698_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_663_, v___y_656_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_object* v_a_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_708_; 
v_a_699_ = lean_ctor_get(v___x_698_, 0);
v_isSharedCheck_708_ = !lean_is_exclusive(v___x_698_);
if (v_isSharedCheck_708_ == 0)
{
v___x_701_ = v___x_698_;
v_isShared_702_ = v_isSharedCheck_708_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_a_699_);
lean_dec(v___x_698_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_708_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_706_; 
v___x_703_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_703_, 0, v_a_699_);
lean_ctor_set(v___x_703_, 1, v_a_686_);
v___x_704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_704_, 0, v_a_697_);
lean_ctor_set(v___x_704_, 1, v___x_703_);
if (v_isShared_702_ == 0)
{
lean_ctor_set(v___x_701_, 0, v___x_704_);
v___x_706_ = v___x_701_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_704_);
v___x_706_ = v_reuseFailAlloc_707_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
return v___x_706_;
}
}
}
else
{
lean_object* v_a_709_; lean_object* v___x_711_; uint8_t v_isShared_712_; uint8_t v_isSharedCheck_716_; 
lean_dec(v_a_697_);
lean_dec(v_a_686_);
v_a_709_ = lean_ctor_get(v___x_698_, 0);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_698_);
if (v_isSharedCheck_716_ == 0)
{
v___x_711_ = v___x_698_;
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
else
{
lean_inc(v_a_709_);
lean_dec(v___x_698_);
v___x_711_ = lean_box(0);
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
v_resetjp_710_:
{
lean_object* v___x_714_; 
if (v_isShared_712_ == 0)
{
v___x_714_ = v___x_711_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_a_709_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
}
else
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_724_; 
lean_dec(v_a_686_);
lean_dec(v_a_663_);
v_a_717_ = lean_ctor_get(v___x_696_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_696_);
if (v_isSharedCheck_724_ == 0)
{
v___x_719_ = v___x_696_;
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_696_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_722_; 
if (v_isShared_720_ == 0)
{
v___x_722_ = v___x_719_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v_a_717_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
}
}
}
}
else
{
lean_object* v_a_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_733_; 
lean_dec(v_a_663_);
lean_dec(v_a_661_);
v_a_726_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_733_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_733_ == 0)
{
v___x_728_ = v___x_685_;
v_isShared_729_ = v_isSharedCheck_733_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_a_726_);
lean_dec(v___x_685_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_733_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v___x_731_; 
if (v_isShared_729_ == 0)
{
v___x_731_ = v___x_728_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_a_726_);
v___x_731_ = v_reuseFailAlloc_732_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
return v___x_731_;
}
}
}
}
}
}
else
{
lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_743_; 
lean_dec(v_a_661_);
lean_dec_ref(v___y_655_);
lean_dec_ref(v_P_654_);
v_a_736_ = lean_ctor_get(v___x_662_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_662_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_662_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_662_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_741_; 
if (v_isShared_739_ == 0)
{
v___x_741_ = v___x_738_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_736_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
else
{
lean_object* v_a_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_751_; 
lean_dec_ref(v___y_655_);
lean_dec_ref(v_P_654_);
lean_dec(v___x_653_);
lean_dec(v___x_651_);
v_a_744_ = lean_ctor_get(v___x_660_, 0);
v_isSharedCheck_751_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_751_ == 0)
{
v___x_746_ = v___x_660_;
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_a_744_);
lean_dec(v___x_660_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_749_; 
if (v_isShared_747_ == 0)
{
v___x_749_ = v___x_746_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_a_744_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed(lean_object* v___x_752_, lean_object* v___x_753_, lean_object* v___x_754_, lean_object* v_P_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_){
_start:
{
uint8_t v___x_10346__boxed_761_; lean_object* v_res_762_; 
v___x_10346__boxed_761_ = lean_unbox(v___x_753_);
v_res_762_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1(v___x_752_, v___x_10346__boxed_761_, v___x_754_, v_P_755_, v___y_756_, v___y_757_, v___y_758_, v___y_759_);
lean_dec(v___y_759_);
lean_dec_ref(v___y_758_);
lean_dec(v___y_757_);
return v_res_762_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0(void){
_start:
{
lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_763_ = lean_box(0);
v___x_764_ = l_Lean_Expr_sort___override(v___x_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(lean_object* v_P_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_){
_start:
{
lean_object* v___x_771_; 
v___x_771_ = l_Lean_Meta_mkFreshLevelMVar(v___y_766_, v___y_767_, v___y_768_, v___y_769_);
if (lean_obj_tag(v___x_771_) == 0)
{
lean_object* v_a_772_; lean_object* v___x_773_; lean_object* v___x_774_; uint8_t v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v_a_772_ = lean_ctor_get(v___x_771_, 0);
lean_inc_n(v_a_772_, 2);
lean_dec_ref_known(v___x_771_, 1);
v___x_773_ = l_Lean_Expr_sort___override(v_a_772_);
v___x_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
v___x_775_ = 0;
v___x_776_ = lean_box(0);
v___x_777_ = l_Lean_Meta_mkFreshExprMVar(v___x_774_, v___x_775_, v___x_776_, v___y_766_, v___y_767_, v___y_768_, v___y_769_);
if (lean_obj_tag(v___x_777_) == 0)
{
lean_object* v_a_778_; lean_object* v___x_779_; uint8_t v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v_a_778_ = lean_ctor_get(v___x_777_, 0);
lean_inc_n(v_a_778_, 2);
lean_dec_ref_known(v___x_777_, 1);
v___x_779_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0);
v___x_780_ = 0;
v___x_781_ = l_Lean_Expr_forallE___override(v___x_776_, v_a_778_, v___x_779_, v___x_780_);
v___x_782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_782_, 0, v___x_781_);
v___x_783_ = l_Lean_Meta_mkFreshExprMVar(v___x_782_, v___x_775_, v___x_776_, v___y_766_, v___y_767_, v___y_768_, v___y_769_);
if (lean_obj_tag(v___x_783_) == 0)
{
lean_object* v_a_784_; lean_object* v_keyedConfig_785_; uint8_t v_trackZetaDelta_786_; lean_object* v_zetaDeltaSet_787_; lean_object* v_lctx_788_; lean_object* v_localInstances_789_; lean_object* v_defEqCtx_x3f_790_; lean_object* v_synthPendingDepth_791_; lean_object* v_customCanUnfoldPredicate_x3f_792_; uint8_t v_univApprox_793_; uint8_t v_inTypeClassResolution_794_; uint8_t v_cacheInferType_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_871_; 
v_a_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_a_784_);
lean_dec_ref_known(v___x_783_, 1);
v_keyedConfig_785_ = lean_ctor_get(v___y_766_, 0);
v_trackZetaDelta_786_ = lean_ctor_get_uint8(v___y_766_, sizeof(void*)*7);
v_zetaDeltaSet_787_ = lean_ctor_get(v___y_766_, 1);
v_lctx_788_ = lean_ctor_get(v___y_766_, 2);
v_localInstances_789_ = lean_ctor_get(v___y_766_, 3);
v_defEqCtx_x3f_790_ = lean_ctor_get(v___y_766_, 4);
v_synthPendingDepth_791_ = lean_ctor_get(v___y_766_, 5);
v_customCanUnfoldPredicate_x3f_792_ = lean_ctor_get(v___y_766_, 6);
v_univApprox_793_ = lean_ctor_get_uint8(v___y_766_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_794_ = lean_ctor_get_uint8(v___y_766_, sizeof(void*)*7 + 2);
v_cacheInferType_795_ = lean_ctor_get_uint8(v___y_766_, sizeof(void*)*7 + 3);
v_isSharedCheck_871_ = !lean_is_exclusive(v___y_766_);
if (v_isSharedCheck_871_ == 0)
{
v___x_797_ = v___y_766_;
v_isShared_798_ = v_isSharedCheck_871_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_792_);
lean_inc(v_synthPendingDepth_791_);
lean_inc(v_defEqCtx_x3f_790_);
lean_inc(v_localInstances_789_);
lean_inc(v_lctx_788_);
lean_inc(v_zetaDeltaSet_787_);
lean_inc(v_keyedConfig_785_);
lean_dec(v___y_766_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_871_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; uint8_t v___x_805_; lean_object* v___x_806_; lean_object* v___x_808_; 
v___x_799_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_800_ = lean_box(0);
lean_inc(v_a_772_);
v___x_801_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_801_, 0, v_a_772_);
lean_ctor_set(v___x_801_, 1, v___x_800_);
v___x_802_ = l_Lean_Expr_const___override(v___x_799_, v___x_801_);
lean_inc(v_a_778_);
v___x_803_ = l_Lean_Expr_app___override(v___x_802_, v_a_778_);
lean_inc(v_a_784_);
v___x_804_ = l_Lean_Expr_app___override(v___x_803_, v_a_784_);
v___x_805_ = 2;
v___x_806_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_805_, v_keyedConfig_785_);
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v___x_806_);
v___x_808_ = v___x_797_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v___x_806_);
lean_ctor_set(v_reuseFailAlloc_870_, 1, v_zetaDeltaSet_787_);
lean_ctor_set(v_reuseFailAlloc_870_, 2, v_lctx_788_);
lean_ctor_set(v_reuseFailAlloc_870_, 3, v_localInstances_789_);
lean_ctor_set(v_reuseFailAlloc_870_, 4, v_defEqCtx_x3f_790_);
lean_ctor_set(v_reuseFailAlloc_870_, 5, v_synthPendingDepth_791_);
lean_ctor_set(v_reuseFailAlloc_870_, 6, v_customCanUnfoldPredicate_x3f_792_);
lean_ctor_set_uint8(v_reuseFailAlloc_870_, sizeof(void*)*7, v_trackZetaDelta_786_);
lean_ctor_set_uint8(v_reuseFailAlloc_870_, sizeof(void*)*7 + 1, v_univApprox_793_);
lean_ctor_set_uint8(v_reuseFailAlloc_870_, sizeof(void*)*7 + 2, v_inTypeClassResolution_794_);
lean_ctor_set_uint8(v_reuseFailAlloc_870_, sizeof(void*)*7 + 3, v_cacheInferType_795_);
v___x_808_ = v_reuseFailAlloc_870_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
lean_object* v___x_809_; 
v___x_809_ = l_Lean_Meta_isExprDefEq(v___x_804_, v_P_765_, v___x_808_, v___y_767_, v___y_768_, v___y_769_);
lean_dec_ref(v___x_808_);
if (lean_obj_tag(v___x_809_) == 0)
{
lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_861_; 
v_a_810_ = lean_ctor_get(v___x_809_, 0);
v_isSharedCheck_861_ = !lean_is_exclusive(v___x_809_);
if (v_isSharedCheck_861_ == 0)
{
v___x_812_ = v___x_809_;
v_isShared_813_ = v_isSharedCheck_861_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_809_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_861_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
uint8_t v___x_814_; 
v___x_814_ = lean_unbox(v_a_810_);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_819_; 
v___x_815_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_815_, 0, v_a_784_);
lean_ctor_set(v___x_815_, 1, v_a_810_);
v___x_816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_816_, 0, v_a_778_);
lean_ctor_set(v___x_816_, 1, v___x_815_);
v___x_817_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_817_, 0, v_a_772_);
lean_ctor_set(v___x_817_, 1, v___x_816_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 0, v___x_817_);
v___x_819_ = v___x_812_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v___x_817_);
v___x_819_ = v_reuseFailAlloc_820_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
return v___x_819_;
}
}
else
{
lean_object* v___x_821_; 
lean_del_object(v___x_812_);
v___x_821_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(v_a_772_, v___y_767_);
if (lean_obj_tag(v___x_821_) == 0)
{
lean_object* v_a_822_; lean_object* v___x_823_; 
v_a_822_ = lean_ctor_get(v___x_821_, 0);
lean_inc(v_a_822_);
lean_dec_ref_known(v___x_821_, 1);
v___x_823_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_778_, v___y_767_);
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v_a_824_; lean_object* v___x_825_; 
v_a_824_ = lean_ctor_get(v___x_823_, 0);
lean_inc(v_a_824_);
lean_dec_ref_known(v___x_823_, 1);
v___x_825_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_784_, v___y_767_);
if (lean_obj_tag(v___x_825_) == 0)
{
lean_object* v_a_826_; lean_object* v___x_828_; uint8_t v_isShared_829_; uint8_t v_isSharedCheck_836_; 
v_a_826_ = lean_ctor_get(v___x_825_, 0);
v_isSharedCheck_836_ = !lean_is_exclusive(v___x_825_);
if (v_isSharedCheck_836_ == 0)
{
v___x_828_ = v___x_825_;
v_isShared_829_ = v_isSharedCheck_836_;
goto v_resetjp_827_;
}
else
{
lean_inc(v_a_826_);
lean_dec(v___x_825_);
v___x_828_ = lean_box(0);
v_isShared_829_ = v_isSharedCheck_836_;
goto v_resetjp_827_;
}
v_resetjp_827_:
{
lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_834_; 
v___x_830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_830_, 0, v_a_826_);
lean_ctor_set(v___x_830_, 1, v_a_810_);
v___x_831_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_831_, 0, v_a_824_);
lean_ctor_set(v___x_831_, 1, v___x_830_);
v___x_832_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_832_, 0, v_a_822_);
lean_ctor_set(v___x_832_, 1, v___x_831_);
if (v_isShared_829_ == 0)
{
lean_ctor_set(v___x_828_, 0, v___x_832_);
v___x_834_ = v___x_828_;
goto v_reusejp_833_;
}
else
{
lean_object* v_reuseFailAlloc_835_; 
v_reuseFailAlloc_835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_835_, 0, v___x_832_);
v___x_834_ = v_reuseFailAlloc_835_;
goto v_reusejp_833_;
}
v_reusejp_833_:
{
return v___x_834_;
}
}
}
else
{
lean_object* v_a_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_844_; 
lean_dec(v_a_824_);
lean_dec(v_a_822_);
lean_dec(v_a_810_);
v_a_837_ = lean_ctor_get(v___x_825_, 0);
v_isSharedCheck_844_ = !lean_is_exclusive(v___x_825_);
if (v_isSharedCheck_844_ == 0)
{
v___x_839_ = v___x_825_;
v_isShared_840_ = v_isSharedCheck_844_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_a_837_);
lean_dec(v___x_825_);
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
else
{
lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_852_; 
lean_dec(v_a_822_);
lean_dec(v_a_810_);
lean_dec(v_a_784_);
v_a_845_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_852_ == 0)
{
v___x_847_ = v___x_823_;
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_823_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
lean_object* v___x_850_; 
if (v_isShared_848_ == 0)
{
v___x_850_ = v___x_847_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_a_845_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
}
else
{
lean_object* v_a_853_; lean_object* v___x_855_; uint8_t v_isShared_856_; uint8_t v_isSharedCheck_860_; 
lean_dec(v_a_810_);
lean_dec(v_a_784_);
lean_dec(v_a_778_);
v_a_853_ = lean_ctor_get(v___x_821_, 0);
v_isSharedCheck_860_ = !lean_is_exclusive(v___x_821_);
if (v_isSharedCheck_860_ == 0)
{
v___x_855_ = v___x_821_;
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
else
{
lean_inc(v_a_853_);
lean_dec(v___x_821_);
v___x_855_ = lean_box(0);
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
v_resetjp_854_:
{
lean_object* v___x_858_; 
if (v_isShared_856_ == 0)
{
v___x_858_ = v___x_855_;
goto v_reusejp_857_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_a_853_);
v___x_858_ = v_reuseFailAlloc_859_;
goto v_reusejp_857_;
}
v_reusejp_857_:
{
return v___x_858_;
}
}
}
}
}
}
else
{
lean_object* v_a_862_; lean_object* v___x_864_; uint8_t v_isShared_865_; uint8_t v_isSharedCheck_869_; 
lean_dec(v_a_784_);
lean_dec(v_a_778_);
lean_dec(v_a_772_);
v_a_862_ = lean_ctor_get(v___x_809_, 0);
v_isSharedCheck_869_ = !lean_is_exclusive(v___x_809_);
if (v_isSharedCheck_869_ == 0)
{
v___x_864_ = v___x_809_;
v_isShared_865_ = v_isSharedCheck_869_;
goto v_resetjp_863_;
}
else
{
lean_inc(v_a_862_);
lean_dec(v___x_809_);
v___x_864_ = lean_box(0);
v_isShared_865_ = v_isSharedCheck_869_;
goto v_resetjp_863_;
}
v_resetjp_863_:
{
lean_object* v___x_867_; 
if (v_isShared_865_ == 0)
{
v___x_867_ = v___x_864_;
goto v_reusejp_866_;
}
else
{
lean_object* v_reuseFailAlloc_868_; 
v_reuseFailAlloc_868_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_868_, 0, v_a_862_);
v___x_867_ = v_reuseFailAlloc_868_;
goto v_reusejp_866_;
}
v_reusejp_866_:
{
return v___x_867_;
}
}
}
}
}
}
else
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_879_; 
lean_dec(v_a_778_);
lean_dec(v_a_772_);
lean_dec_ref(v___y_766_);
lean_dec_ref(v_P_765_);
v_a_872_ = lean_ctor_get(v___x_783_, 0);
v_isSharedCheck_879_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_879_ == 0)
{
v___x_874_ = v___x_783_;
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_783_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v___x_877_; 
if (v_isShared_875_ == 0)
{
v___x_877_ = v___x_874_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v_a_872_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
return v___x_877_;
}
}
}
}
else
{
lean_object* v_a_880_; lean_object* v___x_882_; uint8_t v_isShared_883_; uint8_t v_isSharedCheck_887_; 
lean_dec(v_a_772_);
lean_dec_ref(v___y_766_);
lean_dec_ref(v_P_765_);
v_a_880_ = lean_ctor_get(v___x_777_, 0);
v_isSharedCheck_887_ = !lean_is_exclusive(v___x_777_);
if (v_isSharedCheck_887_ == 0)
{
v___x_882_ = v___x_777_;
v_isShared_883_ = v_isSharedCheck_887_;
goto v_resetjp_881_;
}
else
{
lean_inc(v_a_880_);
lean_dec(v___x_777_);
v___x_882_ = lean_box(0);
v_isShared_883_ = v_isSharedCheck_887_;
goto v_resetjp_881_;
}
v_resetjp_881_:
{
lean_object* v___x_885_; 
if (v_isShared_883_ == 0)
{
v___x_885_ = v___x_882_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v_a_880_);
v___x_885_ = v_reuseFailAlloc_886_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
return v___x_885_;
}
}
}
}
else
{
lean_object* v_a_888_; lean_object* v___x_890_; uint8_t v_isShared_891_; uint8_t v_isSharedCheck_895_; 
lean_dec_ref(v___y_766_);
lean_dec_ref(v_P_765_);
v_a_888_ = lean_ctor_get(v___x_771_, 0);
v_isSharedCheck_895_ = !lean_is_exclusive(v___x_771_);
if (v_isSharedCheck_895_ == 0)
{
v___x_890_ = v___x_771_;
v_isShared_891_ = v_isSharedCheck_895_;
goto v_resetjp_889_;
}
else
{
lean_inc(v_a_888_);
lean_dec(v___x_771_);
v___x_890_ = lean_box(0);
v_isShared_891_ = v_isSharedCheck_895_;
goto v_resetjp_889_;
}
v_resetjp_889_:
{
lean_object* v___x_893_; 
if (v_isShared_891_ == 0)
{
v___x_893_ = v___x_890_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_a_888_);
v___x_893_ = v_reuseFailAlloc_894_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
return v___x_893_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed(lean_object* v_P_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3(v_P_896_, v___y_897_, v___y_898_, v___y_899_, v___y_900_);
lean_dec(v___y_900_);
lean_dec_ref(v___y_899_);
lean_dec(v___y_898_);
return v_res_902_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3(void){
_start:
{
lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; 
v___x_906_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__2));
v___x_907_ = lean_unsigned_to_nat(33u);
v___x_908_ = lean_unsigned_to_nat(125u);
v___x_909_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_910_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_911_ = l_mkPanicMessageWithDecl(v___x_910_, v___x_909_, v___x_908_, v___x_907_, v___x_906_);
return v___x_911_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2(void){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_914_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__1));
v___x_915_ = lean_unsigned_to_nat(4u);
v___x_916_ = lean_unsigned_to_nat(110u);
v___x_917_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_918_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_919_ = l_mkPanicMessageWithDecl(v___x_918_, v___x_917_, v___x_916_, v___x_915_, v___x_914_);
return v___x_919_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0);
v___x_921_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_921_, 0, v___x_920_);
return v___x_921_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5(void){
_start:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; 
v___x_923_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4));
v___x_924_ = lean_unsigned_to_nat(27u);
v___x_925_ = lean_unsigned_to_nat(113u);
v___x_926_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_927_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_928_ = l_mkPanicMessageWithDecl(v___x_927_, v___x_926_, v___x_925_, v___x_924_, v___x_923_);
return v___x_928_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6(void){
_start:
{
lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_929_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__4));
v___x_930_ = lean_unsigned_to_nat(27u);
v___x_931_ = lean_unsigned_to_nat(117u);
v___x_932_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_933_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_934_ = l_mkPanicMessageWithDecl(v___x_933_, v___x_932_, v___x_931_, v___x_930_, v___x_929_);
return v___x_934_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8(void){
_start:
{
lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_936_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7));
v___x_937_ = lean_unsigned_to_nat(4u);
v___x_938_ = lean_unsigned_to_nat(121u);
v___x_939_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_940_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_941_ = l_mkPanicMessageWithDecl(v___x_940_, v___x_939_, v___x_938_, v___x_937_, v___x_936_);
return v___x_941_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10(void){
_start:
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_943_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__9));
v___x_944_ = lean_unsigned_to_nat(34u);
v___x_945_ = lean_unsigned_to_nat(123u);
v___x_946_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_947_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_948_ = l_mkPanicMessageWithDecl(v___x_947_, v___x_946_, v___x_945_, v___x_944_, v___x_943_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed(lean_object* v___x_949_, lean_object* v_u_950_, lean_object* v_a_951_, lean_object* v_tail_952_, lean_object* v_fst_953_, lean_object* v_fst_954_, lean_object* v_bs_955_, lean_object* v_body_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(v___x_949_, v_u_950_, v_a_951_, v_tail_952_, v_fst_953_, v_fst_954_, v_bs_955_, v_body_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec_ref(v_bs_955_);
lean_dec(v___x_949_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(lean_object* v_u_963_, lean_object* v_a_964_, lean_object* v_P_965_, lean_object* v_path_966_, lean_object* v_a_967_, lean_object* v_a_968_, lean_object* v_a_969_, lean_object* v_a_970_){
_start:
{
if (lean_obj_tag(v_path_966_) == 0)
{
lean_object* v___x_972_; lean_object* v___x_973_; uint8_t v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___f_977_; uint8_t v___x_978_; lean_object* v___x_979_; 
lean_inc(v_u_963_);
v___x_972_ = l_Lean_Expr_sort___override(v_u_963_);
v___x_973_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_973_, 0, v___x_972_);
v___x_974_ = 0;
v___x_975_ = lean_box(0);
v___x_976_ = lean_box(v___x_974_);
lean_inc_ref(v_P_965_);
v___f_977_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_977_, 0, v___x_973_);
lean_closure_set(v___f_977_, 1, v___x_976_);
lean_closure_set(v___f_977_, 2, v___x_975_);
lean_closure_set(v___f_977_, 3, v_u_963_);
lean_closure_set(v___f_977_, 4, v_P_965_);
v___x_978_ = 0;
v___x_979_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_977_, v___x_978_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_979_) == 0)
{
lean_object* v_a_980_; lean_object* v___x_982_; uint8_t v_isShared_983_; uint8_t v_isSharedCheck_1058_; 
v_a_980_ = lean_ctor_get(v___x_979_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v___x_979_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_982_ = v___x_979_;
v_isShared_983_ = v_isSharedCheck_1058_;
goto v_resetjp_981_;
}
else
{
lean_inc(v_a_980_);
lean_dec(v___x_979_);
v___x_982_ = lean_box(0);
v_isShared_983_ = v_isSharedCheck_1058_;
goto v_resetjp_981_;
}
v_resetjp_981_:
{
lean_object* v_snd_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_1056_; 
v_snd_984_ = lean_ctor_get(v_a_980_, 1);
v_isSharedCheck_1056_ = !lean_is_exclusive(v_a_980_);
if (v_isSharedCheck_1056_ == 0)
{
lean_object* v_unused_1057_; 
v_unused_1057_ = lean_ctor_get(v_a_980_, 0);
lean_dec(v_unused_1057_);
v___x_986_ = v_a_980_;
v_isShared_987_ = v_isSharedCheck_1056_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_snd_984_);
lean_dec(v_a_980_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_1056_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v_snd_988_; lean_object* v_snd_989_; uint8_t v___x_990_; 
v_snd_988_ = lean_ctor_get(v_snd_984_, 1);
lean_inc(v_snd_988_);
v_snd_989_ = lean_ctor_get(v_snd_988_, 1);
v___x_990_ = lean_unbox(v_snd_989_);
if (v___x_990_ == 0)
{
lean_object* v___x_991_; 
lean_dec(v_snd_988_);
lean_del_object(v___x_986_);
lean_dec(v_snd_984_);
lean_del_object(v___x_982_);
lean_dec_ref(v_a_964_);
v___x_991_ = l_Lean_Meta_ppExpr(v_P_965_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_991_) == 0)
{
lean_object* v_a_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref_known(v___x_991_, 1);
v___x_993_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_994_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__1));
v___x_995_ = lean_unsigned_to_nat(105u);
v___x_996_ = lean_unsigned_to_nat(8u);
v___x_997_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__0));
v___x_998_ = l_Std_Format_defWidth;
v___x_999_ = lean_unsigned_to_nat(0u);
v___x_1000_ = l_Std_Format_pretty(v_a_992_, v___x_998_, v___x_999_, v___x_999_);
v___x_1001_ = lean_string_append(v___x_997_, v___x_1000_);
lean_dec_ref(v___x_1000_);
v___x_1002_ = l_mkPanicMessageWithDecl(v___x_993_, v___x_994_, v___x_995_, v___x_996_, v___x_1001_);
lean_dec_ref(v___x_1001_);
v___x_1003_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1002_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1003_;
}
else
{
lean_object* v_a_1004_; lean_object* v___x_1006_; uint8_t v_isShared_1007_; uint8_t v_isSharedCheck_1011_; 
v_a_1004_ = lean_ctor_get(v___x_991_, 0);
v_isSharedCheck_1011_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1011_ == 0)
{
v___x_1006_ = v___x_991_;
v_isShared_1007_ = v_isSharedCheck_1011_;
goto v_resetjp_1005_;
}
else
{
lean_inc(v_a_1004_);
lean_dec(v___x_991_);
v___x_1006_ = lean_box(0);
v_isShared_1007_ = v_isSharedCheck_1011_;
goto v_resetjp_1005_;
}
v_resetjp_1005_:
{
lean_object* v___x_1009_; 
if (v_isShared_1007_ == 0)
{
v___x_1009_ = v___x_1006_;
goto v_reusejp_1008_;
}
else
{
lean_object* v_reuseFailAlloc_1010_; 
v_reuseFailAlloc_1010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1010_, 0, v_a_1004_);
v___x_1009_ = v_reuseFailAlloc_1010_;
goto v_reusejp_1008_;
}
v_reusejp_1008_:
{
return v___x_1009_;
}
}
}
}
else
{
lean_object* v_fst_1012_; lean_object* v___x_1014_; uint8_t v_isShared_1015_; uint8_t v_isSharedCheck_1054_; 
v_fst_1012_ = lean_ctor_get(v_snd_984_, 0);
v_isSharedCheck_1054_ = !lean_is_exclusive(v_snd_984_);
if (v_isSharedCheck_1054_ == 0)
{
lean_object* v_unused_1055_; 
v_unused_1055_ = lean_ctor_get(v_snd_984_, 1);
lean_dec(v_unused_1055_);
v___x_1014_ = v_snd_984_;
v_isShared_1015_ = v_isSharedCheck_1054_;
goto v_resetjp_1013_;
}
else
{
lean_inc(v_fst_1012_);
lean_dec(v_snd_984_);
v___x_1014_ = lean_box(0);
v_isShared_1015_ = v_isSharedCheck_1054_;
goto v_resetjp_1013_;
}
v_resetjp_1013_:
{
lean_object* v_fst_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1052_; 
v_fst_1016_ = lean_ctor_get(v_snd_988_, 0);
v_isSharedCheck_1052_ = !lean_is_exclusive(v_snd_988_);
if (v_isSharedCheck_1052_ == 0)
{
lean_object* v_unused_1053_; 
v_unused_1053_ = lean_ctor_get(v_snd_988_, 1);
lean_dec(v_unused_1053_);
v___x_1018_ = v_snd_988_;
v_isShared_1019_ = v_isSharedCheck_1052_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_fst_1016_);
lean_dec(v_snd_988_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1052_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
uint8_t v___x_1020_; 
v___x_1020_ = lp_mathlib_ExistsAndEq_eqDetermines(v_a_964_, v_fst_1012_, v_fst_1016_);
if (v___x_1020_ == 0)
{
uint8_t v___x_1021_; 
v___x_1021_ = lp_mathlib_ExistsAndEq_eqDetermines(v_a_964_, v_fst_1016_, v_fst_1012_);
lean_dec(v_fst_1016_);
lean_dec_ref(v_a_964_);
if (v___x_1021_ == 0)
{
lean_object* v___x_1022_; lean_object* v___x_1023_; 
lean_del_object(v___x_1018_);
lean_del_object(v___x_1014_);
lean_dec(v_fst_1012_);
lean_del_object(v___x_986_);
lean_del_object(v___x_982_);
lean_dec_ref(v_P_965_);
v___x_1022_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__2);
v___x_1023_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1022_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1023_;
}
else
{
lean_object* v_lctx_1024_; lean_object* v___x_1025_; lean_object* v___x_1027_; 
v_lctx_1024_ = lean_ctor_get(v_a_967_, 2);
v___x_1025_ = lean_box(0);
if (v_isShared_1019_ == 0)
{
lean_ctor_set(v___x_1018_, 1, v_fst_1012_);
lean_ctor_set(v___x_1018_, 0, v_P_965_);
v___x_1027_ = v___x_1018_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1037_; 
v_reuseFailAlloc_1037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1037_, 0, v_P_965_);
lean_ctor_set(v_reuseFailAlloc_1037_, 1, v_fst_1012_);
v___x_1027_ = v_reuseFailAlloc_1037_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
lean_object* v___x_1029_; 
lean_inc_ref(v_lctx_1024_);
if (v_isShared_1015_ == 0)
{
lean_ctor_set(v___x_1014_, 1, v___x_1027_);
lean_ctor_set(v___x_1014_, 0, v_lctx_1024_);
v___x_1029_ = v___x_1014_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_lctx_1024_);
lean_ctor_set(v_reuseFailAlloc_1036_, 1, v___x_1027_);
v___x_1029_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
lean_object* v___x_1031_; 
if (v_isShared_987_ == 0)
{
lean_ctor_set(v___x_986_, 1, v___x_1029_);
lean_ctor_set(v___x_986_, 0, v___x_1025_);
v___x_1031_ = v___x_986_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1035_; 
v_reuseFailAlloc_1035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1035_, 0, v___x_1025_);
lean_ctor_set(v_reuseFailAlloc_1035_, 1, v___x_1029_);
v___x_1031_ = v_reuseFailAlloc_1035_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
lean_object* v___x_1033_; 
if (v_isShared_983_ == 0)
{
lean_ctor_set(v___x_982_, 0, v___x_1031_);
v___x_1033_ = v___x_982_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v___x_1031_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
return v___x_1033_;
}
}
}
}
}
}
else
{
lean_object* v_lctx_1038_; lean_object* v___x_1039_; lean_object* v___x_1041_; 
lean_dec(v_fst_1012_);
lean_dec_ref(v_a_964_);
v_lctx_1038_ = lean_ctor_get(v_a_967_, 2);
v___x_1039_ = lean_box(0);
if (v_isShared_1019_ == 0)
{
lean_ctor_set(v___x_1018_, 1, v_fst_1016_);
lean_ctor_set(v___x_1018_, 0, v_P_965_);
v___x_1041_ = v___x_1018_;
goto v_reusejp_1040_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v_P_965_);
lean_ctor_set(v_reuseFailAlloc_1051_, 1, v_fst_1016_);
v___x_1041_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1040_;
}
v_reusejp_1040_:
{
lean_object* v___x_1043_; 
lean_inc_ref(v_lctx_1038_);
if (v_isShared_1015_ == 0)
{
lean_ctor_set(v___x_1014_, 1, v___x_1041_);
lean_ctor_set(v___x_1014_, 0, v_lctx_1038_);
v___x_1043_ = v___x_1014_;
goto v_reusejp_1042_;
}
else
{
lean_object* v_reuseFailAlloc_1050_; 
v_reuseFailAlloc_1050_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1050_, 0, v_lctx_1038_);
lean_ctor_set(v_reuseFailAlloc_1050_, 1, v___x_1041_);
v___x_1043_ = v_reuseFailAlloc_1050_;
goto v_reusejp_1042_;
}
v_reusejp_1042_:
{
lean_object* v___x_1045_; 
if (v_isShared_987_ == 0)
{
lean_ctor_set(v___x_986_, 1, v___x_1043_);
lean_ctor_set(v___x_986_, 0, v___x_1039_);
v___x_1045_ = v___x_986_;
goto v_reusejp_1044_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v___x_1039_);
lean_ctor_set(v_reuseFailAlloc_1049_, 1, v___x_1043_);
v___x_1045_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1044_;
}
v_reusejp_1044_:
{
lean_object* v___x_1047_; 
if (v_isShared_983_ == 0)
{
lean_ctor_set(v___x_982_, 0, v___x_1045_);
v___x_1047_ = v___x_982_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v___x_1045_);
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
}
}
}
}
}
}
else
{
lean_object* v_a_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1066_; 
lean_dec_ref(v_P_965_);
lean_dec_ref(v_a_964_);
v_a_1059_ = lean_ctor_get(v___x_979_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_979_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1061_ = v___x_979_;
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_a_1059_);
lean_dec(v___x_979_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_a_1059_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
}
else
{
lean_object* v_head_1067_; uint8_t v___x_1068_; 
v_head_1067_ = lean_ctor_get(v_path_966_, 0);
v___x_1068_ = lean_unbox(v_head_1067_);
switch(v___x_1068_)
{
case 0:
{
lean_object* v_tail_1069_; lean_object* v___x_1070_; uint8_t v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___f_1074_; uint8_t v___x_1075_; lean_object* v___x_1076_; 
v_tail_1069_ = lean_ctor_get(v_path_966_, 1);
lean_inc(v_tail_1069_);
lean_dec_ref_known(v_path_966_, 2);
v___x_1070_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_1071_ = 0;
v___x_1072_ = lean_box(0);
v___x_1073_ = lean_box(v___x_1071_);
v___f_1074_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1074_, 0, v___x_1070_);
lean_closure_set(v___f_1074_, 1, v___x_1073_);
lean_closure_set(v___f_1074_, 2, v___x_1072_);
lean_closure_set(v___f_1074_, 3, v_P_965_);
v___x_1075_ = 0;
v___x_1076_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1074_, v___x_1075_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_1076_) == 0)
{
lean_object* v_a_1077_; lean_object* v_snd_1078_; lean_object* v_snd_1079_; uint8_t v___x_1080_; 
v_a_1077_ = lean_ctor_get(v___x_1076_, 0);
lean_inc(v_a_1077_);
lean_dec_ref_known(v___x_1076_, 1);
v_snd_1078_ = lean_ctor_get(v_a_1077_, 1);
lean_inc(v_snd_1078_);
v_snd_1079_ = lean_ctor_get(v_snd_1078_, 1);
v___x_1080_ = lean_unbox(v_snd_1079_);
if (v___x_1080_ == 0)
{
lean_object* v___x_1081_; lean_object* v___x_1082_; 
lean_dec(v_snd_1078_);
lean_dec(v_a_1077_);
lean_dec(v_tail_1069_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v___x_1081_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__5);
v___x_1082_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1081_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1082_;
}
else
{
lean_object* v_fst_1083_; lean_object* v_fst_1084_; lean_object* v___x_1085_; 
v_fst_1083_ = lean_ctor_get(v_a_1077_, 0);
lean_inc(v_fst_1083_);
lean_dec(v_a_1077_);
v_fst_1084_ = lean_ctor_get(v_snd_1078_, 0);
lean_inc(v_fst_1084_);
lean_dec(v_snd_1078_);
v___x_1085_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_963_, v_a_964_, v_fst_1083_, v_tail_1069_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_1085_) == 0)
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1125_; 
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
v_isSharedCheck_1125_ = !lean_is_exclusive(v___x_1085_);
if (v_isSharedCheck_1125_ == 0)
{
v___x_1088_ = v___x_1085_;
v_isShared_1089_ = v_isSharedCheck_1125_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_1085_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1125_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v_snd_1090_; lean_object* v_snd_1091_; lean_object* v_fst_1092_; lean_object* v___x_1094_; uint8_t v_isShared_1095_; uint8_t v_isSharedCheck_1123_; 
v_snd_1090_ = lean_ctor_get(v_a_1086_, 1);
lean_inc(v_snd_1090_);
v_snd_1091_ = lean_ctor_get(v_snd_1090_, 1);
lean_inc(v_snd_1091_);
v_fst_1092_ = lean_ctor_get(v_a_1086_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v_a_1086_);
if (v_isSharedCheck_1123_ == 0)
{
lean_object* v_unused_1124_; 
v_unused_1124_ = lean_ctor_get(v_a_1086_, 1);
lean_dec(v_unused_1124_);
v___x_1094_ = v_a_1086_;
v_isShared_1095_ = v_isSharedCheck_1123_;
goto v_resetjp_1093_;
}
else
{
lean_inc(v_fst_1092_);
lean_dec(v_a_1086_);
v___x_1094_ = lean_box(0);
v_isShared_1095_ = v_isSharedCheck_1123_;
goto v_resetjp_1093_;
}
v_resetjp_1093_:
{
lean_object* v_fst_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1121_; 
v_fst_1096_ = lean_ctor_get(v_snd_1090_, 0);
v_isSharedCheck_1121_ = !lean_is_exclusive(v_snd_1090_);
if (v_isSharedCheck_1121_ == 0)
{
lean_object* v_unused_1122_; 
v_unused_1122_ = lean_ctor_get(v_snd_1090_, 1);
lean_dec(v_unused_1122_);
v___x_1098_ = v_snd_1090_;
v_isShared_1099_ = v_isSharedCheck_1121_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_fst_1096_);
lean_dec(v_snd_1090_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1121_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v_fst_1100_; lean_object* v_snd_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1120_; 
v_fst_1100_ = lean_ctor_get(v_snd_1091_, 0);
v_snd_1101_ = lean_ctor_get(v_snd_1091_, 1);
v_isSharedCheck_1120_ = !lean_is_exclusive(v_snd_1091_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1103_ = v_snd_1091_;
v_isShared_1104_ = v_isSharedCheck_1120_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_snd_1101_);
lean_inc(v_fst_1100_);
lean_dec(v_snd_1091_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1120_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1109_; 
v___x_1105_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0);
v___x_1106_ = l_Lean_Expr_app___override(v___x_1105_, v_fst_1100_);
v___x_1107_ = l_Lean_Expr_app___override(v___x_1106_, v_fst_1084_);
if (v_isShared_1104_ == 0)
{
lean_ctor_set(v___x_1103_, 0, v___x_1107_);
v___x_1109_ = v___x_1103_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v___x_1107_);
lean_ctor_set(v_reuseFailAlloc_1119_, 1, v_snd_1101_);
v___x_1109_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
lean_object* v___x_1111_; 
if (v_isShared_1099_ == 0)
{
lean_ctor_set(v___x_1098_, 1, v___x_1109_);
v___x_1111_ = v___x_1098_;
goto v_reusejp_1110_;
}
else
{
lean_object* v_reuseFailAlloc_1118_; 
v_reuseFailAlloc_1118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1118_, 0, v_fst_1096_);
lean_ctor_set(v_reuseFailAlloc_1118_, 1, v___x_1109_);
v___x_1111_ = v_reuseFailAlloc_1118_;
goto v_reusejp_1110_;
}
v_reusejp_1110_:
{
lean_object* v___x_1113_; 
if (v_isShared_1095_ == 0)
{
lean_ctor_set(v___x_1094_, 1, v___x_1111_);
v___x_1113_ = v___x_1094_;
goto v_reusejp_1112_;
}
else
{
lean_object* v_reuseFailAlloc_1117_; 
v_reuseFailAlloc_1117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1117_, 0, v_fst_1092_);
lean_ctor_set(v_reuseFailAlloc_1117_, 1, v___x_1111_);
v___x_1113_ = v_reuseFailAlloc_1117_;
goto v_reusejp_1112_;
}
v_reusejp_1112_:
{
lean_object* v___x_1115_; 
if (v_isShared_1089_ == 0)
{
lean_ctor_set(v___x_1088_, 0, v___x_1113_);
v___x_1115_ = v___x_1088_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v___x_1113_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
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
lean_dec(v_fst_1084_);
return v___x_1085_;
}
}
}
else
{
lean_object* v_a_1126_; lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1133_; 
lean_dec(v_tail_1069_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v_a_1126_ = lean_ctor_get(v___x_1076_, 0);
v_isSharedCheck_1133_ = !lean_is_exclusive(v___x_1076_);
if (v_isSharedCheck_1133_ == 0)
{
v___x_1128_ = v___x_1076_;
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
else
{
lean_inc(v_a_1126_);
lean_dec(v___x_1076_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
lean_object* v___x_1131_; 
if (v_isShared_1129_ == 0)
{
v___x_1131_ = v___x_1128_;
goto v_reusejp_1130_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v_a_1126_);
v___x_1131_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1130_;
}
v_reusejp_1130_:
{
return v___x_1131_;
}
}
}
}
case 1:
{
lean_object* v_tail_1134_; lean_object* v___x_1135_; uint8_t v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___f_1139_; uint8_t v___x_1140_; lean_object* v___x_1141_; 
v_tail_1134_ = lean_ctor_get(v_path_966_, 1);
lean_inc(v_tail_1134_);
lean_dec_ref_known(v_path_966_, 2);
v___x_1135_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_1136_ = 0;
v___x_1137_ = lean_box(0);
v___x_1138_ = lean_box(v___x_1136_);
v___f_1139_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1139_, 0, v___x_1135_);
lean_closure_set(v___f_1139_, 1, v___x_1138_);
lean_closure_set(v___f_1139_, 2, v___x_1137_);
lean_closure_set(v___f_1139_, 3, v_P_965_);
v___x_1140_ = 0;
v___x_1141_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1139_, v___x_1140_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_1141_) == 0)
{
lean_object* v_a_1142_; lean_object* v_snd_1143_; lean_object* v_snd_1144_; uint8_t v___x_1145_; 
v_a_1142_ = lean_ctor_get(v___x_1141_, 0);
lean_inc(v_a_1142_);
lean_dec_ref_known(v___x_1141_, 1);
v_snd_1143_ = lean_ctor_get(v_a_1142_, 1);
lean_inc(v_snd_1143_);
v_snd_1144_ = lean_ctor_get(v_snd_1143_, 1);
v___x_1145_ = lean_unbox(v_snd_1144_);
if (v___x_1145_ == 0)
{
lean_object* v___x_1146_; lean_object* v___x_1147_; 
lean_dec(v_snd_1143_);
lean_dec(v_a_1142_);
lean_dec(v_tail_1134_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v___x_1146_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__6);
v___x_1147_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1146_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1147_;
}
else
{
lean_object* v_fst_1148_; lean_object* v_fst_1149_; lean_object* v___x_1150_; 
v_fst_1148_ = lean_ctor_get(v_a_1142_, 0);
lean_inc(v_fst_1148_);
lean_dec(v_a_1142_);
v_fst_1149_ = lean_ctor_get(v_snd_1143_, 0);
lean_inc(v_fst_1149_);
lean_dec(v_snd_1143_);
v___x_1150_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_963_, v_a_964_, v_fst_1149_, v_tail_1134_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_1150_) == 0)
{
lean_object* v_a_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1190_; 
v_a_1151_ = lean_ctor_get(v___x_1150_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1150_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1153_ = v___x_1150_;
v_isShared_1154_ = v_isSharedCheck_1190_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_a_1151_);
lean_dec(v___x_1150_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1190_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v_snd_1155_; lean_object* v_snd_1156_; lean_object* v_fst_1157_; lean_object* v___x_1159_; uint8_t v_isShared_1160_; uint8_t v_isSharedCheck_1188_; 
v_snd_1155_ = lean_ctor_get(v_a_1151_, 1);
lean_inc(v_snd_1155_);
v_snd_1156_ = lean_ctor_get(v_snd_1155_, 1);
lean_inc(v_snd_1156_);
v_fst_1157_ = lean_ctor_get(v_a_1151_, 0);
v_isSharedCheck_1188_ = !lean_is_exclusive(v_a_1151_);
if (v_isSharedCheck_1188_ == 0)
{
lean_object* v_unused_1189_; 
v_unused_1189_ = lean_ctor_get(v_a_1151_, 1);
lean_dec(v_unused_1189_);
v___x_1159_ = v_a_1151_;
v_isShared_1160_ = v_isSharedCheck_1188_;
goto v_resetjp_1158_;
}
else
{
lean_inc(v_fst_1157_);
lean_dec(v_a_1151_);
v___x_1159_ = lean_box(0);
v_isShared_1160_ = v_isSharedCheck_1188_;
goto v_resetjp_1158_;
}
v_resetjp_1158_:
{
lean_object* v_fst_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1186_; 
v_fst_1161_ = lean_ctor_get(v_snd_1155_, 0);
v_isSharedCheck_1186_ = !lean_is_exclusive(v_snd_1155_);
if (v_isSharedCheck_1186_ == 0)
{
lean_object* v_unused_1187_; 
v_unused_1187_ = lean_ctor_get(v_snd_1155_, 1);
lean_dec(v_unused_1187_);
v___x_1163_ = v_snd_1155_;
v_isShared_1164_ = v_isSharedCheck_1186_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_fst_1161_);
lean_dec(v_snd_1155_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1186_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
lean_object* v_fst_1165_; lean_object* v_snd_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1185_; 
v_fst_1165_ = lean_ctor_get(v_snd_1156_, 0);
v_snd_1166_ = lean_ctor_get(v_snd_1156_, 1);
v_isSharedCheck_1185_ = !lean_is_exclusive(v_snd_1156_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1168_ = v_snd_1156_;
v_isShared_1169_ = v_isSharedCheck_1185_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_snd_1166_);
lean_inc(v_fst_1165_);
lean_dec(v_snd_1156_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1185_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1174_; 
v___x_1170_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0);
v___x_1171_ = l_Lean_Expr_app___override(v___x_1170_, v_fst_1148_);
v___x_1172_ = l_Lean_Expr_app___override(v___x_1171_, v_fst_1165_);
if (v_isShared_1169_ == 0)
{
lean_ctor_set(v___x_1168_, 0, v___x_1172_);
v___x_1174_ = v___x_1168_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1184_; 
v_reuseFailAlloc_1184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1184_, 0, v___x_1172_);
lean_ctor_set(v_reuseFailAlloc_1184_, 1, v_snd_1166_);
v___x_1174_ = v_reuseFailAlloc_1184_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
lean_object* v___x_1176_; 
if (v_isShared_1164_ == 0)
{
lean_ctor_set(v___x_1163_, 1, v___x_1174_);
v___x_1176_ = v___x_1163_;
goto v_reusejp_1175_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_fst_1161_);
lean_ctor_set(v_reuseFailAlloc_1183_, 1, v___x_1174_);
v___x_1176_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1175_;
}
v_reusejp_1175_:
{
lean_object* v___x_1178_; 
if (v_isShared_1160_ == 0)
{
lean_ctor_set(v___x_1159_, 1, v___x_1176_);
v___x_1178_ = v___x_1159_;
goto v_reusejp_1177_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_fst_1157_);
lean_ctor_set(v_reuseFailAlloc_1182_, 1, v___x_1176_);
v___x_1178_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1177_;
}
v_reusejp_1177_:
{
lean_object* v___x_1180_; 
if (v_isShared_1154_ == 0)
{
lean_ctor_set(v___x_1153_, 0, v___x_1178_);
v___x_1180_ = v___x_1153_;
goto v_reusejp_1179_;
}
else
{
lean_object* v_reuseFailAlloc_1181_; 
v_reuseFailAlloc_1181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1181_, 0, v___x_1178_);
v___x_1180_ = v_reuseFailAlloc_1181_;
goto v_reusejp_1179_;
}
v_reusejp_1179_:
{
return v___x_1180_;
}
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
lean_dec(v_fst_1148_);
return v___x_1150_;
}
}
}
else
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
lean_dec(v_tail_1134_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v_a_1191_ = lean_ctor_get(v___x_1141_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1141_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1141_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1141_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
}
case 2:
{
lean_object* v___x_1199_; lean_object* v___x_1200_; 
lean_dec_ref_known(v_path_966_, 2);
lean_dec_ref(v_P_965_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v___x_1199_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__8);
v___x_1200_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1199_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1200_;
}
default: 
{
lean_object* v_tail_1201_; lean_object* v___f_1202_; uint8_t v___x_1203_; lean_object* v___x_1204_; 
v_tail_1201_ = lean_ctor_get(v_path_966_, 1);
lean_inc(v_tail_1201_);
lean_dec_ref_known(v_path_966_, 2);
v___f_1202_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___boxed), 6, 1);
lean_closure_set(v___f_1202_, 0, v_P_965_);
v___x_1203_ = 0;
v___x_1204_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1202_, v___x_1203_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
if (lean_obj_tag(v___x_1204_) == 0)
{
lean_object* v_a_1205_; lean_object* v_snd_1206_; lean_object* v_snd_1207_; lean_object* v_snd_1208_; uint8_t v___x_1209_; 
v_a_1205_ = lean_ctor_get(v___x_1204_, 0);
lean_inc(v_a_1205_);
lean_dec_ref_known(v___x_1204_, 1);
v_snd_1206_ = lean_ctor_get(v_a_1205_, 1);
lean_inc(v_snd_1206_);
v_snd_1207_ = lean_ctor_get(v_snd_1206_, 1);
lean_inc(v_snd_1207_);
v_snd_1208_ = lean_ctor_get(v_snd_1207_, 1);
v___x_1209_ = lean_unbox(v_snd_1208_);
if (v___x_1209_ == 0)
{
lean_object* v___x_1210_; lean_object* v___x_1211_; 
lean_dec(v_snd_1207_);
lean_dec(v_snd_1206_);
lean_dec(v_a_1205_);
lean_dec(v_tail_1201_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v___x_1210_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__10);
v___x_1211_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1210_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1211_;
}
else
{
lean_object* v_fst_1212_; lean_object* v_fst_1213_; lean_object* v_fst_1214_; lean_object* v___x_1215_; lean_object* v___f_1216_; lean_object* v___x_1217_; 
v_fst_1212_ = lean_ctor_get(v_a_1205_, 0);
lean_inc(v_fst_1212_);
lean_dec(v_a_1205_);
v_fst_1213_ = lean_ctor_get(v_snd_1206_, 0);
lean_inc(v_fst_1213_);
lean_dec(v_snd_1206_);
v_fst_1214_ = lean_ctor_get(v_snd_1207_, 0);
lean_inc(v_fst_1214_);
lean_dec(v_snd_1207_);
v___x_1215_ = lean_unsigned_to_nat(1u);
v___f_1216_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___boxed), 13, 6);
lean_closure_set(v___f_1216_, 0, v___x_1215_);
lean_closure_set(v___f_1216_, 1, v_u_963_);
lean_closure_set(v___f_1216_, 2, v_a_964_);
lean_closure_set(v___f_1216_, 3, v_tail_1201_);
lean_closure_set(v___f_1216_, 4, v_fst_1213_);
lean_closure_set(v___f_1216_, 5, v_fst_1212_);
v___x_1217_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(v_fst_1214_, v___x_1215_, v___f_1216_, v___x_1203_, v_a_967_, v_a_968_, v_a_969_, v_a_970_);
return v___x_1217_;
}
}
else
{
lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1225_; 
lean_dec(v_tail_1201_);
lean_dec_ref(v_a_964_);
lean_dec(v_u_963_);
v_a_1218_ = lean_ctor_get(v___x_1204_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1204_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1220_ = v___x_1204_;
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1204_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1225_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v___x_1223_; 
if (v_isShared_1221_ == 0)
{
v___x_1223_ = v___x_1220_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v_a_1218_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2(lean_object* v___x_1226_, lean_object* v_u_1227_, lean_object* v_a_1228_, lean_object* v_tail_1229_, lean_object* v_fst_1230_, lean_object* v_fst_1231_, lean_object* v_bs_1232_, lean_object* v_body_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_){
_start:
{
lean_object* v___x_1239_; uint8_t v___x_1240_; 
v___x_1239_ = lean_array_get_size(v_bs_1232_);
v___x_1240_ = lean_nat_dec_eq(v___x_1239_, v___x_1226_);
if (v___x_1240_ == 0)
{
lean_object* v___x_1241_; lean_object* v___x_1242_; 
lean_dec_ref(v_body_1233_);
lean_dec(v_fst_1231_);
lean_dec(v_fst_1230_);
lean_dec(v_tail_1229_);
lean_dec_ref(v_a_1228_);
lean_dec(v_u_1227_);
v___x_1241_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__3);
v___x_1242_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3(v___x_1241_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_);
return v___x_1242_;
}
else
{
lean_object* v___x_1243_; 
v___x_1243_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_1227_, v_a_1228_, v_body_1233_, v_tail_1229_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_);
if (lean_obj_tag(v___x_1243_) == 0)
{
lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1265_; 
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1243_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1246_ = v___x_1243_;
v_isShared_1247_ = v_isSharedCheck_1265_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1243_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1265_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v_fst_1248_; lean_object* v_snd_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1264_; 
v_fst_1248_ = lean_ctor_get(v_a_1244_, 0);
v_snd_1249_ = lean_ctor_get(v_a_1244_, 1);
v_isSharedCheck_1264_ = !lean_is_exclusive(v_a_1244_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1251_ = v_a_1244_;
v_isShared_1252_ = v_isSharedCheck_1264_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_snd_1249_);
lean_inc(v_fst_1248_);
lean_dec(v_a_1244_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1264_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1259_; 
v___x_1253_ = lean_unsigned_to_nat(0u);
v___x_1254_ = lean_array_fget_borrowed(v_bs_1232_, v___x_1253_);
lean_inc(v___x_1254_);
v___x_1255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1255_, 0, v_fst_1230_);
lean_ctor_set(v___x_1255_, 1, v___x_1254_);
v___x_1256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1256_, 0, v_fst_1231_);
lean_ctor_set(v___x_1256_, 1, v___x_1255_);
v___x_1257_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1257_, 0, v___x_1256_);
lean_ctor_set(v___x_1257_, 1, v_fst_1248_);
if (v_isShared_1252_ == 0)
{
lean_ctor_set(v___x_1251_, 0, v___x_1257_);
v___x_1259_ = v___x_1251_;
goto v_reusejp_1258_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v___x_1257_);
lean_ctor_set(v_reuseFailAlloc_1263_, 1, v_snd_1249_);
v___x_1259_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1258_;
}
v_reusejp_1258_:
{
lean_object* v___x_1261_; 
if (v_isShared_1247_ == 0)
{
lean_ctor_set(v___x_1246_, 0, v___x_1259_);
v___x_1261_ = v___x_1246_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v___x_1259_);
v___x_1261_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1260_;
}
v_reusejp_1260_:
{
return v___x_1261_;
}
}
}
}
}
else
{
lean_dec(v_fst_1231_);
lean_dec(v_fst_1230_);
return v___x_1243_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___boxed(lean_object* v_u_1266_, lean_object* v_a_1267_, lean_object* v_P_1268_, lean_object* v_path_1269_, lean_object* v_a_1270_, lean_object* v_a_1271_, lean_object* v_a_1272_, lean_object* v_a_1273_, lean_object* v_a_1274_){
_start:
{
lean_object* v_res_1275_; 
v_res_1275_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_1266_, v_a_1267_, v_P_1268_, v_path_1269_, v_a_1270_, v_a_1271_, v_a_1272_, v_a_1273_);
lean_dec(v_a_1273_);
lean_dec_ref(v_a_1272_);
lean_dec(v_a_1271_);
lean_dec_ref(v_a_1270_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(lean_object* v_u_1276_, lean_object* v_00_u03b1_1277_, lean_object* v_a_1278_, lean_object* v_P_1279_, lean_object* v_path_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_){
_start:
{
lean_object* v___x_1286_; 
v___x_1286_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_1276_, v_a_1278_, v_P_1279_, v_path_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___boxed(lean_object* v_u_1287_, lean_object* v_00_u03b1_1288_, lean_object* v_a_1289_, lean_object* v_P_1290_, lean_object* v_path_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_){
_start:
{
lean_object* v_res_1297_; 
v_res_1297_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go(v_u_1287_, v_00_u03b1_1288_, v_a_1289_, v_P_1290_, v_path_1291_, v_a_1292_, v_a_1293_, v_a_1294_, v_a_1295_);
lean_dec(v_a_1295_);
lean_dec_ref(v_a_1294_);
lean_dec(v_a_1293_);
lean_dec_ref(v_a_1292_);
lean_dec_ref(v_00_u03b1_1288_);
return v_res_1297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg(lean_object* v_u_1298_, lean_object* v_a_1299_, lean_object* v_P_1300_, lean_object* v_path_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_, lean_object* v_a_1305_){
_start:
{
lean_object* v___x_1307_; 
v___x_1307_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_1298_, v_a_1299_, v_P_1300_, v_path_1301_, v_a_1302_, v_a_1303_, v_a_1304_, v_a_1305_);
return v___x_1307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___redArg___boxed(lean_object* v_u_1308_, lean_object* v_a_1309_, lean_object* v_P_1310_, lean_object* v_path_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v_a_1314_, lean_object* v_a_1315_, lean_object* v_a_1316_){
_start:
{
lean_object* v_res_1317_; 
v_res_1317_ = lp_mathlib_ExistsAndEq_findEq___redArg(v_u_1308_, v_a_1309_, v_P_1310_, v_path_1311_, v_a_1312_, v_a_1313_, v_a_1314_, v_a_1315_);
lean_dec(v_a_1315_);
lean_dec_ref(v_a_1314_);
lean_dec(v_a_1313_);
lean_dec_ref(v_a_1312_);
return v_res_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq(lean_object* v_u_1318_, lean_object* v_00_u03b1_1319_, lean_object* v_a_1320_, lean_object* v_P_1321_, lean_object* v_path_1322_, lean_object* v_a_1323_, lean_object* v_a_1324_, lean_object* v_a_1325_, lean_object* v_a_1326_){
_start:
{
lean_object* v___x_1328_; 
v___x_1328_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_u_1318_, v_a_1320_, v_P_1321_, v_path_1322_, v_a_1323_, v_a_1324_, v_a_1325_, v_a_1326_);
return v___x_1328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_findEq___boxed(lean_object* v_u_1329_, lean_object* v_00_u03b1_1330_, lean_object* v_a_1331_, lean_object* v_P_1332_, lean_object* v_path_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_, lean_object* v_a_1337_, lean_object* v_a_1338_){
_start:
{
lean_object* v_res_1339_; 
v_res_1339_ = lp_mathlib_ExistsAndEq_findEq(v_u_1329_, v_00_u03b1_1330_, v_a_1331_, v_P_1332_, v_path_1333_, v_a_1334_, v_a_1335_, v_a_1336_, v_a_1337_);
lean_dec(v_a_1337_);
lean_dec_ref(v_a_1336_);
lean_dec(v_a_1335_);
lean_dec_ref(v_a_1334_);
lean_dec_ref(v_00_u03b1_1330_);
return v_res_1339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists(lean_object* v_fvars_1340_, lean_object* v_body_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_){
_start:
{
if (lean_obj_tag(v_fvars_1340_) == 0)
{
lean_object* v___x_1347_; 
v___x_1347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1347_, 0, v_body_1341_);
return v___x_1347_;
}
else
{
lean_object* v_head_1348_; lean_object* v_snd_1349_; lean_object* v_tail_1350_; lean_object* v___x_1352_; uint8_t v_isShared_1353_; uint8_t v_isSharedCheck_1382_; 
v_head_1348_ = lean_ctor_get(v_fvars_1340_, 0);
lean_inc(v_head_1348_);
v_snd_1349_ = lean_ctor_get(v_head_1348_, 1);
lean_inc(v_snd_1349_);
v_tail_1350_ = lean_ctor_get(v_fvars_1340_, 1);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_fvars_1340_);
if (v_isSharedCheck_1382_ == 0)
{
lean_object* v_unused_1383_; 
v_unused_1383_ = lean_ctor_get(v_fvars_1340_, 0);
lean_dec(v_unused_1383_);
v___x_1352_ = v_fvars_1340_;
v_isShared_1353_ = v_isSharedCheck_1382_;
goto v_resetjp_1351_;
}
else
{
lean_inc(v_tail_1350_);
lean_dec(v_fvars_1340_);
v___x_1352_ = lean_box(0);
v_isShared_1353_ = v_isSharedCheck_1382_;
goto v_resetjp_1351_;
}
v_resetjp_1351_:
{
lean_object* v_fst_1354_; lean_object* v_fst_1355_; lean_object* v_snd_1356_; lean_object* v___x_1357_; 
v_fst_1354_ = lean_ctor_get(v_head_1348_, 0);
lean_inc(v_fst_1354_);
lean_dec(v_head_1348_);
v_fst_1355_ = lean_ctor_get(v_snd_1349_, 0);
lean_inc(v_fst_1355_);
v_snd_1356_ = lean_ctor_get(v_snd_1349_, 1);
lean_inc(v_snd_1356_);
lean_dec(v_snd_1349_);
v___x_1357_ = lp_mathlib_ExistsAndEq_mkNestedExists(v_tail_1350_, v_body_1341_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_);
if (lean_obj_tag(v___x_1357_) == 0)
{
lean_object* v_a_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; uint8_t v___x_1362_; uint8_t v___x_1363_; uint8_t v___x_1364_; lean_object* v___x_1365_; 
v_a_1358_ = lean_ctor_get(v___x_1357_, 0);
lean_inc(v_a_1358_);
lean_dec_ref_known(v___x_1357_, 1);
v___x_1359_ = lean_unsigned_to_nat(1u);
v___x_1360_ = lean_mk_empty_array_with_capacity(v___x_1359_);
v___x_1361_ = lean_array_push(v___x_1360_, v_snd_1356_);
v___x_1362_ = 0;
v___x_1363_ = 1;
v___x_1364_ = 1;
v___x_1365_ = l_Lean_Meta_mkLambdaFVars(v___x_1361_, v_a_1358_, v___x_1362_, v___x_1363_, v___x_1362_, v___x_1363_, v___x_1364_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_);
lean_dec_ref(v___x_1361_);
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v_a_1366_; lean_object* v___x_1368_; uint8_t v_isShared_1369_; uint8_t v_isSharedCheck_1381_; 
v_a_1366_ = lean_ctor_get(v___x_1365_, 0);
v_isSharedCheck_1381_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1368_ = v___x_1365_;
v_isShared_1369_ = v_isSharedCheck_1381_;
goto v_resetjp_1367_;
}
else
{
lean_inc(v_a_1366_);
lean_dec(v___x_1365_);
v___x_1368_ = lean_box(0);
v_isShared_1369_ = v_isSharedCheck_1381_;
goto v_resetjp_1367_;
}
v_resetjp_1367_:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1373_; 
v___x_1370_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_1371_ = lean_box(0);
if (v_isShared_1353_ == 0)
{
lean_ctor_set(v___x_1352_, 1, v___x_1371_);
lean_ctor_set(v___x_1352_, 0, v_fst_1354_);
v___x_1373_ = v___x_1352_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_fst_1354_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v___x_1371_);
v___x_1373_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1378_; 
v___x_1374_ = l_Lean_Expr_const___override(v___x_1370_, v___x_1373_);
v___x_1375_ = l_Lean_Expr_app___override(v___x_1374_, v_fst_1355_);
v___x_1376_ = l_Lean_Expr_app___override(v___x_1375_, v_a_1366_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 0, v___x_1376_);
v___x_1378_ = v___x_1368_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v___x_1376_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
else
{
lean_dec(v_fst_1355_);
lean_dec(v_fst_1354_);
lean_del_object(v___x_1352_);
return v___x_1365_;
}
}
else
{
lean_dec(v_snd_1356_);
lean_dec(v_fst_1355_);
lean_dec(v_fst_1354_);
lean_del_object(v___x_1352_);
return v___x_1357_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkNestedExists___boxed(lean_object* v_fvars_1384_, lean_object* v_body_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_, lean_object* v_a_1388_, lean_object* v_a_1389_, lean_object* v_a_1390_){
_start:
{
lean_object* v_res_1391_; 
v_res_1391_ = lp_mathlib_ExistsAndEq_mkNestedExists(v_fvars_1384_, v_body_1385_, v_a_1386_, v_a_1387_, v_a_1388_, v_a_1389_);
lean_dec(v_a_1389_);
lean_dec_ref(v_a_1388_);
lean_dec(v_a_1387_);
lean_dec_ref(v_a_1386_);
return v_res_1391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_partition_loop___at___00ExistsAndEq_Path_forResult_spec__0(lean_object* v_a_1392_, lean_object* v_a_1393_){
_start:
{
if (lean_obj_tag(v_a_1392_) == 0)
{
lean_object* v_fst_1394_; lean_object* v_snd_1395_; lean_object* v___x_1397_; uint8_t v_isShared_1398_; uint8_t v_isSharedCheck_1404_; 
v_fst_1394_ = lean_ctor_get(v_a_1393_, 0);
v_snd_1395_ = lean_ctor_get(v_a_1393_, 1);
v_isSharedCheck_1404_ = !lean_is_exclusive(v_a_1393_);
if (v_isSharedCheck_1404_ == 0)
{
v___x_1397_ = v_a_1393_;
v_isShared_1398_ = v_isSharedCheck_1404_;
goto v_resetjp_1396_;
}
else
{
lean_inc(v_snd_1395_);
lean_inc(v_fst_1394_);
lean_dec(v_a_1393_);
v___x_1397_ = lean_box(0);
v_isShared_1398_ = v_isSharedCheck_1404_;
goto v_resetjp_1396_;
}
v_resetjp_1396_:
{
lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1402_; 
v___x_1399_ = l_List_reverse___redArg(v_fst_1394_);
v___x_1400_ = l_List_reverse___redArg(v_snd_1395_);
if (v_isShared_1398_ == 0)
{
lean_ctor_set(v___x_1397_, 1, v___x_1400_);
lean_ctor_set(v___x_1397_, 0, v___x_1399_);
v___x_1402_ = v___x_1397_;
goto v_reusejp_1401_;
}
else
{
lean_object* v_reuseFailAlloc_1403_; 
v_reuseFailAlloc_1403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1403_, 0, v___x_1399_);
lean_ctor_set(v_reuseFailAlloc_1403_, 1, v___x_1400_);
v___x_1402_ = v_reuseFailAlloc_1403_;
goto v_reusejp_1401_;
}
v_reusejp_1401_:
{
return v___x_1402_;
}
}
}
else
{
lean_object* v_head_1405_; lean_object* v_tail_1406_; lean_object* v___x_1408_; uint8_t v_isShared_1409_; uint8_t v_isSharedCheck_1433_; 
v_head_1405_ = lean_ctor_get(v_a_1392_, 0);
v_tail_1406_ = lean_ctor_get(v_a_1392_, 1);
v_isSharedCheck_1433_ = !lean_is_exclusive(v_a_1392_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1408_ = v_a_1392_;
v_isShared_1409_ = v_isSharedCheck_1433_;
goto v_resetjp_1407_;
}
else
{
lean_inc(v_tail_1406_);
lean_inc(v_head_1405_);
lean_dec(v_a_1392_);
v___x_1408_ = lean_box(0);
v_isShared_1409_ = v_isSharedCheck_1433_;
goto v_resetjp_1407_;
}
v_resetjp_1407_:
{
lean_object* v_fst_1410_; lean_object* v_snd_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1432_; 
v_fst_1410_ = lean_ctor_get(v_a_1393_, 0);
v_snd_1411_ = lean_ctor_get(v_a_1393_, 1);
v_isSharedCheck_1432_ = !lean_is_exclusive(v_a_1393_);
if (v_isSharedCheck_1432_ == 0)
{
v___x_1413_ = v_a_1393_;
v_isShared_1414_ = v_isSharedCheck_1432_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_snd_1411_);
lean_inc(v_fst_1410_);
lean_dec(v_a_1393_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1432_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
uint8_t v___x_1415_; uint8_t v___x_1416_; uint8_t v___x_1417_; 
v___x_1415_ = 3;
v___x_1416_ = lean_unbox(v_head_1405_);
v___x_1417_ = lp_mathlib_ExistsAndEq_instBEqGoTo_beq(v___x_1416_, v___x_1415_);
if (v___x_1417_ == 0)
{
lean_object* v___x_1419_; 
if (v_isShared_1409_ == 0)
{
lean_ctor_set(v___x_1408_, 1, v_snd_1411_);
v___x_1419_ = v___x_1408_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v_head_1405_);
lean_ctor_set(v_reuseFailAlloc_1424_, 1, v_snd_1411_);
v___x_1419_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
lean_object* v___x_1421_; 
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 1, v___x_1419_);
v___x_1421_ = v___x_1413_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v_fst_1410_);
lean_ctor_set(v_reuseFailAlloc_1423_, 1, v___x_1419_);
v___x_1421_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
v_a_1392_ = v_tail_1406_;
v_a_1393_ = v___x_1421_;
goto _start;
}
}
}
else
{
lean_object* v___x_1426_; 
if (v_isShared_1409_ == 0)
{
lean_ctor_set(v___x_1408_, 1, v_fst_1410_);
v___x_1426_ = v___x_1408_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v_head_1405_);
lean_ctor_set(v_reuseFailAlloc_1431_, 1, v_fst_1410_);
v___x_1426_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
lean_object* v___x_1428_; 
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 0, v___x_1426_);
v___x_1428_ = v___x_1413_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1430_; 
v_reuseFailAlloc_1430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1430_, 0, v___x_1426_);
lean_ctor_set(v_reuseFailAlloc_1430_, 1, v_snd_1411_);
v___x_1428_ = v_reuseFailAlloc_1430_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
v_a_1392_ = v_tail_1406_;
v_a_1393_ = v___x_1428_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_Path_forResult(lean_object* v_path_1436_){
_start:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v_fst_1439_; lean_object* v_snd_1440_; lean_object* v___x_1441_; 
v___x_1437_ = ((lean_object*)(lp_mathlib_ExistsAndEq_Path_forResult___closed__0));
v___x_1438_ = lp_mathlib_List_partition_loop___at___00ExistsAndEq_Path_forResult_spec__0(v_path_1436_, v___x_1437_);
v_fst_1439_ = lean_ctor_get(v___x_1438_, 0);
lean_inc(v_fst_1439_);
v_snd_1440_ = lean_ctor_get(v___x_1438_, 1);
lean_inc(v_snd_1440_);
lean_dec_ref(v___x_1438_);
v___x_1441_ = l_List_appendTR___redArg(v_fst_1439_, v_snd_1440_);
return v___x_1441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(lean_object* v_msg_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v___f_1448_; lean_object* v___x_6028__overap_1449_; lean_object* v___x_1450_; 
v___f_1448_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0));
v___x_6028__overap_1449_ = lean_panic_fn_borrowed(v___f_1448_, v_msg_1442_);
lean_inc(v___y_1446_);
lean_inc_ref(v___y_1445_);
lean_inc(v___y_1444_);
lean_inc_ref(v___y_1443_);
v___x_1450_ = lean_apply_5(v___x_6028__overap_1449_, v___y_1443_, v___y_1444_, v___y_1445_, v___y_1446_, lean_box(0));
return v___x_1450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0___boxed(lean_object* v_msg_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_){
_start:
{
lean_object* v_res_1457_; 
v_res_1457_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v_msg_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_);
lean_dec(v___y_1455_);
lean_dec_ref(v___y_1454_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
return v_res_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0(lean_object* v_k_1458_, lean_object* v_b_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_){
_start:
{
lean_object* v___x_1465_; 
lean_inc(v___y_1463_);
lean_inc_ref(v___y_1462_);
lean_inc(v___y_1461_);
lean_inc_ref(v___y_1460_);
v___x_1465_ = lean_apply_6(v_k_1458_, v_b_1459_, v___y_1460_, v___y_1461_, v___y_1462_, v___y_1463_, lean_box(0));
return v___x_1465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v_k_1466_, lean_object* v_b_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_){
_start:
{
lean_object* v_res_1473_; 
v_res_1473_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0(v_k_1466_, v_b_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_);
lean_dec(v___y_1471_);
lean_dec_ref(v___y_1470_);
lean_dec(v___y_1469_);
lean_dec_ref(v___y_1468_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg(lean_object* v_name_1474_, uint8_t v_bi_1475_, lean_object* v_type_1476_, lean_object* v_k_1477_, uint8_t v_kind_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
lean_object* v___f_1484_; lean_object* v___x_1485_; 
v___f_1484_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1484_, 0, v_k_1477_);
v___x_1485_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1474_, v_bi_1475_, v_type_1476_, v___f_1484_, v_kind_1478_, v___y_1479_, v___y_1480_, v___y_1481_, v___y_1482_);
if (lean_obj_tag(v___x_1485_) == 0)
{
lean_object* v_a_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1493_; 
v_a_1486_ = lean_ctor_get(v___x_1485_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1485_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1488_ = v___x_1485_;
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_a_1486_);
lean_dec(v___x_1485_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___x_1491_; 
if (v_isShared_1489_ == 0)
{
v___x_1491_ = v___x_1488_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v_a_1486_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
else
{
lean_object* v_a_1494_; lean_object* v___x_1496_; uint8_t v_isShared_1497_; uint8_t v_isSharedCheck_1501_; 
v_a_1494_ = lean_ctor_get(v___x_1485_, 0);
v_isSharedCheck_1501_ = !lean_is_exclusive(v___x_1485_);
if (v_isSharedCheck_1501_ == 0)
{
v___x_1496_ = v___x_1485_;
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
else
{
lean_inc(v_a_1494_);
lean_dec(v___x_1485_);
v___x_1496_ = lean_box(0);
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
v_resetjp_1495_:
{
lean_object* v___x_1499_; 
if (v_isShared_1497_ == 0)
{
v___x_1499_ = v___x_1496_;
goto v_reusejp_1498_;
}
else
{
lean_object* v_reuseFailAlloc_1500_; 
v_reuseFailAlloc_1500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1500_, 0, v_a_1494_);
v___x_1499_ = v_reuseFailAlloc_1500_;
goto v_reusejp_1498_;
}
v_reusejp_1498_:
{
return v___x_1499_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg___boxed(lean_object* v_name_1502_, lean_object* v_bi_1503_, lean_object* v_type_1504_, lean_object* v_k_1505_, lean_object* v_kind_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_){
_start:
{
uint8_t v_bi_boxed_1512_; uint8_t v_kind_boxed_1513_; lean_object* v_res_1514_; 
v_bi_boxed_1512_ = lean_unbox(v_bi_1503_);
v_kind_boxed_1513_ = lean_unbox(v_kind_1506_);
v_res_1514_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg(v_name_1502_, v_bi_boxed_1512_, v_type_1504_, v_k_1505_, v_kind_boxed_1513_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_);
lean_dec(v___y_1510_);
lean_dec_ref(v___y_1509_);
lean_dec(v___y_1508_);
lean_dec_ref(v___y_1507_);
return v_res_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(lean_object* v_name_1515_, uint8_t v_bi_1516_, lean_object* v_00_u03b2_1517_, lean_object* v_k_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_){
_start:
{
uint8_t v___x_1524_; lean_object* v___x_1525_; 
v___x_1524_ = 0;
v___x_1525_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg(v_name_1515_, v_bi_1516_, v_00_u03b2_1517_, v_k_1518_, v___x_1524_, v___y_1519_, v___y_1520_, v___y_1521_, v___y_1522_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg___boxed(lean_object* v_name_1526_, lean_object* v_bi_1527_, lean_object* v_00_u03b2_1528_, lean_object* v_k_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
uint8_t v_bi_boxed_1535_; lean_object* v_res_1536_; 
v_bi_boxed_1535_ = lean_unbox(v_bi_1527_);
v_res_1536_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v_name_1526_, v_bi_boxed_1535_, v_00_u03b2_1528_, v_k_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec(v___y_1531_);
lean_dec_ref(v___y_1530_);
return v_res_1536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__4(lean_object* v___x_1537_, uint8_t v___x_1538_, lean_object* v___x_1539_, lean_object* v_fst_1540_, lean_object* v_P_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_){
_start:
{
lean_object* v___x_1547_; 
lean_inc(v___x_1539_);
v___x_1547_ = l_Lean_Meta_mkFreshExprMVar(v___x_1537_, v___x_1538_, v___x_1539_, v___y_1542_, v___y_1543_, v___y_1544_, v___y_1545_);
if (lean_obj_tag(v___x_1547_) == 0)
{
lean_object* v_a_1548_; lean_object* v___x_1549_; uint8_t v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; 
v_a_1548_ = lean_ctor_get(v___x_1547_, 0);
lean_inc_n(v_a_1548_, 2);
lean_dec_ref_known(v___x_1547_, 1);
v___x_1549_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0);
v___x_1550_ = 0;
lean_inc(v___x_1539_);
v___x_1551_ = l_Lean_Expr_forallE___override(v___x_1539_, v_a_1548_, v___x_1549_, v___x_1550_);
v___x_1552_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1551_);
v___x_1553_ = l_Lean_Meta_mkFreshExprMVar(v___x_1552_, v___x_1538_, v___x_1539_, v___y_1542_, v___y_1543_, v___y_1544_, v___y_1545_);
if (lean_obj_tag(v___x_1553_) == 0)
{
lean_object* v_a_1554_; lean_object* v_keyedConfig_1555_; uint8_t v_trackZetaDelta_1556_; lean_object* v_zetaDeltaSet_1557_; lean_object* v_lctx_1558_; lean_object* v_localInstances_1559_; lean_object* v_defEqCtx_x3f_1560_; lean_object* v_synthPendingDepth_1561_; lean_object* v_customCanUnfoldPredicate_x3f_1562_; uint8_t v_univApprox_1563_; uint8_t v_inTypeClassResolution_1564_; uint8_t v_cacheInferType_1565_; lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1629_; 
v_a_1554_ = lean_ctor_get(v___x_1553_, 0);
lean_inc(v_a_1554_);
lean_dec_ref_known(v___x_1553_, 1);
v_keyedConfig_1555_ = lean_ctor_get(v___y_1542_, 0);
v_trackZetaDelta_1556_ = lean_ctor_get_uint8(v___y_1542_, sizeof(void*)*7);
v_zetaDeltaSet_1557_ = lean_ctor_get(v___y_1542_, 1);
v_lctx_1558_ = lean_ctor_get(v___y_1542_, 2);
v_localInstances_1559_ = lean_ctor_get(v___y_1542_, 3);
v_defEqCtx_x3f_1560_ = lean_ctor_get(v___y_1542_, 4);
v_synthPendingDepth_1561_ = lean_ctor_get(v___y_1542_, 5);
v_customCanUnfoldPredicate_x3f_1562_ = lean_ctor_get(v___y_1542_, 6);
v_univApprox_1563_ = lean_ctor_get_uint8(v___y_1542_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1564_ = lean_ctor_get_uint8(v___y_1542_, sizeof(void*)*7 + 2);
v_cacheInferType_1565_ = lean_ctor_get_uint8(v___y_1542_, sizeof(void*)*7 + 3);
v_isSharedCheck_1629_ = !lean_is_exclusive(v___y_1542_);
if (v_isSharedCheck_1629_ == 0)
{
v___x_1567_ = v___y_1542_;
v_isShared_1568_ = v_isSharedCheck_1629_;
goto v_resetjp_1566_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1562_);
lean_inc(v_synthPendingDepth_1561_);
lean_inc(v_defEqCtx_x3f_1560_);
lean_inc(v_localInstances_1559_);
lean_inc(v_lctx_1558_);
lean_inc(v_zetaDeltaSet_1557_);
lean_inc(v_keyedConfig_1555_);
lean_dec(v___y_1542_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1629_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; uint8_t v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1578_; 
v___x_1569_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_1570_ = lean_box(0);
v___x_1571_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1571_, 0, v_fst_1540_);
lean_ctor_set(v___x_1571_, 1, v___x_1570_);
v___x_1572_ = l_Lean_Expr_const___override(v___x_1569_, v___x_1571_);
lean_inc(v_a_1548_);
v___x_1573_ = l_Lean_Expr_app___override(v___x_1572_, v_a_1548_);
lean_inc(v_a_1554_);
v___x_1574_ = l_Lean_Expr_app___override(v___x_1573_, v_a_1554_);
v___x_1575_ = 2;
v___x_1576_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1575_, v_keyedConfig_1555_);
if (v_isShared_1568_ == 0)
{
lean_ctor_set(v___x_1567_, 0, v___x_1576_);
v___x_1578_ = v___x_1567_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v___x_1576_);
lean_ctor_set(v_reuseFailAlloc_1628_, 1, v_zetaDeltaSet_1557_);
lean_ctor_set(v_reuseFailAlloc_1628_, 2, v_lctx_1558_);
lean_ctor_set(v_reuseFailAlloc_1628_, 3, v_localInstances_1559_);
lean_ctor_set(v_reuseFailAlloc_1628_, 4, v_defEqCtx_x3f_1560_);
lean_ctor_set(v_reuseFailAlloc_1628_, 5, v_synthPendingDepth_1561_);
lean_ctor_set(v_reuseFailAlloc_1628_, 6, v_customCanUnfoldPredicate_x3f_1562_);
lean_ctor_set_uint8(v_reuseFailAlloc_1628_, sizeof(void*)*7, v_trackZetaDelta_1556_);
lean_ctor_set_uint8(v_reuseFailAlloc_1628_, sizeof(void*)*7 + 1, v_univApprox_1563_);
lean_ctor_set_uint8(v_reuseFailAlloc_1628_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1564_);
lean_ctor_set_uint8(v_reuseFailAlloc_1628_, sizeof(void*)*7 + 3, v_cacheInferType_1565_);
v___x_1578_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
lean_object* v___x_1579_; 
v___x_1579_ = l_Lean_Meta_isExprDefEq(v___x_1574_, v_P_1541_, v___x_1578_, v___y_1543_, v___y_1544_, v___y_1545_);
lean_dec_ref(v___x_1578_);
if (lean_obj_tag(v___x_1579_) == 0)
{
lean_object* v_a_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1619_; 
v_a_1580_ = lean_ctor_get(v___x_1579_, 0);
v_isSharedCheck_1619_ = !lean_is_exclusive(v___x_1579_);
if (v_isSharedCheck_1619_ == 0)
{
v___x_1582_ = v___x_1579_;
v_isShared_1583_ = v_isSharedCheck_1619_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_a_1580_);
lean_dec(v___x_1579_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1619_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
uint8_t v___x_1584_; 
v___x_1584_ = lean_unbox(v_a_1580_);
if (v___x_1584_ == 0)
{
lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1588_; 
v___x_1585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1585_, 0, v_a_1554_);
lean_ctor_set(v___x_1585_, 1, v_a_1580_);
v___x_1586_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1586_, 0, v_a_1548_);
lean_ctor_set(v___x_1586_, 1, v___x_1585_);
if (v_isShared_1583_ == 0)
{
lean_ctor_set(v___x_1582_, 0, v___x_1586_);
v___x_1588_ = v___x_1582_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v___x_1586_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
else
{
lean_object* v___x_1590_; 
lean_del_object(v___x_1582_);
v___x_1590_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_1548_, v___y_1543_);
if (lean_obj_tag(v___x_1590_) == 0)
{
lean_object* v_a_1591_; lean_object* v___x_1592_; 
v_a_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc(v_a_1591_);
lean_dec_ref_known(v___x_1590_, 1);
v___x_1592_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_1554_, v___y_1543_);
if (lean_obj_tag(v___x_1592_) == 0)
{
lean_object* v_a_1593_; lean_object* v___x_1595_; uint8_t v_isShared_1596_; uint8_t v_isSharedCheck_1602_; 
v_a_1593_ = lean_ctor_get(v___x_1592_, 0);
v_isSharedCheck_1602_ = !lean_is_exclusive(v___x_1592_);
if (v_isSharedCheck_1602_ == 0)
{
v___x_1595_ = v___x_1592_;
v_isShared_1596_ = v_isSharedCheck_1602_;
goto v_resetjp_1594_;
}
else
{
lean_inc(v_a_1593_);
lean_dec(v___x_1592_);
v___x_1595_ = lean_box(0);
v_isShared_1596_ = v_isSharedCheck_1602_;
goto v_resetjp_1594_;
}
v_resetjp_1594_:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1600_; 
v___x_1597_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1597_, 0, v_a_1593_);
lean_ctor_set(v___x_1597_, 1, v_a_1580_);
v___x_1598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1598_, 0, v_a_1591_);
lean_ctor_set(v___x_1598_, 1, v___x_1597_);
if (v_isShared_1596_ == 0)
{
lean_ctor_set(v___x_1595_, 0, v___x_1598_);
v___x_1600_ = v___x_1595_;
goto v_reusejp_1599_;
}
else
{
lean_object* v_reuseFailAlloc_1601_; 
v_reuseFailAlloc_1601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1601_, 0, v___x_1598_);
v___x_1600_ = v_reuseFailAlloc_1601_;
goto v_reusejp_1599_;
}
v_reusejp_1599_:
{
return v___x_1600_;
}
}
}
else
{
lean_object* v_a_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1610_; 
lean_dec(v_a_1591_);
lean_dec(v_a_1580_);
v_a_1603_ = lean_ctor_get(v___x_1592_, 0);
v_isSharedCheck_1610_ = !lean_is_exclusive(v___x_1592_);
if (v_isSharedCheck_1610_ == 0)
{
v___x_1605_ = v___x_1592_;
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_a_1603_);
lean_dec(v___x_1592_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v___x_1608_; 
if (v_isShared_1606_ == 0)
{
v___x_1608_ = v___x_1605_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1609_; 
v_reuseFailAlloc_1609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1609_, 0, v_a_1603_);
v___x_1608_ = v_reuseFailAlloc_1609_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
return v___x_1608_;
}
}
}
}
else
{
lean_object* v_a_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1618_; 
lean_dec(v_a_1580_);
lean_dec(v_a_1554_);
v_a_1611_ = lean_ctor_get(v___x_1590_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v___x_1590_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1613_ = v___x_1590_;
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_a_1611_);
lean_dec(v___x_1590_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
lean_object* v___x_1616_; 
if (v_isShared_1614_ == 0)
{
v___x_1616_ = v___x_1613_;
goto v_reusejp_1615_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v_a_1611_);
v___x_1616_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1615_;
}
v_reusejp_1615_:
{
return v___x_1616_;
}
}
}
}
}
}
else
{
lean_object* v_a_1620_; lean_object* v___x_1622_; uint8_t v_isShared_1623_; uint8_t v_isSharedCheck_1627_; 
lean_dec(v_a_1554_);
lean_dec(v_a_1548_);
v_a_1620_ = lean_ctor_get(v___x_1579_, 0);
v_isSharedCheck_1627_ = !lean_is_exclusive(v___x_1579_);
if (v_isSharedCheck_1627_ == 0)
{
v___x_1622_ = v___x_1579_;
v_isShared_1623_ = v_isSharedCheck_1627_;
goto v_resetjp_1621_;
}
else
{
lean_inc(v_a_1620_);
lean_dec(v___x_1579_);
v___x_1622_ = lean_box(0);
v_isShared_1623_ = v_isSharedCheck_1627_;
goto v_resetjp_1621_;
}
v_resetjp_1621_:
{
lean_object* v___x_1625_; 
if (v_isShared_1623_ == 0)
{
v___x_1625_ = v___x_1622_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1626_; 
v_reuseFailAlloc_1626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1626_, 0, v_a_1620_);
v___x_1625_ = v_reuseFailAlloc_1626_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
return v___x_1625_;
}
}
}
}
}
}
else
{
lean_object* v_a_1630_; lean_object* v___x_1632_; uint8_t v_isShared_1633_; uint8_t v_isSharedCheck_1637_; 
lean_dec(v_a_1548_);
lean_dec_ref(v___y_1542_);
lean_dec_ref(v_P_1541_);
lean_dec(v_fst_1540_);
v_a_1630_ = lean_ctor_get(v___x_1553_, 0);
v_isSharedCheck_1637_ = !lean_is_exclusive(v___x_1553_);
if (v_isSharedCheck_1637_ == 0)
{
v___x_1632_ = v___x_1553_;
v_isShared_1633_ = v_isSharedCheck_1637_;
goto v_resetjp_1631_;
}
else
{
lean_inc(v_a_1630_);
lean_dec(v___x_1553_);
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
lean_dec_ref(v___y_1542_);
lean_dec_ref(v_P_1541_);
lean_dec(v_fst_1540_);
lean_dec(v___x_1539_);
v_a_1638_ = lean_ctor_get(v___x_1547_, 0);
v_isSharedCheck_1645_ = !lean_is_exclusive(v___x_1547_);
if (v_isSharedCheck_1645_ == 0)
{
v___x_1640_ = v___x_1547_;
v_isShared_1641_ = v_isSharedCheck_1645_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_a_1638_);
lean_dec(v___x_1547_);
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
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__4___boxed(lean_object* v___x_1646_, lean_object* v___x_1647_, lean_object* v___x_1648_, lean_object* v_fst_1649_, lean_object* v_P_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_){
_start:
{
uint8_t v___x_6788__boxed_1656_; lean_object* v_res_1657_; 
v___x_6788__boxed_1656_ = lean_unbox(v___x_1647_);
v_res_1657_ = lp_mathlib_ExistsAndEq_destruct___lam__4(v___x_1646_, v___x_6788__boxed_1656_, v___x_1648_, v_fst_1649_, v_P_1650_, v___y_1651_, v___y_1652_, v___y_1653_, v___y_1654_);
lean_dec(v___y_1654_);
lean_dec_ref(v___y_1653_);
lean_dec(v___y_1652_);
return v_res_1657_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_destruct___closed__2(void){
_start:
{
lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1660_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__1));
v___x_1661_ = lean_unsigned_to_nat(27u);
v___x_1662_ = lean_unsigned_to_nat(174u);
v___x_1663_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__0));
v___x_1664_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_1665_ = l_mkPanicMessageWithDecl(v___x_1664_, v___x_1663_, v___x_1662_, v___x_1661_, v___x_1660_);
return v___x_1665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__0___boxed(lean_object** _args){
lean_object* v_fst_1666_ = _args[0];
lean_object* v_acc_1667_ = _args[1];
lean_object* v_fst_1668_ = _args[2];
lean_object* v_goal_1669_ = _args[3];
lean_object* v_exs_1670_ = _args[4];
lean_object* v_tail_1671_ = _args[5];
lean_object* v_k_1672_ = _args[6];
lean_object* v___x_1673_ = _args[7];
lean_object* v_snd_1674_ = _args[8];
lean_object* v___x_1675_ = _args[9];
lean_object* v_h_1676_ = _args[10];
lean_object* v___x_1677_ = _args[11];
lean_object* v___x_1678_ = _args[12];
lean_object* v_h_x27_1679_ = _args[13];
lean_object* v___y_1680_ = _args[14];
lean_object* v___y_1681_ = _args[15];
lean_object* v___y_1682_ = _args[16];
lean_object* v___y_1683_ = _args[17];
lean_object* v___y_1684_ = _args[18];
_start:
{
uint8_t v___x_7061__boxed_1685_; uint8_t v___x_7064__boxed_1686_; lean_object* v_res_1687_; 
v___x_7061__boxed_1685_ = lean_unbox(v___x_1673_);
v___x_7064__boxed_1686_ = lean_unbox(v___x_1678_);
v_res_1687_ = lp_mathlib_ExistsAndEq_destruct___lam__0(v_fst_1666_, v_acc_1667_, v_fst_1668_, v_goal_1669_, v_exs_1670_, v_tail_1671_, v_k_1672_, v___x_7061__boxed_1685_, v_snd_1674_, v___x_1675_, v_h_1676_, v___x_1677_, v___x_7064__boxed_1686_, v_h_x27_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
return v_res_1687_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_destruct___closed__4(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; 
v___x_1689_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__3));
v___x_1690_ = lean_unsigned_to_nat(27u);
v___x_1691_ = lean_unsigned_to_nat(167u);
v___x_1692_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__0));
v___x_1693_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_1694_ = l_mkPanicMessageWithDecl(v___x_1693_, v___x_1692_, v___x_1691_, v___x_1690_, v___x_1689_);
return v___x_1694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__3(lean_object* v_fst_1699_, lean_object* v_leaf_1700_, lean_object* v_acc_1701_, lean_object* v_fst_1702_, lean_object* v_goal_1703_, lean_object* v_exs_1704_, lean_object* v_tail_1705_, lean_object* v_k_1706_, uint8_t v___x_1707_, lean_object* v_snd_1708_, lean_object* v___x_1709_, lean_object* v_h_1710_, lean_object* v_h_x27_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_){
_start:
{
lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; 
lean_inc_ref(v_leaf_1700_);
lean_inc(v_fst_1699_);
v___x_1717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1717_, 0, v_fst_1699_);
lean_ctor_set(v___x_1717_, 1, v_leaf_1700_);
v___x_1718_ = lean_box(0);
v___x_1719_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1719_, 0, v___x_1717_);
lean_ctor_set(v___x_1719_, 1, v___x_1718_);
v___x_1720_ = l_List_appendTR___redArg(v_acc_1701_, v___x_1719_);
lean_inc_ref(v_h_x27_1711_);
lean_inc_ref(v_goal_1703_);
lean_inc(v_fst_1702_);
v___x_1721_ = lp_mathlib_ExistsAndEq_destruct(v_fst_1702_, v_goal_1703_, v_h_x27_1711_, v_exs_1704_, v_tail_1705_, v___x_1720_, v_k_1706_, v___y_1712_, v___y_1713_, v___y_1714_, v___y_1715_);
if (lean_obj_tag(v___x_1721_) == 0)
{
lean_object* v_a_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; uint8_t v___x_1727_; uint8_t v___x_1728_; uint8_t v___x_1729_; lean_object* v___x_1730_; 
v_a_1722_ = lean_ctor_get(v___x_1721_, 0);
lean_inc(v_a_1722_);
lean_dec_ref_known(v___x_1721_, 1);
v___x_1723_ = lean_unsigned_to_nat(2u);
v___x_1724_ = lean_mk_empty_array_with_capacity(v___x_1723_);
v___x_1725_ = lean_array_push(v___x_1724_, v_leaf_1700_);
v___x_1726_ = lean_array_push(v___x_1725_, v_h_x27_1711_);
v___x_1727_ = 1;
v___x_1728_ = lean_unbox(v_snd_1708_);
v___x_1729_ = lean_unbox(v_snd_1708_);
v___x_1730_ = l_Lean_Meta_mkLambdaFVars(v___x_1726_, v_a_1722_, v___x_1707_, v___x_1728_, v___x_1707_, v___x_1729_, v___x_1727_, v___y_1712_, v___y_1713_, v___y_1714_, v___y_1715_);
lean_dec_ref(v___x_1726_);
if (lean_obj_tag(v___x_1730_) == 0)
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1746_; 
v_a_1731_ = lean_ctor_get(v___x_1730_, 0);
v_isSharedCheck_1746_ = !lean_is_exclusive(v___x_1730_);
if (v_isSharedCheck_1746_ == 0)
{
v___x_1733_ = v___x_1730_;
v_isShared_1734_ = v_isSharedCheck_1746_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1730_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1746_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1744_; 
v___x_1735_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0));
v___x_1736_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1736_, 0, v___x_1709_);
lean_ctor_set(v___x_1736_, 1, v___x_1718_);
v___x_1737_ = l_Lean_Expr_const___override(v___x_1735_, v___x_1736_);
v___x_1738_ = l_Lean_Expr_app___override(v___x_1737_, v_fst_1699_);
v___x_1739_ = l_Lean_Expr_app___override(v___x_1738_, v_fst_1702_);
v___x_1740_ = l_Lean_Expr_app___override(v___x_1739_, v_goal_1703_);
v___x_1741_ = l_Lean_Expr_app___override(v___x_1740_, v_a_1731_);
v___x_1742_ = l_Lean_Expr_app___override(v___x_1741_, v_h_1710_);
if (v_isShared_1734_ == 0)
{
lean_ctor_set(v___x_1733_, 0, v___x_1742_);
v___x_1744_ = v___x_1733_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1742_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
return v___x_1744_;
}
}
}
else
{
lean_dec_ref(v_h_1710_);
lean_dec(v___x_1709_);
lean_dec_ref(v_goal_1703_);
lean_dec(v_fst_1702_);
lean_dec(v_fst_1699_);
return v___x_1730_;
}
}
else
{
lean_dec_ref(v_h_x27_1711_);
lean_dec_ref(v_h_1710_);
lean_dec(v___x_1709_);
lean_dec_ref(v_goal_1703_);
lean_dec(v_fst_1702_);
lean_dec_ref(v_leaf_1700_);
lean_dec(v_fst_1699_);
return v___x_1721_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__3___boxed(lean_object** _args){
lean_object* v_fst_1747_ = _args[0];
lean_object* v_leaf_1748_ = _args[1];
lean_object* v_acc_1749_ = _args[2];
lean_object* v_fst_1750_ = _args[3];
lean_object* v_goal_1751_ = _args[4];
lean_object* v_exs_1752_ = _args[5];
lean_object* v_tail_1753_ = _args[6];
lean_object* v_k_1754_ = _args[7];
lean_object* v___x_1755_ = _args[8];
lean_object* v_snd_1756_ = _args[9];
lean_object* v___x_1757_ = _args[10];
lean_object* v_h_1758_ = _args[11];
lean_object* v_h_x27_1759_ = _args[12];
lean_object* v___y_1760_ = _args[13];
lean_object* v___y_1761_ = _args[14];
lean_object* v___y_1762_ = _args[15];
lean_object* v___y_1763_ = _args[16];
lean_object* v___y_1764_ = _args[17];
_start:
{
uint8_t v___x_7156__boxed_1765_; lean_object* v_res_1766_; 
v___x_7156__boxed_1765_ = lean_unbox(v___x_1755_);
v_res_1766_ = lp_mathlib_ExistsAndEq_destruct___lam__3(v_fst_1747_, v_leaf_1748_, v_acc_1749_, v_fst_1750_, v_goal_1751_, v_exs_1752_, v_tail_1753_, v_k_1754_, v___x_7156__boxed_1765_, v_snd_1756_, v___x_1757_, v_h_1758_, v_h_x27_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
lean_dec(v___y_1763_);
lean_dec_ref(v___y_1762_);
lean_dec(v___y_1761_);
lean_dec_ref(v___y_1760_);
lean_dec(v_snd_1756_);
return v_res_1766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__2(lean_object* v_fst_1767_, lean_object* v_acc_1768_, lean_object* v_fst_1769_, lean_object* v_goal_1770_, lean_object* v_exs_1771_, lean_object* v_tail_1772_, lean_object* v_k_1773_, uint8_t v___x_1774_, lean_object* v_snd_1775_, lean_object* v___x_1776_, lean_object* v_h_1777_, lean_object* v___x_1778_, uint8_t v___x_1779_, lean_object* v_leaf_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_){
_start:
{
lean_object* v___x_1786_; lean_object* v___f_1787_; lean_object* v___x_1788_; 
v___x_1786_ = lean_box(v___x_1774_);
lean_inc(v_fst_1769_);
v___f_1787_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__3___boxed), 18, 12);
lean_closure_set(v___f_1787_, 0, v_fst_1767_);
lean_closure_set(v___f_1787_, 1, v_leaf_1780_);
lean_closure_set(v___f_1787_, 2, v_acc_1768_);
lean_closure_set(v___f_1787_, 3, v_fst_1769_);
lean_closure_set(v___f_1787_, 4, v_goal_1770_);
lean_closure_set(v___f_1787_, 5, v_exs_1771_);
lean_closure_set(v___f_1787_, 6, v_tail_1772_);
lean_closure_set(v___f_1787_, 7, v_k_1773_);
lean_closure_set(v___f_1787_, 8, v___x_1786_);
lean_closure_set(v___f_1787_, 9, v_snd_1775_);
lean_closure_set(v___f_1787_, 10, v___x_1776_);
lean_closure_set(v___f_1787_, 11, v_h_1777_);
v___x_1788_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_1778_, v___x_1779_, v_fst_1769_, v___f_1787_, v___y_1781_, v___y_1782_, v___y_1783_, v___y_1784_);
return v___x_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__2___boxed(lean_object** _args){
lean_object* v_fst_1789_ = _args[0];
lean_object* v_acc_1790_ = _args[1];
lean_object* v_fst_1791_ = _args[2];
lean_object* v_goal_1792_ = _args[3];
lean_object* v_exs_1793_ = _args[4];
lean_object* v_tail_1794_ = _args[5];
lean_object* v_k_1795_ = _args[6];
lean_object* v___x_1796_ = _args[7];
lean_object* v_snd_1797_ = _args[8];
lean_object* v___x_1798_ = _args[9];
lean_object* v_h_1799_ = _args[10];
lean_object* v___x_1800_ = _args[11];
lean_object* v___x_1801_ = _args[12];
lean_object* v_leaf_1802_ = _args[13];
lean_object* v___y_1803_ = _args[14];
lean_object* v___y_1804_ = _args[15];
lean_object* v___y_1805_ = _args[16];
lean_object* v___y_1806_ = _args[17];
lean_object* v___y_1807_ = _args[18];
_start:
{
uint8_t v___x_7073__boxed_1808_; uint8_t v___x_7076__boxed_1809_; lean_object* v_res_1810_; 
v___x_7073__boxed_1808_ = lean_unbox(v___x_1796_);
v___x_7076__boxed_1809_ = lean_unbox(v___x_1801_);
v_res_1810_ = lp_mathlib_ExistsAndEq_destruct___lam__2(v_fst_1789_, v_acc_1790_, v_fst_1791_, v_goal_1792_, v_exs_1793_, v_tail_1794_, v_k_1795_, v___x_7073__boxed_1808_, v_snd_1797_, v___x_1798_, v_h_1799_, v___x_1800_, v___x_7076__boxed_1809_, v_leaf_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v___y_1804_);
lean_dec_ref(v___y_1803_);
return v_res_1810_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_destruct___closed__5(void){
_start:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; 
v___x_1811_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7));
v___x_1812_ = lean_unsigned_to_nat(24u);
v___x_1813_ = lean_unsigned_to_nat(180u);
v___x_1814_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__0));
v___x_1815_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_1816_ = l_mkPanicMessageWithDecl(v___x_1815_, v___x_1814_, v___x_1813_, v___x_1812_, v___x_1811_);
return v___x_1816_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_destruct___closed__7(void){
_start:
{
lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; 
v___x_1818_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__6));
v___x_1819_ = lean_unsigned_to_nat(12u);
v___x_1820_ = lean_unsigned_to_nat(157u);
v___x_1821_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__0));
v___x_1822_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_1823_ = l_mkPanicMessageWithDecl(v___x_1822_, v___x_1821_, v___x_1820_, v___x_1819_, v___x_1818_);
return v___x_1823_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_destruct___closed__8(void){
_start:
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; 
v___x_1824_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__9));
v___x_1825_ = lean_unsigned_to_nat(8u);
v___x_1826_ = lean_unsigned_to_nat(160u);
v___x_1827_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__0));
v___x_1828_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_1829_ = l_mkPanicMessageWithDecl(v___x_1828_, v___x_1827_, v___x_1826_, v___x_1825_, v___x_1824_);
return v___x_1829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5(lean_object* v___x_1833_, lean_object* v_goal_1834_, lean_object* v_tail_1835_, lean_object* v_tail_1836_, lean_object* v_acc_1837_, lean_object* v_k_1838_, lean_object* v_snd_1839_, uint8_t v___x_1840_, lean_object* v_snd_1841_, lean_object* v_fst_1842_, lean_object* v_fst_1843_, lean_object* v_fst_1844_, lean_object* v_h_1845_, lean_object* v_h_x27_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_){
_start:
{
lean_object* v___x_1852_; 
lean_inc_ref(v_h_x27_1846_);
lean_inc_ref(v_goal_1834_);
v___x_1852_ = lp_mathlib_ExistsAndEq_destruct(v___x_1833_, v_goal_1834_, v_h_x27_1846_, v_tail_1835_, v_tail_1836_, v_acc_1837_, v_k_1838_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_);
if (lean_obj_tag(v___x_1852_) == 0)
{
lean_object* v_a_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; uint8_t v___x_1858_; uint8_t v___x_1859_; uint8_t v___x_1860_; lean_object* v___x_1861_; 
v_a_1853_ = lean_ctor_get(v___x_1852_, 0);
lean_inc(v_a_1853_);
lean_dec_ref_known(v___x_1852_, 1);
v___x_1854_ = lean_unsigned_to_nat(2u);
v___x_1855_ = lean_mk_empty_array_with_capacity(v___x_1854_);
v___x_1856_ = lean_array_push(v___x_1855_, v_snd_1839_);
v___x_1857_ = lean_array_push(v___x_1856_, v_h_x27_1846_);
v___x_1858_ = 1;
v___x_1859_ = lean_unbox(v_snd_1841_);
v___x_1860_ = lean_unbox(v_snd_1841_);
v___x_1861_ = l_Lean_Meta_mkLambdaFVars(v___x_1857_, v_a_1853_, v___x_1840_, v___x_1859_, v___x_1840_, v___x_1860_, v___x_1858_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_);
lean_dec_ref(v___x_1857_);
if (lean_obj_tag(v___x_1861_) == 0)
{
lean_object* v_a_1862_; lean_object* v___x_1864_; uint8_t v_isShared_1865_; uint8_t v_isSharedCheck_1878_; 
v_a_1862_ = lean_ctor_get(v___x_1861_, 0);
v_isSharedCheck_1878_ = !lean_is_exclusive(v___x_1861_);
if (v_isSharedCheck_1878_ == 0)
{
v___x_1864_ = v___x_1861_;
v_isShared_1865_ = v_isSharedCheck_1878_;
goto v_resetjp_1863_;
}
else
{
lean_inc(v_a_1862_);
lean_dec(v___x_1861_);
v___x_1864_ = lean_box(0);
v_isShared_1865_ = v_isSharedCheck_1878_;
goto v_resetjp_1863_;
}
v_resetjp_1863_:
{
lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1876_; 
v___x_1866_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___lam__5___closed__1));
v___x_1867_ = lean_box(0);
v___x_1868_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1868_, 0, v_fst_1842_);
lean_ctor_set(v___x_1868_, 1, v___x_1867_);
v___x_1869_ = l_Lean_Expr_const___override(v___x_1866_, v___x_1868_);
v___x_1870_ = l_Lean_Expr_app___override(v___x_1869_, v_fst_1843_);
v___x_1871_ = l_Lean_Expr_app___override(v___x_1870_, v_fst_1844_);
v___x_1872_ = l_Lean_Expr_app___override(v___x_1871_, v_goal_1834_);
v___x_1873_ = l_Lean_Expr_app___override(v___x_1872_, v_h_1845_);
v___x_1874_ = l_Lean_Expr_app___override(v___x_1873_, v_a_1862_);
if (v_isShared_1865_ == 0)
{
lean_ctor_set(v___x_1864_, 0, v___x_1874_);
v___x_1876_ = v___x_1864_;
goto v_reusejp_1875_;
}
else
{
lean_object* v_reuseFailAlloc_1877_; 
v_reuseFailAlloc_1877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1877_, 0, v___x_1874_);
v___x_1876_ = v_reuseFailAlloc_1877_;
goto v_reusejp_1875_;
}
v_reusejp_1875_:
{
return v___x_1876_;
}
}
}
else
{
lean_dec_ref(v_h_1845_);
lean_dec(v_fst_1844_);
lean_dec(v_fst_1843_);
lean_dec(v_fst_1842_);
lean_dec_ref(v_goal_1834_);
return v___x_1861_;
}
}
else
{
lean_dec_ref(v_h_x27_1846_);
lean_dec_ref(v_h_1845_);
lean_dec(v_fst_1844_);
lean_dec(v_fst_1843_);
lean_dec(v_fst_1842_);
lean_dec_ref(v_snd_1839_);
lean_dec_ref(v_goal_1834_);
return v___x_1852_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__5___boxed(lean_object** _args){
lean_object* v___x_1879_ = _args[0];
lean_object* v_goal_1880_ = _args[1];
lean_object* v_tail_1881_ = _args[2];
lean_object* v_tail_1882_ = _args[3];
lean_object* v_acc_1883_ = _args[4];
lean_object* v_k_1884_ = _args[5];
lean_object* v_snd_1885_ = _args[6];
lean_object* v___x_1886_ = _args[7];
lean_object* v_snd_1887_ = _args[8];
lean_object* v_fst_1888_ = _args[9];
lean_object* v_fst_1889_ = _args[10];
lean_object* v_fst_1890_ = _args[11];
lean_object* v_h_1891_ = _args[12];
lean_object* v_h_x27_1892_ = _args[13];
lean_object* v___y_1893_ = _args[14];
lean_object* v___y_1894_ = _args[15];
lean_object* v___y_1895_ = _args[16];
lean_object* v___y_1896_ = _args[17];
lean_object* v___y_1897_ = _args[18];
_start:
{
uint8_t v___x_7092__boxed_1898_; lean_object* v_res_1899_; 
v___x_7092__boxed_1898_ = lean_unbox(v___x_1886_);
v_res_1899_ = lp_mathlib_ExistsAndEq_destruct___lam__5(v___x_1879_, v_goal_1880_, v_tail_1881_, v_tail_1882_, v_acc_1883_, v_k_1884_, v_snd_1885_, v___x_7092__boxed_1898_, v_snd_1887_, v_fst_1888_, v_fst_1889_, v_fst_1890_, v_h_1891_, v_h_x27_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_);
lean_dec(v___y_1896_);
lean_dec_ref(v___y_1895_);
lean_dec(v___y_1894_);
lean_dec_ref(v___y_1893_);
lean_dec(v_snd_1887_);
return v_res_1899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct(lean_object* v_P_1900_, lean_object* v_goal_1901_, lean_object* v_h_1902_, lean_object* v_exs_1903_, lean_object* v_path_1904_, lean_object* v_acc_1905_, lean_object* v_k_1906_, lean_object* v_a_1907_, lean_object* v_a_1908_, lean_object* v_a_1909_, lean_object* v_a_1910_){
_start:
{
if (lean_obj_tag(v_path_1904_) == 0)
{
lean_object* v___x_1912_; lean_object* v___x_1913_; 
lean_dec(v_exs_1903_);
lean_dec_ref(v_goal_1901_);
v___x_1912_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1912_, 0, v_P_1900_);
lean_ctor_set(v___x_1912_, 1, v_h_1902_);
lean_inc(v_a_1910_);
lean_inc_ref(v_a_1909_);
lean_inc(v_a_1908_);
lean_inc_ref(v_a_1907_);
v___x_1913_ = lean_apply_7(v_k_1906_, v_acc_1905_, v___x_1912_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_, lean_box(0));
return v___x_1913_;
}
else
{
lean_object* v_head_1914_; uint8_t v___x_1915_; 
v_head_1914_ = lean_ctor_get(v_path_1904_, 0);
v___x_1915_ = lean_unbox(v_head_1914_);
switch(v___x_1915_)
{
case 0:
{
lean_object* v_tail_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; uint8_t v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___f_1922_; uint8_t v___x_1923_; lean_object* v___x_1924_; 
v_tail_1916_ = lean_ctor_get(v_path_1904_, 1);
lean_inc(v_tail_1916_);
lean_dec_ref_known(v_path_1904_, 2);
v___x_1917_ = lean_box(0);
v___x_1918_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_1919_ = 0;
v___x_1920_ = lean_box(0);
v___x_1921_ = lean_box(v___x_1919_);
v___f_1922_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1922_, 0, v___x_1918_);
lean_closure_set(v___f_1922_, 1, v___x_1921_);
lean_closure_set(v___f_1922_, 2, v___x_1920_);
lean_closure_set(v___f_1922_, 3, v_P_1900_);
v___x_1923_ = 0;
v___x_1924_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1922_, v___x_1923_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
if (lean_obj_tag(v___x_1924_) == 0)
{
lean_object* v_a_1925_; lean_object* v_snd_1926_; lean_object* v_snd_1927_; uint8_t v___x_1928_; 
v_a_1925_ = lean_ctor_get(v___x_1924_, 0);
lean_inc(v_a_1925_);
lean_dec_ref_known(v___x_1924_, 1);
v_snd_1926_ = lean_ctor_get(v_a_1925_, 1);
lean_inc(v_snd_1926_);
v_snd_1927_ = lean_ctor_get(v_snd_1926_, 1);
lean_inc(v_snd_1927_);
v___x_1928_ = lean_unbox(v_snd_1927_);
if (v___x_1928_ == 0)
{
lean_object* v___x_1929_; lean_object* v___x_1930_; 
lean_dec(v_snd_1927_);
lean_dec(v_snd_1926_);
lean_dec(v_a_1925_);
lean_dec(v_tail_1916_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec(v_exs_1903_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v___x_1929_ = lean_obj_once(&lp_mathlib_ExistsAndEq_destruct___closed__2, &lp_mathlib_ExistsAndEq_destruct___closed__2_once, _init_lp_mathlib_ExistsAndEq_destruct___closed__2);
v___x_1930_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_1929_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1930_;
}
else
{
lean_object* v_fst_1931_; lean_object* v_fst_1932_; uint8_t v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___f_1936_; lean_object* v___x_1937_; 
v_fst_1931_ = lean_ctor_get(v_a_1925_, 0);
lean_inc_n(v_fst_1931_, 2);
lean_dec(v_a_1925_);
v_fst_1932_ = lean_ctor_get(v_snd_1926_, 0);
lean_inc(v_fst_1932_);
lean_dec(v_snd_1926_);
v___x_1933_ = 0;
v___x_1934_ = lean_box(v___x_1923_);
v___x_1935_ = lean_box(v___x_1933_);
v___f_1936_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__0___boxed), 19, 13);
lean_closure_set(v___f_1936_, 0, v_fst_1932_);
lean_closure_set(v___f_1936_, 1, v_acc_1905_);
lean_closure_set(v___f_1936_, 2, v_fst_1931_);
lean_closure_set(v___f_1936_, 3, v_goal_1901_);
lean_closure_set(v___f_1936_, 4, v_exs_1903_);
lean_closure_set(v___f_1936_, 5, v_tail_1916_);
lean_closure_set(v___f_1936_, 6, v_k_1906_);
lean_closure_set(v___f_1936_, 7, v___x_1934_);
lean_closure_set(v___f_1936_, 8, v_snd_1927_);
lean_closure_set(v___f_1936_, 9, v___x_1917_);
lean_closure_set(v___f_1936_, 10, v_h_1902_);
lean_closure_set(v___f_1936_, 11, v___x_1920_);
lean_closure_set(v___f_1936_, 12, v___x_1935_);
v___x_1937_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_1920_, v___x_1933_, v_fst_1931_, v___f_1936_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1937_;
}
}
else
{
lean_object* v_a_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1945_; 
lean_dec(v_tail_1916_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec(v_exs_1903_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v_a_1938_ = lean_ctor_get(v___x_1924_, 0);
v_isSharedCheck_1945_ = !lean_is_exclusive(v___x_1924_);
if (v_isSharedCheck_1945_ == 0)
{
v___x_1940_ = v___x_1924_;
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_a_1938_);
lean_dec(v___x_1924_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v___x_1943_; 
if (v_isShared_1941_ == 0)
{
v___x_1943_ = v___x_1940_;
goto v_reusejp_1942_;
}
else
{
lean_object* v_reuseFailAlloc_1944_; 
v_reuseFailAlloc_1944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1944_, 0, v_a_1938_);
v___x_1943_ = v_reuseFailAlloc_1944_;
goto v_reusejp_1942_;
}
v_reusejp_1942_:
{
return v___x_1943_;
}
}
}
}
case 1:
{
lean_object* v_tail_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; uint8_t v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___f_1952_; uint8_t v___x_1953_; lean_object* v___x_1954_; 
v_tail_1946_ = lean_ctor_get(v_path_1904_, 1);
lean_inc(v_tail_1946_);
lean_dec_ref_known(v_path_1904_, 2);
v___x_1947_ = lean_box(0);
v___x_1948_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_1949_ = 0;
v___x_1950_ = lean_box(0);
v___x_1951_ = lean_box(v___x_1949_);
v___f_1952_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1952_, 0, v___x_1948_);
lean_closure_set(v___f_1952_, 1, v___x_1951_);
lean_closure_set(v___f_1952_, 2, v___x_1950_);
lean_closure_set(v___f_1952_, 3, v_P_1900_);
v___x_1953_ = 0;
v___x_1954_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1952_, v___x_1953_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
if (lean_obj_tag(v___x_1954_) == 0)
{
lean_object* v_a_1955_; lean_object* v_snd_1956_; lean_object* v_snd_1957_; uint8_t v___x_1958_; 
v_a_1955_ = lean_ctor_get(v___x_1954_, 0);
lean_inc(v_a_1955_);
lean_dec_ref_known(v___x_1954_, 1);
v_snd_1956_ = lean_ctor_get(v_a_1955_, 1);
lean_inc(v_snd_1956_);
v_snd_1957_ = lean_ctor_get(v_snd_1956_, 1);
lean_inc(v_snd_1957_);
v___x_1958_ = lean_unbox(v_snd_1957_);
if (v___x_1958_ == 0)
{
lean_object* v___x_1959_; lean_object* v___x_1960_; 
lean_dec(v_snd_1957_);
lean_dec(v_snd_1956_);
lean_dec(v_a_1955_);
lean_dec(v_tail_1946_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec(v_exs_1903_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v___x_1959_ = lean_obj_once(&lp_mathlib_ExistsAndEq_destruct___closed__4, &lp_mathlib_ExistsAndEq_destruct___closed__4_once, _init_lp_mathlib_ExistsAndEq_destruct___closed__4);
v___x_1960_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_1959_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1960_;
}
else
{
lean_object* v_fst_1961_; lean_object* v_fst_1962_; uint8_t v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___f_1966_; lean_object* v___x_1967_; 
v_fst_1961_ = lean_ctor_get(v_a_1955_, 0);
lean_inc_n(v_fst_1961_, 2);
lean_dec(v_a_1955_);
v_fst_1962_ = lean_ctor_get(v_snd_1956_, 0);
lean_inc(v_fst_1962_);
lean_dec(v_snd_1956_);
v___x_1963_ = 0;
v___x_1964_ = lean_box(v___x_1953_);
v___x_1965_ = lean_box(v___x_1963_);
v___f_1966_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__2___boxed), 19, 13);
lean_closure_set(v___f_1966_, 0, v_fst_1961_);
lean_closure_set(v___f_1966_, 1, v_acc_1905_);
lean_closure_set(v___f_1966_, 2, v_fst_1962_);
lean_closure_set(v___f_1966_, 3, v_goal_1901_);
lean_closure_set(v___f_1966_, 4, v_exs_1903_);
lean_closure_set(v___f_1966_, 5, v_tail_1946_);
lean_closure_set(v___f_1966_, 6, v_k_1906_);
lean_closure_set(v___f_1966_, 7, v___x_1964_);
lean_closure_set(v___f_1966_, 8, v_snd_1957_);
lean_closure_set(v___f_1966_, 9, v___x_1947_);
lean_closure_set(v___f_1966_, 10, v_h_1902_);
lean_closure_set(v___f_1966_, 11, v___x_1950_);
lean_closure_set(v___f_1966_, 12, v___x_1965_);
v___x_1967_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_1950_, v___x_1963_, v_fst_1961_, v___f_1966_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1967_;
}
}
else
{
lean_object* v_a_1968_; lean_object* v___x_1970_; uint8_t v_isShared_1971_; uint8_t v_isSharedCheck_1975_; 
lean_dec(v_tail_1946_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec(v_exs_1903_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v_a_1968_ = lean_ctor_get(v___x_1954_, 0);
v_isSharedCheck_1975_ = !lean_is_exclusive(v___x_1954_);
if (v_isSharedCheck_1975_ == 0)
{
v___x_1970_ = v___x_1954_;
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
else
{
lean_inc(v_a_1968_);
lean_dec(v___x_1954_);
v___x_1970_ = lean_box(0);
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
v_resetjp_1969_:
{
lean_object* v___x_1973_; 
if (v_isShared_1971_ == 0)
{
v___x_1973_ = v___x_1970_;
goto v_reusejp_1972_;
}
else
{
lean_object* v_reuseFailAlloc_1974_; 
v_reuseFailAlloc_1974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1974_, 0, v_a_1968_);
v___x_1973_ = v_reuseFailAlloc_1974_;
goto v_reusejp_1972_;
}
v_reusejp_1972_:
{
return v___x_1973_;
}
}
}
}
case 2:
{
lean_object* v___x_1976_; lean_object* v___x_1977_; 
lean_dec_ref_known(v_path_1904_, 2);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec(v_exs_1903_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
lean_dec_ref(v_P_1900_);
v___x_1976_ = lean_obj_once(&lp_mathlib_ExistsAndEq_destruct___closed__5, &lp_mathlib_ExistsAndEq_destruct___closed__5_once, _init_lp_mathlib_ExistsAndEq_destruct___closed__5);
v___x_1977_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_1976_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1977_;
}
default: 
{
if (lean_obj_tag(v_exs_1903_) == 0)
{
lean_object* v___x_1978_; lean_object* v___x_1979_; 
lean_dec_ref_known(v_path_1904_, 2);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
lean_dec_ref(v_P_1900_);
v___x_1978_ = lean_obj_once(&lp_mathlib_ExistsAndEq_destruct___closed__7, &lp_mathlib_ExistsAndEq_destruct___closed__7_once, _init_lp_mathlib_ExistsAndEq_destruct___closed__7);
v___x_1979_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_1978_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_1979_;
}
else
{
lean_object* v_head_1980_; lean_object* v_snd_1981_; lean_object* v_tail_1982_; lean_object* v_tail_1983_; lean_object* v___x_1985_; uint8_t v_isShared_1986_; uint8_t v_isSharedCheck_2023_; 
v_head_1980_ = lean_ctor_get(v_exs_1903_, 0);
lean_inc(v_head_1980_);
v_snd_1981_ = lean_ctor_get(v_head_1980_, 1);
lean_inc(v_snd_1981_);
v_tail_1982_ = lean_ctor_get(v_path_1904_, 1);
lean_inc(v_tail_1982_);
lean_dec_ref_known(v_path_1904_, 2);
v_tail_1983_ = lean_ctor_get(v_exs_1903_, 1);
v_isSharedCheck_2023_ = !lean_is_exclusive(v_exs_1903_);
if (v_isSharedCheck_2023_ == 0)
{
lean_object* v_unused_2024_; 
v_unused_2024_ = lean_ctor_get(v_exs_1903_, 0);
lean_dec(v_unused_2024_);
v___x_1985_ = v_exs_1903_;
v_isShared_1986_ = v_isSharedCheck_2023_;
goto v_resetjp_1984_;
}
else
{
lean_inc(v_tail_1983_);
lean_dec(v_exs_1903_);
v___x_1985_ = lean_box(0);
v_isShared_1986_ = v_isSharedCheck_2023_;
goto v_resetjp_1984_;
}
v_resetjp_1984_:
{
lean_object* v_fst_1987_; lean_object* v_snd_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; uint8_t v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___f_1994_; uint8_t v___x_1995_; lean_object* v___x_1996_; 
v_fst_1987_ = lean_ctor_get(v_head_1980_, 0);
lean_inc_n(v_fst_1987_, 3);
lean_dec(v_head_1980_);
v_snd_1988_ = lean_ctor_get(v_snd_1981_, 1);
lean_inc(v_snd_1988_);
lean_dec(v_snd_1981_);
v___x_1989_ = l_Lean_Expr_sort___override(v_fst_1987_);
v___x_1990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1990_, 0, v___x_1989_);
v___x_1991_ = 0;
v___x_1992_ = lean_box(0);
v___x_1993_ = lean_box(v___x_1991_);
v___f_1994_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__4___boxed), 10, 5);
lean_closure_set(v___f_1994_, 0, v___x_1990_);
lean_closure_set(v___f_1994_, 1, v___x_1993_);
lean_closure_set(v___f_1994_, 2, v___x_1992_);
lean_closure_set(v___f_1994_, 3, v_fst_1987_);
lean_closure_set(v___f_1994_, 4, v_P_1900_);
v___x_1995_ = 0;
v___x_1996_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_1994_, v___x_1995_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
if (lean_obj_tag(v___x_1996_) == 0)
{
lean_object* v_a_1997_; lean_object* v_snd_1998_; lean_object* v_snd_1999_; uint8_t v___x_2000_; 
v_a_1997_ = lean_ctor_get(v___x_1996_, 0);
lean_inc(v_a_1997_);
lean_dec_ref_known(v___x_1996_, 1);
v_snd_1998_ = lean_ctor_get(v_a_1997_, 1);
lean_inc(v_snd_1998_);
v_snd_1999_ = lean_ctor_get(v_snd_1998_, 1);
lean_inc(v_snd_1999_);
v___x_2000_ = lean_unbox(v_snd_1999_);
if (v___x_2000_ == 0)
{
lean_object* v___x_2001_; lean_object* v___x_2002_; 
lean_dec(v_snd_1999_);
lean_dec(v_snd_1998_);
lean_dec(v_a_1997_);
lean_dec(v_snd_1988_);
lean_dec(v_fst_1987_);
lean_del_object(v___x_1985_);
lean_dec(v_tail_1983_);
lean_dec(v_tail_1982_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v___x_2001_ = lean_obj_once(&lp_mathlib_ExistsAndEq_destruct___closed__8, &lp_mathlib_ExistsAndEq_destruct___closed__8_once, _init_lp_mathlib_ExistsAndEq_destruct___closed__8);
v___x_2002_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2001_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_2002_;
}
else
{
lean_object* v_fst_2003_; lean_object* v_fst_2004_; uint8_t v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2008_; 
v_fst_2003_ = lean_ctor_get(v_a_1997_, 0);
lean_inc(v_fst_2003_);
lean_dec(v_a_1997_);
v_fst_2004_ = lean_ctor_get(v_snd_1998_, 0);
lean_inc(v_fst_2004_);
lean_dec(v_snd_1998_);
v___x_2005_ = 0;
v___x_2006_ = lean_box(0);
lean_inc(v_snd_1988_);
if (v_isShared_1986_ == 0)
{
lean_ctor_set(v___x_1985_, 1, v___x_2006_);
lean_ctor_set(v___x_1985_, 0, v_snd_1988_);
v___x_2008_ = v___x_1985_;
goto v_reusejp_2007_;
}
else
{
lean_object* v_reuseFailAlloc_2014_; 
v_reuseFailAlloc_2014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2014_, 0, v_snd_1988_);
lean_ctor_set(v_reuseFailAlloc_2014_, 1, v___x_2006_);
v___x_2008_ = v_reuseFailAlloc_2014_;
goto v_reusejp_2007_;
}
v_reusejp_2007_:
{
lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___f_2012_; lean_object* v___x_2013_; 
v___x_2009_ = lean_array_mk(v___x_2008_);
lean_inc(v_fst_2004_);
v___x_2010_ = l_Lean_Expr_betaRev(v_fst_2004_, v___x_2009_, v___x_1995_, v___x_1995_);
lean_dec_ref(v___x_2009_);
v___x_2011_ = lean_box(v___x_1995_);
lean_inc_ref(v___x_2010_);
v___f_2012_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__5___boxed), 19, 13);
lean_closure_set(v___f_2012_, 0, v___x_2010_);
lean_closure_set(v___f_2012_, 1, v_goal_1901_);
lean_closure_set(v___f_2012_, 2, v_tail_1983_);
lean_closure_set(v___f_2012_, 3, v_tail_1982_);
lean_closure_set(v___f_2012_, 4, v_acc_1905_);
lean_closure_set(v___f_2012_, 5, v_k_1906_);
lean_closure_set(v___f_2012_, 6, v_snd_1988_);
lean_closure_set(v___f_2012_, 7, v___x_2011_);
lean_closure_set(v___f_2012_, 8, v_snd_1999_);
lean_closure_set(v___f_2012_, 9, v_fst_1987_);
lean_closure_set(v___f_2012_, 10, v_fst_2003_);
lean_closure_set(v___f_2012_, 11, v_fst_2004_);
lean_closure_set(v___f_2012_, 12, v_h_1902_);
v___x_2013_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_1992_, v___x_2005_, v___x_2010_, v___f_2012_, v_a_1907_, v_a_1908_, v_a_1909_, v_a_1910_);
return v___x_2013_;
}
}
}
else
{
lean_object* v_a_2015_; lean_object* v___x_2017_; uint8_t v_isShared_2018_; uint8_t v_isSharedCheck_2022_; 
lean_dec(v_snd_1988_);
lean_dec(v_fst_1987_);
lean_del_object(v___x_1985_);
lean_dec(v_tail_1983_);
lean_dec(v_tail_1982_);
lean_dec_ref(v_k_1906_);
lean_dec(v_acc_1905_);
lean_dec_ref(v_h_1902_);
lean_dec_ref(v_goal_1901_);
v_a_2015_ = lean_ctor_get(v___x_1996_, 0);
v_isSharedCheck_2022_ = !lean_is_exclusive(v___x_1996_);
if (v_isSharedCheck_2022_ == 0)
{
v___x_2017_ = v___x_1996_;
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
else
{
lean_inc(v_a_2015_);
lean_dec(v___x_1996_);
v___x_2017_ = lean_box(0);
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
v_resetjp_2016_:
{
lean_object* v___x_2020_; 
if (v_isShared_2018_ == 0)
{
v___x_2020_ = v___x_2017_;
goto v_reusejp_2019_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v_a_2015_);
v___x_2020_ = v_reuseFailAlloc_2021_;
goto v_reusejp_2019_;
}
v_reusejp_2019_:
{
return v___x_2020_;
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
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__1(lean_object* v_fst_2025_, lean_object* v_acc_2026_, lean_object* v_fst_2027_, lean_object* v_goal_2028_, lean_object* v_h_x27_2029_, lean_object* v_exs_2030_, lean_object* v_tail_2031_, lean_object* v_k_2032_, uint8_t v___x_2033_, lean_object* v_snd_2034_, lean_object* v___x_2035_, lean_object* v_h_2036_, lean_object* v_leaf_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_){
_start:
{
lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; 
lean_inc_ref(v_leaf_2037_);
lean_inc(v_fst_2025_);
v___x_2043_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2043_, 0, v_fst_2025_);
lean_ctor_set(v___x_2043_, 1, v_leaf_2037_);
v___x_2044_ = lean_box(0);
v___x_2045_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2045_, 0, v___x_2043_);
lean_ctor_set(v___x_2045_, 1, v___x_2044_);
v___x_2046_ = l_List_appendTR___redArg(v_acc_2026_, v___x_2045_);
lean_inc_ref(v_h_x27_2029_);
lean_inc_ref(v_goal_2028_);
lean_inc(v_fst_2027_);
v___x_2047_ = lp_mathlib_ExistsAndEq_destruct(v_fst_2027_, v_goal_2028_, v_h_x27_2029_, v_exs_2030_, v_tail_2031_, v___x_2046_, v_k_2032_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_);
if (lean_obj_tag(v___x_2047_) == 0)
{
lean_object* v_a_2048_; lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; uint8_t v___x_2053_; uint8_t v___x_2054_; uint8_t v___x_2055_; lean_object* v___x_2056_; 
v_a_2048_ = lean_ctor_get(v___x_2047_, 0);
lean_inc(v_a_2048_);
lean_dec_ref_known(v___x_2047_, 1);
v___x_2049_ = lean_unsigned_to_nat(2u);
v___x_2050_ = lean_mk_empty_array_with_capacity(v___x_2049_);
v___x_2051_ = lean_array_push(v___x_2050_, v_h_x27_2029_);
v___x_2052_ = lean_array_push(v___x_2051_, v_leaf_2037_);
v___x_2053_ = 1;
v___x_2054_ = lean_unbox(v_snd_2034_);
v___x_2055_ = lean_unbox(v_snd_2034_);
v___x_2056_ = l_Lean_Meta_mkLambdaFVars(v___x_2052_, v_a_2048_, v___x_2033_, v___x_2054_, v___x_2033_, v___x_2055_, v___x_2053_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_);
lean_dec_ref(v___x_2052_);
if (lean_obj_tag(v___x_2056_) == 0)
{
lean_object* v_a_2057_; lean_object* v___x_2059_; uint8_t v_isShared_2060_; uint8_t v_isSharedCheck_2072_; 
v_a_2057_ = lean_ctor_get(v___x_2056_, 0);
v_isSharedCheck_2072_ = !lean_is_exclusive(v___x_2056_);
if (v_isSharedCheck_2072_ == 0)
{
v___x_2059_ = v___x_2056_;
v_isShared_2060_ = v_isSharedCheck_2072_;
goto v_resetjp_2058_;
}
else
{
lean_inc(v_a_2057_);
lean_dec(v___x_2056_);
v___x_2059_ = lean_box(0);
v_isShared_2060_ = v_isSharedCheck_2072_;
goto v_resetjp_2058_;
}
v_resetjp_2058_:
{
lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2070_; 
v___x_2061_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___lam__1___closed__0));
v___x_2062_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2035_);
lean_ctor_set(v___x_2062_, 1, v___x_2044_);
v___x_2063_ = l_Lean_Expr_const___override(v___x_2061_, v___x_2062_);
v___x_2064_ = l_Lean_Expr_app___override(v___x_2063_, v_fst_2027_);
v___x_2065_ = l_Lean_Expr_app___override(v___x_2064_, v_fst_2025_);
v___x_2066_ = l_Lean_Expr_app___override(v___x_2065_, v_goal_2028_);
v___x_2067_ = l_Lean_Expr_app___override(v___x_2066_, v_a_2057_);
v___x_2068_ = l_Lean_Expr_app___override(v___x_2067_, v_h_2036_);
if (v_isShared_2060_ == 0)
{
lean_ctor_set(v___x_2059_, 0, v___x_2068_);
v___x_2070_ = v___x_2059_;
goto v_reusejp_2069_;
}
else
{
lean_object* v_reuseFailAlloc_2071_; 
v_reuseFailAlloc_2071_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2071_, 0, v___x_2068_);
v___x_2070_ = v_reuseFailAlloc_2071_;
goto v_reusejp_2069_;
}
v_reusejp_2069_:
{
return v___x_2070_;
}
}
}
else
{
lean_dec_ref(v_h_2036_);
lean_dec(v___x_2035_);
lean_dec_ref(v_goal_2028_);
lean_dec(v_fst_2027_);
lean_dec(v_fst_2025_);
return v___x_2056_;
}
}
else
{
lean_dec_ref(v_leaf_2037_);
lean_dec_ref(v_h_2036_);
lean_dec(v___x_2035_);
lean_dec_ref(v_h_x27_2029_);
lean_dec_ref(v_goal_2028_);
lean_dec(v_fst_2027_);
lean_dec(v_fst_2025_);
return v___x_2047_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__1___boxed(lean_object** _args){
lean_object* v_fst_2073_ = _args[0];
lean_object* v_acc_2074_ = _args[1];
lean_object* v_fst_2075_ = _args[2];
lean_object* v_goal_2076_ = _args[3];
lean_object* v_h_x27_2077_ = _args[4];
lean_object* v_exs_2078_ = _args[5];
lean_object* v_tail_2079_ = _args[6];
lean_object* v_k_2080_ = _args[7];
lean_object* v___x_2081_ = _args[8];
lean_object* v_snd_2082_ = _args[9];
lean_object* v___x_2083_ = _args[10];
lean_object* v_h_2084_ = _args[11];
lean_object* v_leaf_2085_ = _args[12];
lean_object* v___y_2086_ = _args[13];
lean_object* v___y_2087_ = _args[14];
lean_object* v___y_2088_ = _args[15];
lean_object* v___y_2089_ = _args[16];
lean_object* v___y_2090_ = _args[17];
_start:
{
uint8_t v___x_7124__boxed_2091_; lean_object* v_res_2092_; 
v___x_7124__boxed_2091_ = lean_unbox(v___x_2081_);
v_res_2092_ = lp_mathlib_ExistsAndEq_destruct___lam__1(v_fst_2073_, v_acc_2074_, v_fst_2075_, v_goal_2076_, v_h_x27_2077_, v_exs_2078_, v_tail_2079_, v_k_2080_, v___x_7124__boxed_2091_, v_snd_2082_, v___x_2083_, v_h_2084_, v_leaf_2085_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_);
lean_dec(v___y_2089_);
lean_dec_ref(v___y_2088_);
lean_dec(v___y_2087_);
lean_dec_ref(v___y_2086_);
lean_dec(v_snd_2082_);
return v_res_2092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___lam__0(lean_object* v_fst_2093_, lean_object* v_acc_2094_, lean_object* v_fst_2095_, lean_object* v_goal_2096_, lean_object* v_exs_2097_, lean_object* v_tail_2098_, lean_object* v_k_2099_, uint8_t v___x_2100_, lean_object* v_snd_2101_, lean_object* v___x_2102_, lean_object* v_h_2103_, lean_object* v___x_2104_, uint8_t v___x_2105_, lean_object* v_h_x27_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_){
_start:
{
lean_object* v___x_2112_; lean_object* v___f_2113_; lean_object* v___x_2114_; 
v___x_2112_ = lean_box(v___x_2100_);
lean_inc(v_fst_2093_);
v___f_2113_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_destruct___lam__1___boxed), 18, 12);
lean_closure_set(v___f_2113_, 0, v_fst_2093_);
lean_closure_set(v___f_2113_, 1, v_acc_2094_);
lean_closure_set(v___f_2113_, 2, v_fst_2095_);
lean_closure_set(v___f_2113_, 3, v_goal_2096_);
lean_closure_set(v___f_2113_, 4, v_h_x27_2106_);
lean_closure_set(v___f_2113_, 5, v_exs_2097_);
lean_closure_set(v___f_2113_, 6, v_tail_2098_);
lean_closure_set(v___f_2113_, 7, v_k_2099_);
lean_closure_set(v___f_2113_, 8, v___x_2112_);
lean_closure_set(v___f_2113_, 9, v_snd_2101_);
lean_closure_set(v___f_2113_, 10, v___x_2102_);
lean_closure_set(v___f_2113_, 11, v_h_2103_);
v___x_2114_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_2104_, v___x_2105_, v_fst_2093_, v___f_2113_, v___y_2107_, v___y_2108_, v___y_2109_, v___y_2110_);
return v___x_2114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_destruct___boxed(lean_object* v_P_2115_, lean_object* v_goal_2116_, lean_object* v_h_2117_, lean_object* v_exs_2118_, lean_object* v_path_2119_, lean_object* v_acc_2120_, lean_object* v_k_2121_, lean_object* v_a_2122_, lean_object* v_a_2123_, lean_object* v_a_2124_, lean_object* v_a_2125_, lean_object* v_a_2126_){
_start:
{
lean_object* v_res_2127_; 
v_res_2127_ = lp_mathlib_ExistsAndEq_destruct(v_P_2115_, v_goal_2116_, v_h_2117_, v_exs_2118_, v_path_2119_, v_acc_2120_, v_k_2121_, v_a_2122_, v_a_2123_, v_a_2124_, v_a_2125_);
lean_dec(v_a_2125_);
lean_dec_ref(v_a_2124_);
lean_dec(v_a_2123_);
lean_dec_ref(v_a_2122_);
return v_res_2127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1(lean_object* v_00_u03b1_2128_, lean_object* v_name_2129_, uint8_t v_bi_2130_, lean_object* v_type_2131_, lean_object* v_k_2132_, uint8_t v_kind_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_){
_start:
{
lean_object* v___x_2139_; 
v___x_2139_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___redArg(v_name_2129_, v_bi_2130_, v_type_2131_, v_k_2132_, v_kind_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_);
return v___x_2139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1___boxed(lean_object* v_00_u03b1_2140_, lean_object* v_name_2141_, lean_object* v_bi_2142_, lean_object* v_type_2143_, lean_object* v_k_2144_, lean_object* v_kind_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_){
_start:
{
uint8_t v_bi_boxed_2151_; uint8_t v_kind_boxed_2152_; lean_object* v_res_2153_; 
v_bi_boxed_2151_ = lean_unbox(v_bi_2142_);
v_kind_boxed_2152_ = lean_unbox(v_kind_2145_);
v_res_2153_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1_spec__1(v_00_u03b1_2140_, v_name_2141_, v_bi_boxed_2151_, v_type_2143_, v_k_2144_, v_kind_boxed_2152_, v___y_2146_, v___y_2147_, v___y_2148_, v___y_2149_);
lean_dec(v___y_2149_);
lean_dec_ref(v___y_2148_);
lean_dec(v___y_2147_);
lean_dec_ref(v___y_2146_);
return v_res_2153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1(lean_object* v_u_2154_, lean_object* v_00_u03b1_2155_, lean_object* v_name_2156_, uint8_t v_bi_2157_, lean_object* v_00_u03b2_2158_, lean_object* v_k_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_){
_start:
{
lean_object* v___x_2165_; 
v___x_2165_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v_name_2156_, v_bi_2157_, v_00_u03b2_2158_, v_k_2159_, v___y_2160_, v___y_2161_, v___y_2162_, v___y_2163_);
return v___x_2165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___boxed(lean_object* v_u_2166_, lean_object* v_00_u03b1_2167_, lean_object* v_name_2168_, lean_object* v_bi_2169_, lean_object* v_00_u03b2_2170_, lean_object* v_k_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_){
_start:
{
uint8_t v_bi_boxed_2177_; lean_object* v_res_2178_; 
v_bi_boxed_2177_ = lean_unbox(v_bi_2169_);
v_res_2178_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1(v_u_2166_, v_00_u03b1_2167_, v_name_2168_, v_bi_boxed_2177_, v_00_u03b2_2170_, v_k_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_);
lean_dec(v___y_2175_);
lean_dec_ref(v___y_2174_);
lean_dec(v___y_2173_);
lean_dec_ref(v___y_2172_);
lean_dec(v_u_2166_);
return v_res_2178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__0(lean_object* v_goal_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_){
_start:
{
lean_object* v___x_2185_; 
v___x_2185_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2185_) == 0)
{
lean_object* v_a_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; uint8_t v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; 
v_a_2186_ = lean_ctor_get(v___x_2185_, 0);
lean_inc_n(v_a_2186_, 2);
lean_dec_ref_known(v___x_2185_, 1);
v___x_2187_ = l_Lean_Expr_sort___override(v_a_2186_);
v___x_2188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2188_, 0, v___x_2187_);
v___x_2189_ = 0;
v___x_2190_ = lean_box(0);
v___x_2191_ = l_Lean_Meta_mkFreshExprMVar(v___x_2188_, v___x_2189_, v___x_2190_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2191_) == 0)
{
lean_object* v_a_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; 
v_a_2192_ = lean_ctor_get(v___x_2191_, 0);
lean_inc_n(v_a_2192_, 2);
lean_dec_ref_known(v___x_2191_, 1);
v___x_2193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2193_, 0, v_a_2192_);
lean_inc_ref(v___x_2193_);
v___x_2194_ = l_Lean_Meta_mkFreshExprMVar(v___x_2193_, v___x_2189_, v___x_2190_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2194_) == 0)
{
lean_object* v_a_2195_; lean_object* v___x_2196_; 
v_a_2195_ = lean_ctor_get(v___x_2194_, 0);
lean_inc(v_a_2195_);
lean_dec_ref_known(v___x_2194_, 1);
v___x_2196_ = l_Lean_Meta_mkFreshExprMVar(v___x_2193_, v___x_2189_, v___x_2190_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2196_) == 0)
{
lean_object* v_a_2197_; lean_object* v_keyedConfig_2198_; uint8_t v_trackZetaDelta_2199_; lean_object* v_zetaDeltaSet_2200_; lean_object* v_lctx_2201_; lean_object* v_localInstances_2202_; lean_object* v_defEqCtx_x3f_2203_; lean_object* v_synthPendingDepth_2204_; lean_object* v_customCanUnfoldPredicate_x3f_2205_; uint8_t v_univApprox_2206_; uint8_t v_inTypeClassResolution_2207_; uint8_t v_cacheInferType_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2297_; 
v_a_2197_ = lean_ctor_get(v___x_2196_, 0);
lean_inc(v_a_2197_);
lean_dec_ref_known(v___x_2196_, 1);
v_keyedConfig_2198_ = lean_ctor_get(v___y_2180_, 0);
v_trackZetaDelta_2199_ = lean_ctor_get_uint8(v___y_2180_, sizeof(void*)*7);
v_zetaDeltaSet_2200_ = lean_ctor_get(v___y_2180_, 1);
v_lctx_2201_ = lean_ctor_get(v___y_2180_, 2);
v_localInstances_2202_ = lean_ctor_get(v___y_2180_, 3);
v_defEqCtx_x3f_2203_ = lean_ctor_get(v___y_2180_, 4);
v_synthPendingDepth_2204_ = lean_ctor_get(v___y_2180_, 5);
v_customCanUnfoldPredicate_x3f_2205_ = lean_ctor_get(v___y_2180_, 6);
v_univApprox_2206_ = lean_ctor_get_uint8(v___y_2180_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2207_ = lean_ctor_get_uint8(v___y_2180_, sizeof(void*)*7 + 2);
v_cacheInferType_2208_ = lean_ctor_get_uint8(v___y_2180_, sizeof(void*)*7 + 3);
v_isSharedCheck_2297_ = !lean_is_exclusive(v___y_2180_);
if (v_isSharedCheck_2297_ == 0)
{
v___x_2210_ = v___y_2180_;
v_isShared_2211_ = v_isSharedCheck_2297_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2205_);
lean_inc(v_synthPendingDepth_2204_);
lean_inc(v_defEqCtx_x3f_2203_);
lean_inc(v_localInstances_2202_);
lean_inc(v_lctx_2201_);
lean_inc(v_zetaDeltaSet_2200_);
lean_inc(v_keyedConfig_2198_);
lean_dec(v___y_2180_);
v___x_2210_ = lean_box(0);
v_isShared_2211_ = v_isSharedCheck_2297_;
goto v_resetjp_2209_;
}
v_resetjp_2209_:
{
lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; uint8_t v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2222_; 
v___x_2212_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5));
v___x_2213_ = lean_box(0);
lean_inc(v_a_2186_);
v___x_2214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2214_, 0, v_a_2186_);
lean_ctor_set(v___x_2214_, 1, v___x_2213_);
v___x_2215_ = l_Lean_Expr_const___override(v___x_2212_, v___x_2214_);
lean_inc(v_a_2192_);
v___x_2216_ = l_Lean_Expr_app___override(v___x_2215_, v_a_2192_);
lean_inc(v_a_2195_);
v___x_2217_ = l_Lean_Expr_app___override(v___x_2216_, v_a_2195_);
lean_inc(v_a_2197_);
v___x_2218_ = l_Lean_Expr_app___override(v___x_2217_, v_a_2197_);
v___x_2219_ = 2;
v___x_2220_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2219_, v_keyedConfig_2198_);
if (v_isShared_2211_ == 0)
{
lean_ctor_set(v___x_2210_, 0, v___x_2220_);
v___x_2222_ = v___x_2210_;
goto v_reusejp_2221_;
}
else
{
lean_object* v_reuseFailAlloc_2296_; 
v_reuseFailAlloc_2296_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2296_, 0, v___x_2220_);
lean_ctor_set(v_reuseFailAlloc_2296_, 1, v_zetaDeltaSet_2200_);
lean_ctor_set(v_reuseFailAlloc_2296_, 2, v_lctx_2201_);
lean_ctor_set(v_reuseFailAlloc_2296_, 3, v_localInstances_2202_);
lean_ctor_set(v_reuseFailAlloc_2296_, 4, v_defEqCtx_x3f_2203_);
lean_ctor_set(v_reuseFailAlloc_2296_, 5, v_synthPendingDepth_2204_);
lean_ctor_set(v_reuseFailAlloc_2296_, 6, v_customCanUnfoldPredicate_x3f_2205_);
lean_ctor_set_uint8(v_reuseFailAlloc_2296_, sizeof(void*)*7, v_trackZetaDelta_2199_);
lean_ctor_set_uint8(v_reuseFailAlloc_2296_, sizeof(void*)*7 + 1, v_univApprox_2206_);
lean_ctor_set_uint8(v_reuseFailAlloc_2296_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2207_);
lean_ctor_set_uint8(v_reuseFailAlloc_2296_, sizeof(void*)*7 + 3, v_cacheInferType_2208_);
v___x_2222_ = v_reuseFailAlloc_2296_;
goto v_reusejp_2221_;
}
v_reusejp_2221_:
{
lean_object* v___x_2223_; 
v___x_2223_ = l_Lean_Meta_isExprDefEq(v___x_2218_, v_goal_2179_, v___x_2222_, v___y_2181_, v___y_2182_, v___y_2183_);
lean_dec_ref(v___x_2222_);
if (lean_obj_tag(v___x_2223_) == 0)
{
lean_object* v_a_2224_; lean_object* v___x_2226_; uint8_t v_isShared_2227_; uint8_t v_isSharedCheck_2287_; 
v_a_2224_ = lean_ctor_get(v___x_2223_, 0);
v_isSharedCheck_2287_ = !lean_is_exclusive(v___x_2223_);
if (v_isSharedCheck_2287_ == 0)
{
v___x_2226_ = v___x_2223_;
v_isShared_2227_ = v_isSharedCheck_2287_;
goto v_resetjp_2225_;
}
else
{
lean_inc(v_a_2224_);
lean_dec(v___x_2223_);
v___x_2226_ = lean_box(0);
v_isShared_2227_ = v_isSharedCheck_2287_;
goto v_resetjp_2225_;
}
v_resetjp_2225_:
{
uint8_t v___x_2228_; 
v___x_2228_ = lean_unbox(v_a_2224_);
if (v___x_2228_ == 0)
{
lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2234_; 
v___x_2229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2229_, 0, v_a_2197_);
lean_ctor_set(v___x_2229_, 1, v_a_2224_);
v___x_2230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2230_, 0, v_a_2195_);
lean_ctor_set(v___x_2230_, 1, v___x_2229_);
v___x_2231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2231_, 0, v_a_2192_);
lean_ctor_set(v___x_2231_, 1, v___x_2230_);
v___x_2232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2232_, 0, v_a_2186_);
lean_ctor_set(v___x_2232_, 1, v___x_2231_);
if (v_isShared_2227_ == 0)
{
lean_ctor_set(v___x_2226_, 0, v___x_2232_);
v___x_2234_ = v___x_2226_;
goto v_reusejp_2233_;
}
else
{
lean_object* v_reuseFailAlloc_2235_; 
v_reuseFailAlloc_2235_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2235_, 0, v___x_2232_);
v___x_2234_ = v_reuseFailAlloc_2235_;
goto v_reusejp_2233_;
}
v_reusejp_2233_:
{
return v___x_2234_;
}
}
else
{
lean_object* v___x_2236_; 
lean_del_object(v___x_2226_);
v___x_2236_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__0___redArg(v_a_2186_, v___y_2181_);
if (lean_obj_tag(v___x_2236_) == 0)
{
lean_object* v_a_2237_; lean_object* v___x_2238_; 
v_a_2237_ = lean_ctor_get(v___x_2236_, 0);
lean_inc(v_a_2237_);
lean_dec_ref_known(v___x_2236_, 1);
v___x_2238_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2192_, v___y_2181_);
if (lean_obj_tag(v___x_2238_) == 0)
{
lean_object* v_a_2239_; lean_object* v___x_2240_; 
v_a_2239_ = lean_ctor_get(v___x_2238_, 0);
lean_inc(v_a_2239_);
lean_dec_ref_known(v___x_2238_, 1);
v___x_2240_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2195_, v___y_2181_);
if (lean_obj_tag(v___x_2240_) == 0)
{
lean_object* v_a_2241_; lean_object* v___x_2242_; 
v_a_2241_ = lean_ctor_get(v___x_2240_, 0);
lean_inc(v_a_2241_);
lean_dec_ref_known(v___x_2240_, 1);
v___x_2242_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2197_, v___y_2181_);
if (lean_obj_tag(v___x_2242_) == 0)
{
lean_object* v_a_2243_; lean_object* v___x_2245_; uint8_t v_isShared_2246_; uint8_t v_isSharedCheck_2254_; 
v_a_2243_ = lean_ctor_get(v___x_2242_, 0);
v_isSharedCheck_2254_ = !lean_is_exclusive(v___x_2242_);
if (v_isSharedCheck_2254_ == 0)
{
v___x_2245_ = v___x_2242_;
v_isShared_2246_ = v_isSharedCheck_2254_;
goto v_resetjp_2244_;
}
else
{
lean_inc(v_a_2243_);
lean_dec(v___x_2242_);
v___x_2245_ = lean_box(0);
v_isShared_2246_ = v_isSharedCheck_2254_;
goto v_resetjp_2244_;
}
v_resetjp_2244_:
{
lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2252_; 
v___x_2247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2247_, 0, v_a_2243_);
lean_ctor_set(v___x_2247_, 1, v_a_2224_);
v___x_2248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2248_, 0, v_a_2241_);
lean_ctor_set(v___x_2248_, 1, v___x_2247_);
v___x_2249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2249_, 0, v_a_2239_);
lean_ctor_set(v___x_2249_, 1, v___x_2248_);
v___x_2250_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2250_, 0, v_a_2237_);
lean_ctor_set(v___x_2250_, 1, v___x_2249_);
if (v_isShared_2246_ == 0)
{
lean_ctor_set(v___x_2245_, 0, v___x_2250_);
v___x_2252_ = v___x_2245_;
goto v_reusejp_2251_;
}
else
{
lean_object* v_reuseFailAlloc_2253_; 
v_reuseFailAlloc_2253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2253_, 0, v___x_2250_);
v___x_2252_ = v_reuseFailAlloc_2253_;
goto v_reusejp_2251_;
}
v_reusejp_2251_:
{
return v___x_2252_;
}
}
}
else
{
lean_object* v_a_2255_; lean_object* v___x_2257_; uint8_t v_isShared_2258_; uint8_t v_isSharedCheck_2262_; 
lean_dec(v_a_2241_);
lean_dec(v_a_2239_);
lean_dec(v_a_2237_);
lean_dec(v_a_2224_);
v_a_2255_ = lean_ctor_get(v___x_2242_, 0);
v_isSharedCheck_2262_ = !lean_is_exclusive(v___x_2242_);
if (v_isSharedCheck_2262_ == 0)
{
v___x_2257_ = v___x_2242_;
v_isShared_2258_ = v_isSharedCheck_2262_;
goto v_resetjp_2256_;
}
else
{
lean_inc(v_a_2255_);
lean_dec(v___x_2242_);
v___x_2257_ = lean_box(0);
v_isShared_2258_ = v_isSharedCheck_2262_;
goto v_resetjp_2256_;
}
v_resetjp_2256_:
{
lean_object* v___x_2260_; 
if (v_isShared_2258_ == 0)
{
v___x_2260_ = v___x_2257_;
goto v_reusejp_2259_;
}
else
{
lean_object* v_reuseFailAlloc_2261_; 
v_reuseFailAlloc_2261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2261_, 0, v_a_2255_);
v___x_2260_ = v_reuseFailAlloc_2261_;
goto v_reusejp_2259_;
}
v_reusejp_2259_:
{
return v___x_2260_;
}
}
}
}
else
{
lean_object* v_a_2263_; lean_object* v___x_2265_; uint8_t v_isShared_2266_; uint8_t v_isSharedCheck_2270_; 
lean_dec(v_a_2239_);
lean_dec(v_a_2237_);
lean_dec(v_a_2224_);
lean_dec(v_a_2197_);
v_a_2263_ = lean_ctor_get(v___x_2240_, 0);
v_isSharedCheck_2270_ = !lean_is_exclusive(v___x_2240_);
if (v_isSharedCheck_2270_ == 0)
{
v___x_2265_ = v___x_2240_;
v_isShared_2266_ = v_isSharedCheck_2270_;
goto v_resetjp_2264_;
}
else
{
lean_inc(v_a_2263_);
lean_dec(v___x_2240_);
v___x_2265_ = lean_box(0);
v_isShared_2266_ = v_isSharedCheck_2270_;
goto v_resetjp_2264_;
}
v_resetjp_2264_:
{
lean_object* v___x_2268_; 
if (v_isShared_2266_ == 0)
{
v___x_2268_ = v___x_2265_;
goto v_reusejp_2267_;
}
else
{
lean_object* v_reuseFailAlloc_2269_; 
v_reuseFailAlloc_2269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2269_, 0, v_a_2263_);
v___x_2268_ = v_reuseFailAlloc_2269_;
goto v_reusejp_2267_;
}
v_reusejp_2267_:
{
return v___x_2268_;
}
}
}
}
else
{
lean_object* v_a_2271_; lean_object* v___x_2273_; uint8_t v_isShared_2274_; uint8_t v_isSharedCheck_2278_; 
lean_dec(v_a_2237_);
lean_dec(v_a_2224_);
lean_dec(v_a_2197_);
lean_dec(v_a_2195_);
v_a_2271_ = lean_ctor_get(v___x_2238_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v___x_2238_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2273_ = v___x_2238_;
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
else
{
lean_inc(v_a_2271_);
lean_dec(v___x_2238_);
v___x_2273_ = lean_box(0);
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
v_resetjp_2272_:
{
lean_object* v___x_2276_; 
if (v_isShared_2274_ == 0)
{
v___x_2276_ = v___x_2273_;
goto v_reusejp_2275_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v_a_2271_);
v___x_2276_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2275_;
}
v_reusejp_2275_:
{
return v___x_2276_;
}
}
}
}
else
{
lean_object* v_a_2279_; lean_object* v___x_2281_; uint8_t v_isShared_2282_; uint8_t v_isSharedCheck_2286_; 
lean_dec(v_a_2224_);
lean_dec(v_a_2197_);
lean_dec(v_a_2195_);
lean_dec(v_a_2192_);
v_a_2279_ = lean_ctor_get(v___x_2236_, 0);
v_isSharedCheck_2286_ = !lean_is_exclusive(v___x_2236_);
if (v_isSharedCheck_2286_ == 0)
{
v___x_2281_ = v___x_2236_;
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
else
{
lean_inc(v_a_2279_);
lean_dec(v___x_2236_);
v___x_2281_ = lean_box(0);
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
v_resetjp_2280_:
{
lean_object* v___x_2284_; 
if (v_isShared_2282_ == 0)
{
v___x_2284_ = v___x_2281_;
goto v_reusejp_2283_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v_a_2279_);
v___x_2284_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2283_;
}
v_reusejp_2283_:
{
return v___x_2284_;
}
}
}
}
}
}
else
{
lean_object* v_a_2288_; lean_object* v___x_2290_; uint8_t v_isShared_2291_; uint8_t v_isSharedCheck_2295_; 
lean_dec(v_a_2197_);
lean_dec(v_a_2195_);
lean_dec(v_a_2192_);
lean_dec(v_a_2186_);
v_a_2288_ = lean_ctor_get(v___x_2223_, 0);
v_isSharedCheck_2295_ = !lean_is_exclusive(v___x_2223_);
if (v_isSharedCheck_2295_ == 0)
{
v___x_2290_ = v___x_2223_;
v_isShared_2291_ = v_isSharedCheck_2295_;
goto v_resetjp_2289_;
}
else
{
lean_inc(v_a_2288_);
lean_dec(v___x_2223_);
v___x_2290_ = lean_box(0);
v_isShared_2291_ = v_isSharedCheck_2295_;
goto v_resetjp_2289_;
}
v_resetjp_2289_:
{
lean_object* v___x_2293_; 
if (v_isShared_2291_ == 0)
{
v___x_2293_ = v___x_2290_;
goto v_reusejp_2292_;
}
else
{
lean_object* v_reuseFailAlloc_2294_; 
v_reuseFailAlloc_2294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2294_, 0, v_a_2288_);
v___x_2293_ = v_reuseFailAlloc_2294_;
goto v_reusejp_2292_;
}
v_reusejp_2292_:
{
return v___x_2293_;
}
}
}
}
}
}
else
{
lean_object* v_a_2298_; lean_object* v___x_2300_; uint8_t v_isShared_2301_; uint8_t v_isSharedCheck_2305_; 
lean_dec(v_a_2195_);
lean_dec(v_a_2192_);
lean_dec(v_a_2186_);
lean_dec_ref(v___y_2180_);
lean_dec_ref(v_goal_2179_);
v_a_2298_ = lean_ctor_get(v___x_2196_, 0);
v_isSharedCheck_2305_ = !lean_is_exclusive(v___x_2196_);
if (v_isSharedCheck_2305_ == 0)
{
v___x_2300_ = v___x_2196_;
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
else
{
lean_inc(v_a_2298_);
lean_dec(v___x_2196_);
v___x_2300_ = lean_box(0);
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
v_resetjp_2299_:
{
lean_object* v___x_2303_; 
if (v_isShared_2301_ == 0)
{
v___x_2303_ = v___x_2300_;
goto v_reusejp_2302_;
}
else
{
lean_object* v_reuseFailAlloc_2304_; 
v_reuseFailAlloc_2304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2304_, 0, v_a_2298_);
v___x_2303_ = v_reuseFailAlloc_2304_;
goto v_reusejp_2302_;
}
v_reusejp_2302_:
{
return v___x_2303_;
}
}
}
}
else
{
lean_object* v_a_2306_; lean_object* v___x_2308_; uint8_t v_isShared_2309_; uint8_t v_isSharedCheck_2313_; 
lean_dec_ref_known(v___x_2193_, 1);
lean_dec(v_a_2192_);
lean_dec(v_a_2186_);
lean_dec_ref(v___y_2180_);
lean_dec_ref(v_goal_2179_);
v_a_2306_ = lean_ctor_get(v___x_2194_, 0);
v_isSharedCheck_2313_ = !lean_is_exclusive(v___x_2194_);
if (v_isSharedCheck_2313_ == 0)
{
v___x_2308_ = v___x_2194_;
v_isShared_2309_ = v_isSharedCheck_2313_;
goto v_resetjp_2307_;
}
else
{
lean_inc(v_a_2306_);
lean_dec(v___x_2194_);
v___x_2308_ = lean_box(0);
v_isShared_2309_ = v_isSharedCheck_2313_;
goto v_resetjp_2307_;
}
v_resetjp_2307_:
{
lean_object* v___x_2311_; 
if (v_isShared_2309_ == 0)
{
v___x_2311_ = v___x_2308_;
goto v_reusejp_2310_;
}
else
{
lean_object* v_reuseFailAlloc_2312_; 
v_reuseFailAlloc_2312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2312_, 0, v_a_2306_);
v___x_2311_ = v_reuseFailAlloc_2312_;
goto v_reusejp_2310_;
}
v_reusejp_2310_:
{
return v___x_2311_;
}
}
}
}
else
{
lean_object* v_a_2314_; lean_object* v___x_2316_; uint8_t v_isShared_2317_; uint8_t v_isSharedCheck_2321_; 
lean_dec(v_a_2186_);
lean_dec_ref(v___y_2180_);
lean_dec_ref(v_goal_2179_);
v_a_2314_ = lean_ctor_get(v___x_2191_, 0);
v_isSharedCheck_2321_ = !lean_is_exclusive(v___x_2191_);
if (v_isSharedCheck_2321_ == 0)
{
v___x_2316_ = v___x_2191_;
v_isShared_2317_ = v_isSharedCheck_2321_;
goto v_resetjp_2315_;
}
else
{
lean_inc(v_a_2314_);
lean_dec(v___x_2191_);
v___x_2316_ = lean_box(0);
v_isShared_2317_ = v_isSharedCheck_2321_;
goto v_resetjp_2315_;
}
v_resetjp_2315_:
{
lean_object* v___x_2319_; 
if (v_isShared_2317_ == 0)
{
v___x_2319_ = v___x_2316_;
goto v_reusejp_2318_;
}
else
{
lean_object* v_reuseFailAlloc_2320_; 
v_reuseFailAlloc_2320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2320_, 0, v_a_2314_);
v___x_2319_ = v_reuseFailAlloc_2320_;
goto v_reusejp_2318_;
}
v_reusejp_2318_:
{
return v___x_2319_;
}
}
}
}
else
{
lean_object* v_a_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2329_; 
lean_dec_ref(v___y_2180_);
lean_dec_ref(v_goal_2179_);
v_a_2322_ = lean_ctor_get(v___x_2185_, 0);
v_isSharedCheck_2329_ = !lean_is_exclusive(v___x_2185_);
if (v_isSharedCheck_2329_ == 0)
{
v___x_2324_ = v___x_2185_;
v_isShared_2325_ = v_isSharedCheck_2329_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_a_2322_);
lean_dec(v___x_2185_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2329_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
lean_object* v___x_2327_; 
if (v_isShared_2325_ == 0)
{
v___x_2327_ = v___x_2324_;
goto v_reusejp_2326_;
}
else
{
lean_object* v_reuseFailAlloc_2328_; 
v_reuseFailAlloc_2328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2328_, 0, v_a_2322_);
v___x_2327_ = v_reuseFailAlloc_2328_;
goto v_reusejp_2326_;
}
v_reusejp_2326_:
{
return v___x_2327_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__0___boxed(lean_object* v_goal_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_){
_start:
{
lean_object* v_res_2336_; 
v_res_2336_ = lp_mathlib_ExistsAndEq_construct___lam__0(v_goal_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
lean_dec(v___y_2334_);
lean_dec_ref(v___y_2333_);
lean_dec(v___y_2332_);
return v_res_2336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__1(lean_object* v___x_2337_, uint8_t v___x_2338_, lean_object* v___x_2339_, lean_object* v_goal_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_){
_start:
{
lean_object* v___x_2346_; 
lean_inc(v___x_2339_);
lean_inc(v___x_2337_);
v___x_2346_ = l_Lean_Meta_mkFreshExprMVar(v___x_2337_, v___x_2338_, v___x_2339_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_);
if (lean_obj_tag(v___x_2346_) == 0)
{
lean_object* v_a_2347_; lean_object* v___x_2348_; 
v_a_2347_ = lean_ctor_get(v___x_2346_, 0);
lean_inc(v_a_2347_);
lean_dec_ref_known(v___x_2346_, 1);
v___x_2348_ = l_Lean_Meta_mkFreshExprMVar(v___x_2337_, v___x_2338_, v___x_2339_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_);
if (lean_obj_tag(v___x_2348_) == 0)
{
lean_object* v_a_2349_; lean_object* v_keyedConfig_2350_; uint8_t v_trackZetaDelta_2351_; lean_object* v_zetaDeltaSet_2352_; lean_object* v_lctx_2353_; lean_object* v_localInstances_2354_; lean_object* v_defEqCtx_x3f_2355_; lean_object* v_synthPendingDepth_2356_; lean_object* v_customCanUnfoldPredicate_x3f_2357_; uint8_t v_univApprox_2358_; uint8_t v_inTypeClassResolution_2359_; uint8_t v_cacheInferType_2360_; lean_object* v___x_2362_; uint8_t v_isShared_2363_; uint8_t v_isSharedCheck_2421_; 
v_a_2349_ = lean_ctor_get(v___x_2348_, 0);
lean_inc(v_a_2349_);
lean_dec_ref_known(v___x_2348_, 1);
v_keyedConfig_2350_ = lean_ctor_get(v___y_2341_, 0);
v_trackZetaDelta_2351_ = lean_ctor_get_uint8(v___y_2341_, sizeof(void*)*7);
v_zetaDeltaSet_2352_ = lean_ctor_get(v___y_2341_, 1);
v_lctx_2353_ = lean_ctor_get(v___y_2341_, 2);
v_localInstances_2354_ = lean_ctor_get(v___y_2341_, 3);
v_defEqCtx_x3f_2355_ = lean_ctor_get(v___y_2341_, 4);
v_synthPendingDepth_2356_ = lean_ctor_get(v___y_2341_, 5);
v_customCanUnfoldPredicate_x3f_2357_ = lean_ctor_get(v___y_2341_, 6);
v_univApprox_2358_ = lean_ctor_get_uint8(v___y_2341_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2359_ = lean_ctor_get_uint8(v___y_2341_, sizeof(void*)*7 + 2);
v_cacheInferType_2360_ = lean_ctor_get_uint8(v___y_2341_, sizeof(void*)*7 + 3);
v_isSharedCheck_2421_ = !lean_is_exclusive(v___y_2341_);
if (v_isSharedCheck_2421_ == 0)
{
v___x_2362_ = v___y_2341_;
v_isShared_2363_ = v_isSharedCheck_2421_;
goto v_resetjp_2361_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2357_);
lean_inc(v_synthPendingDepth_2356_);
lean_inc(v_defEqCtx_x3f_2355_);
lean_inc(v_localInstances_2354_);
lean_inc(v_lctx_2353_);
lean_inc(v_zetaDeltaSet_2352_);
lean_inc(v_keyedConfig_2350_);
lean_dec(v___y_2341_);
v___x_2362_ = lean_box(0);
v_isShared_2363_ = v_isSharedCheck_2421_;
goto v_resetjp_2361_;
}
v_resetjp_2361_:
{
lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; uint8_t v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2370_; 
v___x_2364_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__1___closed__0);
lean_inc(v_a_2347_);
v___x_2365_ = l_Lean_Expr_app___override(v___x_2364_, v_a_2347_);
lean_inc(v_a_2349_);
v___x_2366_ = l_Lean_Expr_app___override(v___x_2365_, v_a_2349_);
v___x_2367_ = 2;
v___x_2368_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2367_, v_keyedConfig_2350_);
if (v_isShared_2363_ == 0)
{
lean_ctor_set(v___x_2362_, 0, v___x_2368_);
v___x_2370_ = v___x_2362_;
goto v_reusejp_2369_;
}
else
{
lean_object* v_reuseFailAlloc_2420_; 
v_reuseFailAlloc_2420_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2420_, 0, v___x_2368_);
lean_ctor_set(v_reuseFailAlloc_2420_, 1, v_zetaDeltaSet_2352_);
lean_ctor_set(v_reuseFailAlloc_2420_, 2, v_lctx_2353_);
lean_ctor_set(v_reuseFailAlloc_2420_, 3, v_localInstances_2354_);
lean_ctor_set(v_reuseFailAlloc_2420_, 4, v_defEqCtx_x3f_2355_);
lean_ctor_set(v_reuseFailAlloc_2420_, 5, v_synthPendingDepth_2356_);
lean_ctor_set(v_reuseFailAlloc_2420_, 6, v_customCanUnfoldPredicate_x3f_2357_);
lean_ctor_set_uint8(v_reuseFailAlloc_2420_, sizeof(void*)*7, v_trackZetaDelta_2351_);
lean_ctor_set_uint8(v_reuseFailAlloc_2420_, sizeof(void*)*7 + 1, v_univApprox_2358_);
lean_ctor_set_uint8(v_reuseFailAlloc_2420_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2359_);
lean_ctor_set_uint8(v_reuseFailAlloc_2420_, sizeof(void*)*7 + 3, v_cacheInferType_2360_);
v___x_2370_ = v_reuseFailAlloc_2420_;
goto v_reusejp_2369_;
}
v_reusejp_2369_:
{
lean_object* v___x_2371_; 
v___x_2371_ = l_Lean_Meta_isExprDefEq(v___x_2366_, v_goal_2340_, v___x_2370_, v___y_2342_, v___y_2343_, v___y_2344_);
lean_dec_ref(v___x_2370_);
if (lean_obj_tag(v___x_2371_) == 0)
{
lean_object* v_a_2372_; lean_object* v___x_2374_; uint8_t v_isShared_2375_; uint8_t v_isSharedCheck_2411_; 
v_a_2372_ = lean_ctor_get(v___x_2371_, 0);
v_isSharedCheck_2411_ = !lean_is_exclusive(v___x_2371_);
if (v_isSharedCheck_2411_ == 0)
{
v___x_2374_ = v___x_2371_;
v_isShared_2375_ = v_isSharedCheck_2411_;
goto v_resetjp_2373_;
}
else
{
lean_inc(v_a_2372_);
lean_dec(v___x_2371_);
v___x_2374_ = lean_box(0);
v_isShared_2375_ = v_isSharedCheck_2411_;
goto v_resetjp_2373_;
}
v_resetjp_2373_:
{
uint8_t v___x_2376_; 
v___x_2376_ = lean_unbox(v_a_2372_);
if (v___x_2376_ == 0)
{
lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2380_; 
v___x_2377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2377_, 0, v_a_2349_);
lean_ctor_set(v___x_2377_, 1, v_a_2372_);
v___x_2378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2378_, 0, v_a_2347_);
lean_ctor_set(v___x_2378_, 1, v___x_2377_);
if (v_isShared_2375_ == 0)
{
lean_ctor_set(v___x_2374_, 0, v___x_2378_);
v___x_2380_ = v___x_2374_;
goto v_reusejp_2379_;
}
else
{
lean_object* v_reuseFailAlloc_2381_; 
v_reuseFailAlloc_2381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2381_, 0, v___x_2378_);
v___x_2380_ = v_reuseFailAlloc_2381_;
goto v_reusejp_2379_;
}
v_reusejp_2379_:
{
return v___x_2380_;
}
}
else
{
lean_object* v___x_2382_; 
lean_del_object(v___x_2374_);
v___x_2382_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2347_, v___y_2342_);
if (lean_obj_tag(v___x_2382_) == 0)
{
lean_object* v_a_2383_; lean_object* v___x_2384_; 
v_a_2383_ = lean_ctor_get(v___x_2382_, 0);
lean_inc(v_a_2383_);
lean_dec_ref_known(v___x_2382_, 1);
v___x_2384_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2349_, v___y_2342_);
if (lean_obj_tag(v___x_2384_) == 0)
{
lean_object* v_a_2385_; lean_object* v___x_2387_; uint8_t v_isShared_2388_; uint8_t v_isSharedCheck_2394_; 
v_a_2385_ = lean_ctor_get(v___x_2384_, 0);
v_isSharedCheck_2394_ = !lean_is_exclusive(v___x_2384_);
if (v_isSharedCheck_2394_ == 0)
{
v___x_2387_ = v___x_2384_;
v_isShared_2388_ = v_isSharedCheck_2394_;
goto v_resetjp_2386_;
}
else
{
lean_inc(v_a_2385_);
lean_dec(v___x_2384_);
v___x_2387_ = lean_box(0);
v_isShared_2388_ = v_isSharedCheck_2394_;
goto v_resetjp_2386_;
}
v_resetjp_2386_:
{
lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2392_; 
v___x_2389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2389_, 0, v_a_2385_);
lean_ctor_set(v___x_2389_, 1, v_a_2372_);
v___x_2390_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2390_, 0, v_a_2383_);
lean_ctor_set(v___x_2390_, 1, v___x_2389_);
if (v_isShared_2388_ == 0)
{
lean_ctor_set(v___x_2387_, 0, v___x_2390_);
v___x_2392_ = v___x_2387_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2393_; 
v_reuseFailAlloc_2393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2393_, 0, v___x_2390_);
v___x_2392_ = v_reuseFailAlloc_2393_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
return v___x_2392_;
}
}
}
else
{
lean_object* v_a_2395_; lean_object* v___x_2397_; uint8_t v_isShared_2398_; uint8_t v_isSharedCheck_2402_; 
lean_dec(v_a_2383_);
lean_dec(v_a_2372_);
v_a_2395_ = lean_ctor_get(v___x_2384_, 0);
v_isSharedCheck_2402_ = !lean_is_exclusive(v___x_2384_);
if (v_isSharedCheck_2402_ == 0)
{
v___x_2397_ = v___x_2384_;
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
else
{
lean_inc(v_a_2395_);
lean_dec(v___x_2384_);
v___x_2397_ = lean_box(0);
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
v_resetjp_2396_:
{
lean_object* v___x_2400_; 
if (v_isShared_2398_ == 0)
{
v___x_2400_ = v___x_2397_;
goto v_reusejp_2399_;
}
else
{
lean_object* v_reuseFailAlloc_2401_; 
v_reuseFailAlloc_2401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2401_, 0, v_a_2395_);
v___x_2400_ = v_reuseFailAlloc_2401_;
goto v_reusejp_2399_;
}
v_reusejp_2399_:
{
return v___x_2400_;
}
}
}
}
else
{
lean_object* v_a_2403_; lean_object* v___x_2405_; uint8_t v_isShared_2406_; uint8_t v_isSharedCheck_2410_; 
lean_dec(v_a_2372_);
lean_dec(v_a_2349_);
v_a_2403_ = lean_ctor_get(v___x_2382_, 0);
v_isSharedCheck_2410_ = !lean_is_exclusive(v___x_2382_);
if (v_isSharedCheck_2410_ == 0)
{
v___x_2405_ = v___x_2382_;
v_isShared_2406_ = v_isSharedCheck_2410_;
goto v_resetjp_2404_;
}
else
{
lean_inc(v_a_2403_);
lean_dec(v___x_2382_);
v___x_2405_ = lean_box(0);
v_isShared_2406_ = v_isSharedCheck_2410_;
goto v_resetjp_2404_;
}
v_resetjp_2404_:
{
lean_object* v___x_2408_; 
if (v_isShared_2406_ == 0)
{
v___x_2408_ = v___x_2405_;
goto v_reusejp_2407_;
}
else
{
lean_object* v_reuseFailAlloc_2409_; 
v_reuseFailAlloc_2409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2409_, 0, v_a_2403_);
v___x_2408_ = v_reuseFailAlloc_2409_;
goto v_reusejp_2407_;
}
v_reusejp_2407_:
{
return v___x_2408_;
}
}
}
}
}
}
else
{
lean_object* v_a_2412_; lean_object* v___x_2414_; uint8_t v_isShared_2415_; uint8_t v_isSharedCheck_2419_; 
lean_dec(v_a_2349_);
lean_dec(v_a_2347_);
v_a_2412_ = lean_ctor_get(v___x_2371_, 0);
v_isSharedCheck_2419_ = !lean_is_exclusive(v___x_2371_);
if (v_isSharedCheck_2419_ == 0)
{
v___x_2414_ = v___x_2371_;
v_isShared_2415_ = v_isSharedCheck_2419_;
goto v_resetjp_2413_;
}
else
{
lean_inc(v_a_2412_);
lean_dec(v___x_2371_);
v___x_2414_ = lean_box(0);
v_isShared_2415_ = v_isSharedCheck_2419_;
goto v_resetjp_2413_;
}
v_resetjp_2413_:
{
lean_object* v___x_2417_; 
if (v_isShared_2415_ == 0)
{
v___x_2417_ = v___x_2414_;
goto v_reusejp_2416_;
}
else
{
lean_object* v_reuseFailAlloc_2418_; 
v_reuseFailAlloc_2418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2418_, 0, v_a_2412_);
v___x_2417_ = v_reuseFailAlloc_2418_;
goto v_reusejp_2416_;
}
v_reusejp_2416_:
{
return v___x_2417_;
}
}
}
}
}
}
else
{
lean_object* v_a_2422_; lean_object* v___x_2424_; uint8_t v_isShared_2425_; uint8_t v_isSharedCheck_2429_; 
lean_dec(v_a_2347_);
lean_dec_ref(v___y_2341_);
lean_dec_ref(v_goal_2340_);
v_a_2422_ = lean_ctor_get(v___x_2348_, 0);
v_isSharedCheck_2429_ = !lean_is_exclusive(v___x_2348_);
if (v_isSharedCheck_2429_ == 0)
{
v___x_2424_ = v___x_2348_;
v_isShared_2425_ = v_isSharedCheck_2429_;
goto v_resetjp_2423_;
}
else
{
lean_inc(v_a_2422_);
lean_dec(v___x_2348_);
v___x_2424_ = lean_box(0);
v_isShared_2425_ = v_isSharedCheck_2429_;
goto v_resetjp_2423_;
}
v_resetjp_2423_:
{
lean_object* v___x_2427_; 
if (v_isShared_2425_ == 0)
{
v___x_2427_ = v___x_2424_;
goto v_reusejp_2426_;
}
else
{
lean_object* v_reuseFailAlloc_2428_; 
v_reuseFailAlloc_2428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2428_, 0, v_a_2422_);
v___x_2427_ = v_reuseFailAlloc_2428_;
goto v_reusejp_2426_;
}
v_reusejp_2426_:
{
return v___x_2427_;
}
}
}
}
else
{
lean_object* v_a_2430_; lean_object* v___x_2432_; uint8_t v_isShared_2433_; uint8_t v_isSharedCheck_2437_; 
lean_dec_ref(v___y_2341_);
lean_dec_ref(v_goal_2340_);
lean_dec(v___x_2339_);
lean_dec(v___x_2337_);
v_a_2430_ = lean_ctor_get(v___x_2346_, 0);
v_isSharedCheck_2437_ = !lean_is_exclusive(v___x_2346_);
if (v_isSharedCheck_2437_ == 0)
{
v___x_2432_ = v___x_2346_;
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
else
{
lean_inc(v_a_2430_);
lean_dec(v___x_2346_);
v___x_2432_ = lean_box(0);
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
v_resetjp_2431_:
{
lean_object* v___x_2435_; 
if (v_isShared_2433_ == 0)
{
v___x_2435_ = v___x_2432_;
goto v_reusejp_2434_;
}
else
{
lean_object* v_reuseFailAlloc_2436_; 
v_reuseFailAlloc_2436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2436_, 0, v_a_2430_);
v___x_2435_ = v_reuseFailAlloc_2436_;
goto v_reusejp_2434_;
}
v_reusejp_2434_:
{
return v___x_2435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__1___boxed(lean_object* v___x_2438_, lean_object* v___x_2439_, lean_object* v___x_2440_, lean_object* v_goal_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_, lean_object* v___y_2446_){
_start:
{
uint8_t v___x_8269__boxed_2447_; lean_object* v_res_2448_; 
v___x_8269__boxed_2447_ = lean_unbox(v___x_2439_);
v_res_2448_ = lp_mathlib_ExistsAndEq_construct___lam__1(v___x_2438_, v___x_8269__boxed_2447_, v___x_2440_, v_goal_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
lean_dec(v___y_2445_);
lean_dec_ref(v___y_2444_);
lean_dec(v___y_2443_);
return v_res_2448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__3(lean_object* v___x_2449_, uint8_t v___x_2450_, lean_object* v___x_2451_, lean_object* v_fst_2452_, lean_object* v_goal_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_){
_start:
{
lean_object* v___x_2459_; 
lean_inc(v___x_2451_);
v___x_2459_ = l_Lean_Meta_mkFreshExprMVar(v___x_2449_, v___x_2450_, v___x_2451_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2459_) == 0)
{
lean_object* v_a_2460_; lean_object* v___x_2461_; uint8_t v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; 
v_a_2460_ = lean_ctor_get(v___x_2459_, 0);
lean_inc_n(v_a_2460_, 2);
lean_dec_ref_known(v___x_2459_, 1);
v___x_2461_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__3___closed__0);
v___x_2462_ = 0;
lean_inc(v___x_2451_);
v___x_2463_ = l_Lean_Expr_forallE___override(v___x_2451_, v_a_2460_, v___x_2461_, v___x_2462_);
v___x_2464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2464_, 0, v___x_2463_);
v___x_2465_ = l_Lean_Meta_mkFreshExprMVar(v___x_2464_, v___x_2450_, v___x_2451_, v___y_2454_, v___y_2455_, v___y_2456_, v___y_2457_);
if (lean_obj_tag(v___x_2465_) == 0)
{
lean_object* v_a_2466_; lean_object* v_keyedConfig_2467_; uint8_t v_trackZetaDelta_2468_; lean_object* v_zetaDeltaSet_2469_; lean_object* v_lctx_2470_; lean_object* v_localInstances_2471_; lean_object* v_defEqCtx_x3f_2472_; lean_object* v_synthPendingDepth_2473_; lean_object* v_customCanUnfoldPredicate_x3f_2474_; uint8_t v_univApprox_2475_; uint8_t v_inTypeClassResolution_2476_; uint8_t v_cacheInferType_2477_; lean_object* v___x_2479_; uint8_t v_isShared_2480_; uint8_t v_isSharedCheck_2541_; 
v_a_2466_ = lean_ctor_get(v___x_2465_, 0);
lean_inc(v_a_2466_);
lean_dec_ref_known(v___x_2465_, 1);
v_keyedConfig_2467_ = lean_ctor_get(v___y_2454_, 0);
v_trackZetaDelta_2468_ = lean_ctor_get_uint8(v___y_2454_, sizeof(void*)*7);
v_zetaDeltaSet_2469_ = lean_ctor_get(v___y_2454_, 1);
v_lctx_2470_ = lean_ctor_get(v___y_2454_, 2);
v_localInstances_2471_ = lean_ctor_get(v___y_2454_, 3);
v_defEqCtx_x3f_2472_ = lean_ctor_get(v___y_2454_, 4);
v_synthPendingDepth_2473_ = lean_ctor_get(v___y_2454_, 5);
v_customCanUnfoldPredicate_x3f_2474_ = lean_ctor_get(v___y_2454_, 6);
v_univApprox_2475_ = lean_ctor_get_uint8(v___y_2454_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2476_ = lean_ctor_get_uint8(v___y_2454_, sizeof(void*)*7 + 2);
v_cacheInferType_2477_ = lean_ctor_get_uint8(v___y_2454_, sizeof(void*)*7 + 3);
v_isSharedCheck_2541_ = !lean_is_exclusive(v___y_2454_);
if (v_isSharedCheck_2541_ == 0)
{
v___x_2479_ = v___y_2454_;
v_isShared_2480_ = v_isSharedCheck_2541_;
goto v_resetjp_2478_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2474_);
lean_inc(v_synthPendingDepth_2473_);
lean_inc(v_defEqCtx_x3f_2472_);
lean_inc(v_localInstances_2471_);
lean_inc(v_lctx_2470_);
lean_inc(v_zetaDeltaSet_2469_);
lean_inc(v_keyedConfig_2467_);
lean_dec(v___y_2454_);
v___x_2479_ = lean_box(0);
v_isShared_2480_ = v_isSharedCheck_2541_;
goto v_resetjp_2478_;
}
v_resetjp_2478_:
{
lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; uint8_t v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2490_; 
v___x_2481_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_2482_ = lean_box(0);
v___x_2483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2483_, 0, v_fst_2452_);
lean_ctor_set(v___x_2483_, 1, v___x_2482_);
v___x_2484_ = l_Lean_Expr_const___override(v___x_2481_, v___x_2483_);
lean_inc(v_a_2460_);
v___x_2485_ = l_Lean_Expr_app___override(v___x_2484_, v_a_2460_);
lean_inc(v_a_2466_);
v___x_2486_ = l_Lean_Expr_app___override(v___x_2485_, v_a_2466_);
v___x_2487_ = 2;
v___x_2488_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2487_, v_keyedConfig_2467_);
if (v_isShared_2480_ == 0)
{
lean_ctor_set(v___x_2479_, 0, v___x_2488_);
v___x_2490_ = v___x_2479_;
goto v_reusejp_2489_;
}
else
{
lean_object* v_reuseFailAlloc_2540_; 
v_reuseFailAlloc_2540_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2540_, 0, v___x_2488_);
lean_ctor_set(v_reuseFailAlloc_2540_, 1, v_zetaDeltaSet_2469_);
lean_ctor_set(v_reuseFailAlloc_2540_, 2, v_lctx_2470_);
lean_ctor_set(v_reuseFailAlloc_2540_, 3, v_localInstances_2471_);
lean_ctor_set(v_reuseFailAlloc_2540_, 4, v_defEqCtx_x3f_2472_);
lean_ctor_set(v_reuseFailAlloc_2540_, 5, v_synthPendingDepth_2473_);
lean_ctor_set(v_reuseFailAlloc_2540_, 6, v_customCanUnfoldPredicate_x3f_2474_);
lean_ctor_set_uint8(v_reuseFailAlloc_2540_, sizeof(void*)*7, v_trackZetaDelta_2468_);
lean_ctor_set_uint8(v_reuseFailAlloc_2540_, sizeof(void*)*7 + 1, v_univApprox_2475_);
lean_ctor_set_uint8(v_reuseFailAlloc_2540_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2476_);
lean_ctor_set_uint8(v_reuseFailAlloc_2540_, sizeof(void*)*7 + 3, v_cacheInferType_2477_);
v___x_2490_ = v_reuseFailAlloc_2540_;
goto v_reusejp_2489_;
}
v_reusejp_2489_:
{
lean_object* v___x_2491_; 
v___x_2491_ = l_Lean_Meta_isExprDefEq(v___x_2486_, v_goal_2453_, v___x_2490_, v___y_2455_, v___y_2456_, v___y_2457_);
lean_dec_ref(v___x_2490_);
if (lean_obj_tag(v___x_2491_) == 0)
{
lean_object* v_a_2492_; lean_object* v___x_2494_; uint8_t v_isShared_2495_; uint8_t v_isSharedCheck_2531_; 
v_a_2492_ = lean_ctor_get(v___x_2491_, 0);
v_isSharedCheck_2531_ = !lean_is_exclusive(v___x_2491_);
if (v_isSharedCheck_2531_ == 0)
{
v___x_2494_ = v___x_2491_;
v_isShared_2495_ = v_isSharedCheck_2531_;
goto v_resetjp_2493_;
}
else
{
lean_inc(v_a_2492_);
lean_dec(v___x_2491_);
v___x_2494_ = lean_box(0);
v_isShared_2495_ = v_isSharedCheck_2531_;
goto v_resetjp_2493_;
}
v_resetjp_2493_:
{
uint8_t v___x_2496_; 
v___x_2496_ = lean_unbox(v_a_2492_);
if (v___x_2496_ == 0)
{
lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___x_2500_; 
v___x_2497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2497_, 0, v_a_2466_);
lean_ctor_set(v___x_2497_, 1, v_a_2492_);
v___x_2498_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2498_, 0, v_a_2460_);
lean_ctor_set(v___x_2498_, 1, v___x_2497_);
if (v_isShared_2495_ == 0)
{
lean_ctor_set(v___x_2494_, 0, v___x_2498_);
v___x_2500_ = v___x_2494_;
goto v_reusejp_2499_;
}
else
{
lean_object* v_reuseFailAlloc_2501_; 
v_reuseFailAlloc_2501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2501_, 0, v___x_2498_);
v___x_2500_ = v_reuseFailAlloc_2501_;
goto v_reusejp_2499_;
}
v_reusejp_2499_:
{
return v___x_2500_;
}
}
else
{
lean_object* v___x_2502_; 
lean_del_object(v___x_2494_);
v___x_2502_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2460_, v___y_2455_);
if (lean_obj_tag(v___x_2502_) == 0)
{
lean_object* v_a_2503_; lean_object* v___x_2504_; 
v_a_2503_ = lean_ctor_get(v___x_2502_, 0);
lean_inc(v_a_2503_);
lean_dec_ref_known(v___x_2502_, 1);
v___x_2504_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2466_, v___y_2455_);
if (lean_obj_tag(v___x_2504_) == 0)
{
lean_object* v_a_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2514_; 
v_a_2505_ = lean_ctor_get(v___x_2504_, 0);
v_isSharedCheck_2514_ = !lean_is_exclusive(v___x_2504_);
if (v_isSharedCheck_2514_ == 0)
{
v___x_2507_ = v___x_2504_;
v_isShared_2508_ = v_isSharedCheck_2514_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_a_2505_);
lean_dec(v___x_2504_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2514_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2512_; 
v___x_2509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2509_, 0, v_a_2505_);
lean_ctor_set(v___x_2509_, 1, v_a_2492_);
v___x_2510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2510_, 0, v_a_2503_);
lean_ctor_set(v___x_2510_, 1, v___x_2509_);
if (v_isShared_2508_ == 0)
{
lean_ctor_set(v___x_2507_, 0, v___x_2510_);
v___x_2512_ = v___x_2507_;
goto v_reusejp_2511_;
}
else
{
lean_object* v_reuseFailAlloc_2513_; 
v_reuseFailAlloc_2513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2513_, 0, v___x_2510_);
v___x_2512_ = v_reuseFailAlloc_2513_;
goto v_reusejp_2511_;
}
v_reusejp_2511_:
{
return v___x_2512_;
}
}
}
else
{
lean_object* v_a_2515_; lean_object* v___x_2517_; uint8_t v_isShared_2518_; uint8_t v_isSharedCheck_2522_; 
lean_dec(v_a_2503_);
lean_dec(v_a_2492_);
v_a_2515_ = lean_ctor_get(v___x_2504_, 0);
v_isSharedCheck_2522_ = !lean_is_exclusive(v___x_2504_);
if (v_isSharedCheck_2522_ == 0)
{
v___x_2517_ = v___x_2504_;
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
else
{
lean_inc(v_a_2515_);
lean_dec(v___x_2504_);
v___x_2517_ = lean_box(0);
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
v_resetjp_2516_:
{
lean_object* v___x_2520_; 
if (v_isShared_2518_ == 0)
{
v___x_2520_ = v___x_2517_;
goto v_reusejp_2519_;
}
else
{
lean_object* v_reuseFailAlloc_2521_; 
v_reuseFailAlloc_2521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2521_, 0, v_a_2515_);
v___x_2520_ = v_reuseFailAlloc_2521_;
goto v_reusejp_2519_;
}
v_reusejp_2519_:
{
return v___x_2520_;
}
}
}
}
else
{
lean_object* v_a_2523_; lean_object* v___x_2525_; uint8_t v_isShared_2526_; uint8_t v_isSharedCheck_2530_; 
lean_dec(v_a_2492_);
lean_dec(v_a_2466_);
v_a_2523_ = lean_ctor_get(v___x_2502_, 0);
v_isSharedCheck_2530_ = !lean_is_exclusive(v___x_2502_);
if (v_isSharedCheck_2530_ == 0)
{
v___x_2525_ = v___x_2502_;
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
else
{
lean_inc(v_a_2523_);
lean_dec(v___x_2502_);
v___x_2525_ = lean_box(0);
v_isShared_2526_ = v_isSharedCheck_2530_;
goto v_resetjp_2524_;
}
v_resetjp_2524_:
{
lean_object* v___x_2528_; 
if (v_isShared_2526_ == 0)
{
v___x_2528_ = v___x_2525_;
goto v_reusejp_2527_;
}
else
{
lean_object* v_reuseFailAlloc_2529_; 
v_reuseFailAlloc_2529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2529_, 0, v_a_2523_);
v___x_2528_ = v_reuseFailAlloc_2529_;
goto v_reusejp_2527_;
}
v_reusejp_2527_:
{
return v___x_2528_;
}
}
}
}
}
}
else
{
lean_object* v_a_2532_; lean_object* v___x_2534_; uint8_t v_isShared_2535_; uint8_t v_isSharedCheck_2539_; 
lean_dec(v_a_2466_);
lean_dec(v_a_2460_);
v_a_2532_ = lean_ctor_get(v___x_2491_, 0);
v_isSharedCheck_2539_ = !lean_is_exclusive(v___x_2491_);
if (v_isSharedCheck_2539_ == 0)
{
v___x_2534_ = v___x_2491_;
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
else
{
lean_inc(v_a_2532_);
lean_dec(v___x_2491_);
v___x_2534_ = lean_box(0);
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
v_resetjp_2533_:
{
lean_object* v___x_2537_; 
if (v_isShared_2535_ == 0)
{
v___x_2537_ = v___x_2534_;
goto v_reusejp_2536_;
}
else
{
lean_object* v_reuseFailAlloc_2538_; 
v_reuseFailAlloc_2538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2538_, 0, v_a_2532_);
v___x_2537_ = v_reuseFailAlloc_2538_;
goto v_reusejp_2536_;
}
v_reusejp_2536_:
{
return v___x_2537_;
}
}
}
}
}
}
else
{
lean_object* v_a_2542_; lean_object* v___x_2544_; uint8_t v_isShared_2545_; uint8_t v_isSharedCheck_2549_; 
lean_dec(v_a_2460_);
lean_dec_ref(v___y_2454_);
lean_dec_ref(v_goal_2453_);
lean_dec(v_fst_2452_);
v_a_2542_ = lean_ctor_get(v___x_2465_, 0);
v_isSharedCheck_2549_ = !lean_is_exclusive(v___x_2465_);
if (v_isSharedCheck_2549_ == 0)
{
v___x_2544_ = v___x_2465_;
v_isShared_2545_ = v_isSharedCheck_2549_;
goto v_resetjp_2543_;
}
else
{
lean_inc(v_a_2542_);
lean_dec(v___x_2465_);
v___x_2544_ = lean_box(0);
v_isShared_2545_ = v_isSharedCheck_2549_;
goto v_resetjp_2543_;
}
v_resetjp_2543_:
{
lean_object* v___x_2547_; 
if (v_isShared_2545_ == 0)
{
v___x_2547_ = v___x_2544_;
goto v_reusejp_2546_;
}
else
{
lean_object* v_reuseFailAlloc_2548_; 
v_reuseFailAlloc_2548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2548_, 0, v_a_2542_);
v___x_2547_ = v_reuseFailAlloc_2548_;
goto v_reusejp_2546_;
}
v_reusejp_2546_:
{
return v___x_2547_;
}
}
}
}
else
{
lean_object* v_a_2550_; lean_object* v___x_2552_; uint8_t v_isShared_2553_; uint8_t v_isSharedCheck_2557_; 
lean_dec_ref(v___y_2454_);
lean_dec_ref(v_goal_2453_);
lean_dec(v_fst_2452_);
lean_dec(v___x_2451_);
v_a_2550_ = lean_ctor_get(v___x_2459_, 0);
v_isSharedCheck_2557_ = !lean_is_exclusive(v___x_2459_);
if (v_isSharedCheck_2557_ == 0)
{
v___x_2552_ = v___x_2459_;
v_isShared_2553_ = v_isSharedCheck_2557_;
goto v_resetjp_2551_;
}
else
{
lean_inc(v_a_2550_);
lean_dec(v___x_2459_);
v___x_2552_ = lean_box(0);
v_isShared_2553_ = v_isSharedCheck_2557_;
goto v_resetjp_2551_;
}
v_resetjp_2551_:
{
lean_object* v___x_2555_; 
if (v_isShared_2553_ == 0)
{
v___x_2555_ = v___x_2552_;
goto v_reusejp_2554_;
}
else
{
lean_object* v_reuseFailAlloc_2556_; 
v_reuseFailAlloc_2556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2556_, 0, v_a_2550_);
v___x_2555_ = v_reuseFailAlloc_2556_;
goto v_reusejp_2554_;
}
v_reusejp_2554_:
{
return v___x_2555_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___lam__3___boxed(lean_object* v___x_2558_, lean_object* v___x_2559_, lean_object* v___x_2560_, lean_object* v_fst_2561_, lean_object* v_goal_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
uint8_t v___x_8463__boxed_2568_; lean_object* v_res_2569_; 
v___x_8463__boxed_2568_ = lean_unbox(v___x_2559_);
v_res_2569_ = lp_mathlib_ExistsAndEq_construct___lam__3(v___x_2558_, v___x_8463__boxed_2568_, v___x_2560_, v_fst_2561_, v_goal_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
lean_dec(v___y_2564_);
return v_res_2569_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__2(void){
_start:
{
lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; 
v___x_2572_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__1));
v___x_2573_ = lean_unsigned_to_nat(30u);
v___x_2574_ = lean_unsigned_to_nat(190u);
v___x_2575_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2576_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2577_ = l_mkPanicMessageWithDecl(v___x_2576_, v___x_2575_, v___x_2574_, v___x_2573_, v___x_2572_);
return v___x_2577_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__6(void){
_start:
{
lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2587_; 
v___x_2582_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__5));
v___x_2583_ = lean_unsigned_to_nat(8u);
v___x_2584_ = lean_unsigned_to_nat(214u);
v___x_2585_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2586_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2587_ = l_mkPanicMessageWithDecl(v___x_2586_, v___x_2585_, v___x_2584_, v___x_2583_, v___x_2582_);
return v___x_2587_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__8(void){
_start:
{
lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2594_; 
v___x_2589_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__7));
v___x_2590_ = lean_unsigned_to_nat(12u);
v___x_2591_ = lean_unsigned_to_nat(216u);
v___x_2592_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2593_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2594_ = l_mkPanicMessageWithDecl(v___x_2593_, v___x_2592_, v___x_2591_, v___x_2590_, v___x_2589_);
return v___x_2594_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__11(void){
_start:
{
lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; 
v___x_2599_ = lean_box(0);
v___x_2600_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__10));
v___x_2601_ = l_Lean_Expr_const___override(v___x_2600_, v___x_2599_);
return v___x_2601_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__13(void){
_start:
{
lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; 
v___x_2603_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__12));
v___x_2604_ = lean_unsigned_to_nat(8u);
v___x_2605_ = lean_unsigned_to_nat(204u);
v___x_2606_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2607_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2608_ = l_mkPanicMessageWithDecl(v___x_2607_, v___x_2606_, v___x_2605_, v___x_2604_, v___x_2603_);
return v___x_2608_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__15(void){
_start:
{
lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; 
v___x_2610_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__14));
v___x_2611_ = lean_unsigned_to_nat(12u);
v___x_2612_ = lean_unsigned_to_nat(206u);
v___x_2613_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2614_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2615_ = l_mkPanicMessageWithDecl(v___x_2614_, v___x_2613_, v___x_2612_, v___x_2611_, v___x_2610_);
return v___x_2615_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__16(void){
_start:
{
lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2616_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__7));
v___x_2617_ = lean_unsigned_to_nat(24u);
v___x_2618_ = lean_unsigned_to_nat(222u);
v___x_2619_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2620_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2621_ = l_mkPanicMessageWithDecl(v___x_2620_, v___x_2619_, v___x_2618_, v___x_2617_, v___x_2616_);
return v___x_2621_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__17(void){
_start:
{
lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; 
v___x_2622_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___closed__6));
v___x_2623_ = lean_unsigned_to_nat(12u);
v___x_2624_ = lean_unsigned_to_nat(195u);
v___x_2625_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2626_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2627_ = l_mkPanicMessageWithDecl(v___x_2626_, v___x_2625_, v___x_2624_, v___x_2623_, v___x_2622_);
return v___x_2627_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_construct___closed__19(void){
_start:
{
lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; 
v___x_2629_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__18));
v___x_2630_ = lean_unsigned_to_nat(8u);
v___x_2631_ = lean_unsigned_to_nat(198u);
v___x_2632_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__0));
v___x_2633_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_2634_ = l_mkPanicMessageWithDecl(v___x_2633_, v___x_2632_, v___x_2631_, v___x_2630_, v___x_2629_);
return v___x_2634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct(lean_object* v_goal_2638_, lean_object* v_exs_2639_, lean_object* v_path_2640_, lean_object* v_leaves_2641_, lean_object* v_a_2642_, lean_object* v_a_2643_, lean_object* v_a_2644_, lean_object* v_a_2645_){
_start:
{
if (lean_obj_tag(v_path_2640_) == 0)
{
lean_object* v___f_2647_; uint8_t v___x_2648_; lean_object* v___x_2649_; 
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
v___f_2647_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_construct___lam__0___boxed), 6, 1);
lean_closure_set(v___f_2647_, 0, v_goal_2638_);
v___x_2648_ = 0;
v___x_2649_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_2647_, v___x_2648_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2649_) == 0)
{
lean_object* v_a_2650_; lean_object* v___x_2652_; uint8_t v_isShared_2653_; uint8_t v_isSharedCheck_2680_; 
v_a_2650_ = lean_ctor_get(v___x_2649_, 0);
v_isSharedCheck_2680_ = !lean_is_exclusive(v___x_2649_);
if (v_isSharedCheck_2680_ == 0)
{
v___x_2652_ = v___x_2649_;
v_isShared_2653_ = v_isSharedCheck_2680_;
goto v_resetjp_2651_;
}
else
{
lean_inc(v_a_2650_);
lean_dec(v___x_2649_);
v___x_2652_ = lean_box(0);
v_isShared_2653_ = v_isSharedCheck_2680_;
goto v_resetjp_2651_;
}
v_resetjp_2651_:
{
lean_object* v_snd_2654_; lean_object* v_snd_2655_; lean_object* v_snd_2656_; lean_object* v_snd_2657_; lean_object* v___x_2659_; uint8_t v_isShared_2660_; uint8_t v_isSharedCheck_2678_; 
v_snd_2654_ = lean_ctor_get(v_a_2650_, 1);
lean_inc(v_snd_2654_);
v_snd_2655_ = lean_ctor_get(v_snd_2654_, 1);
lean_inc(v_snd_2655_);
v_snd_2656_ = lean_ctor_get(v_snd_2655_, 1);
lean_inc(v_snd_2656_);
v_snd_2657_ = lean_ctor_get(v_snd_2656_, 1);
v_isSharedCheck_2678_ = !lean_is_exclusive(v_snd_2656_);
if (v_isSharedCheck_2678_ == 0)
{
lean_object* v_unused_2679_; 
v_unused_2679_ = lean_ctor_get(v_snd_2656_, 0);
lean_dec(v_unused_2679_);
v___x_2659_ = v_snd_2656_;
v_isShared_2660_ = v_isSharedCheck_2678_;
goto v_resetjp_2658_;
}
else
{
lean_inc(v_snd_2657_);
lean_dec(v_snd_2656_);
v___x_2659_ = lean_box(0);
v_isShared_2660_ = v_isSharedCheck_2678_;
goto v_resetjp_2658_;
}
v_resetjp_2658_:
{
uint8_t v___x_2661_; 
v___x_2661_ = lean_unbox(v_snd_2657_);
lean_dec(v_snd_2657_);
if (v___x_2661_ == 0)
{
lean_object* v___x_2662_; lean_object* v___x_2663_; 
lean_del_object(v___x_2659_);
lean_dec(v_snd_2655_);
lean_dec(v_snd_2654_);
lean_del_object(v___x_2652_);
lean_dec(v_a_2650_);
v___x_2662_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__2, &lp_mathlib_ExistsAndEq_construct___closed__2_once, _init_lp_mathlib_ExistsAndEq_construct___closed__2);
v___x_2663_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2662_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2663_;
}
else
{
lean_object* v_fst_2664_; lean_object* v_fst_2665_; lean_object* v_fst_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2670_; 
v_fst_2664_ = lean_ctor_get(v_a_2650_, 0);
lean_inc(v_fst_2664_);
lean_dec(v_a_2650_);
v_fst_2665_ = lean_ctor_get(v_snd_2654_, 0);
lean_inc(v_fst_2665_);
lean_dec(v_snd_2654_);
v_fst_2666_ = lean_ctor_get(v_snd_2655_, 0);
lean_inc(v_fst_2666_);
lean_dec(v_snd_2655_);
v___x_2667_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__4));
v___x_2668_ = lean_box(0);
if (v_isShared_2660_ == 0)
{
lean_ctor_set_tag(v___x_2659_, 1);
lean_ctor_set(v___x_2659_, 1, v___x_2668_);
lean_ctor_set(v___x_2659_, 0, v_fst_2664_);
v___x_2670_ = v___x_2659_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2677_; 
v_reuseFailAlloc_2677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2677_, 0, v_fst_2664_);
lean_ctor_set(v_reuseFailAlloc_2677_, 1, v___x_2668_);
v___x_2670_ = v_reuseFailAlloc_2677_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; lean_object* v___x_2675_; 
v___x_2671_ = l_Lean_Expr_const___override(v___x_2667_, v___x_2670_);
v___x_2672_ = l_Lean_Expr_app___override(v___x_2671_, v_fst_2665_);
v___x_2673_ = l_Lean_Expr_app___override(v___x_2672_, v_fst_2666_);
if (v_isShared_2653_ == 0)
{
lean_ctor_set(v___x_2652_, 0, v___x_2673_);
v___x_2675_ = v___x_2652_;
goto v_reusejp_2674_;
}
else
{
lean_object* v_reuseFailAlloc_2676_; 
v_reuseFailAlloc_2676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2676_, 0, v___x_2673_);
v___x_2675_ = v_reuseFailAlloc_2676_;
goto v_reusejp_2674_;
}
v_reusejp_2674_:
{
return v___x_2675_;
}
}
}
}
}
}
else
{
lean_object* v_a_2681_; lean_object* v___x_2683_; uint8_t v_isShared_2684_; uint8_t v_isSharedCheck_2688_; 
v_a_2681_ = lean_ctor_get(v___x_2649_, 0);
v_isSharedCheck_2688_ = !lean_is_exclusive(v___x_2649_);
if (v_isSharedCheck_2688_ == 0)
{
v___x_2683_ = v___x_2649_;
v_isShared_2684_ = v_isSharedCheck_2688_;
goto v_resetjp_2682_;
}
else
{
lean_inc(v_a_2681_);
lean_dec(v___x_2649_);
v___x_2683_ = lean_box(0);
v_isShared_2684_ = v_isSharedCheck_2688_;
goto v_resetjp_2682_;
}
v_resetjp_2682_:
{
lean_object* v___x_2686_; 
if (v_isShared_2684_ == 0)
{
v___x_2686_ = v___x_2683_;
goto v_reusejp_2685_;
}
else
{
lean_object* v_reuseFailAlloc_2687_; 
v_reuseFailAlloc_2687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2687_, 0, v_a_2681_);
v___x_2686_ = v_reuseFailAlloc_2687_;
goto v_reusejp_2685_;
}
v_reusejp_2685_:
{
return v___x_2686_;
}
}
}
}
else
{
lean_object* v_head_2689_; uint8_t v___x_2690_; 
v_head_2689_ = lean_ctor_get(v_path_2640_, 0);
v___x_2690_ = lean_unbox(v_head_2689_);
switch(v___x_2690_)
{
case 0:
{
lean_object* v_tail_2691_; lean_object* v___x_2692_; uint8_t v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___f_2696_; uint8_t v___x_2697_; lean_object* v___x_2698_; 
v_tail_2691_ = lean_ctor_get(v_path_2640_, 1);
lean_inc(v_tail_2691_);
lean_dec_ref_known(v_path_2640_, 2);
v___x_2692_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_2693_ = 0;
v___x_2694_ = lean_box(0);
v___x_2695_ = lean_box(v___x_2693_);
v___f_2696_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_construct___lam__1___boxed), 9, 4);
lean_closure_set(v___f_2696_, 0, v___x_2692_);
lean_closure_set(v___f_2696_, 1, v___x_2695_);
lean_closure_set(v___f_2696_, 2, v___x_2694_);
lean_closure_set(v___f_2696_, 3, v_goal_2638_);
v___x_2697_ = 0;
v___x_2698_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_2696_, v___x_2697_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2698_) == 0)
{
lean_object* v_a_2699_; lean_object* v_snd_2700_; lean_object* v_snd_2701_; uint8_t v___x_2702_; 
v_a_2699_ = lean_ctor_get(v___x_2698_, 0);
lean_inc(v_a_2699_);
lean_dec_ref_known(v___x_2698_, 1);
v_snd_2700_ = lean_ctor_get(v_a_2699_, 1);
lean_inc(v_snd_2700_);
v_snd_2701_ = lean_ctor_get(v_snd_2700_, 1);
v___x_2702_ = lean_unbox(v_snd_2701_);
if (v___x_2702_ == 0)
{
lean_object* v___x_2703_; lean_object* v___x_2704_; 
lean_dec(v_snd_2700_);
lean_dec(v_a_2699_);
lean_dec(v_tail_2691_);
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
v___x_2703_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__6, &lp_mathlib_ExistsAndEq_construct___closed__6_once, _init_lp_mathlib_ExistsAndEq_construct___closed__6);
v___x_2704_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2703_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2704_;
}
else
{
if (lean_obj_tag(v_leaves_2641_) == 0)
{
lean_object* v___x_2705_; lean_object* v___x_2706_; 
lean_dec(v_snd_2700_);
lean_dec(v_a_2699_);
lean_dec(v_tail_2691_);
lean_dec(v_exs_2639_);
v___x_2705_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__8, &lp_mathlib_ExistsAndEq_construct___closed__8_once, _init_lp_mathlib_ExistsAndEq_construct___closed__8);
v___x_2706_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2705_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2706_;
}
else
{
lean_object* v_head_2707_; lean_object* v_fst_2708_; lean_object* v_fst_2709_; lean_object* v_tail_2710_; lean_object* v_snd_2711_; lean_object* v___x_2712_; 
v_head_2707_ = lean_ctor_get(v_leaves_2641_, 0);
lean_inc(v_head_2707_);
v_fst_2708_ = lean_ctor_get(v_a_2699_, 0);
lean_inc_n(v_fst_2708_, 2);
lean_dec(v_a_2699_);
v_fst_2709_ = lean_ctor_get(v_snd_2700_, 0);
lean_inc(v_fst_2709_);
lean_dec(v_snd_2700_);
v_tail_2710_ = lean_ctor_get(v_leaves_2641_, 1);
lean_inc(v_tail_2710_);
lean_dec_ref_known(v_leaves_2641_, 2);
v_snd_2711_ = lean_ctor_get(v_head_2707_, 1);
lean_inc(v_snd_2711_);
lean_dec(v_head_2707_);
v___x_2712_ = lp_mathlib_ExistsAndEq_construct(v_fst_2708_, v_exs_2639_, v_tail_2691_, v_tail_2710_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2712_) == 0)
{
lean_object* v_a_2713_; lean_object* v___x_2715_; uint8_t v_isShared_2716_; uint8_t v_isSharedCheck_2725_; 
v_a_2713_ = lean_ctor_get(v___x_2712_, 0);
v_isSharedCheck_2725_ = !lean_is_exclusive(v___x_2712_);
if (v_isSharedCheck_2725_ == 0)
{
v___x_2715_ = v___x_2712_;
v_isShared_2716_ = v_isSharedCheck_2725_;
goto v_resetjp_2714_;
}
else
{
lean_inc(v_a_2713_);
lean_dec(v___x_2712_);
v___x_2715_ = lean_box(0);
v_isShared_2716_ = v_isSharedCheck_2725_;
goto v_resetjp_2714_;
}
v_resetjp_2714_:
{
lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2723_; 
v___x_2717_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__11, &lp_mathlib_ExistsAndEq_construct___closed__11_once, _init_lp_mathlib_ExistsAndEq_construct___closed__11);
v___x_2718_ = l_Lean_Expr_app___override(v___x_2717_, v_fst_2708_);
v___x_2719_ = l_Lean_Expr_app___override(v___x_2718_, v_fst_2709_);
v___x_2720_ = l_Lean_Expr_app___override(v___x_2719_, v_a_2713_);
v___x_2721_ = l_Lean_Expr_app___override(v___x_2720_, v_snd_2711_);
if (v_isShared_2716_ == 0)
{
lean_ctor_set(v___x_2715_, 0, v___x_2721_);
v___x_2723_ = v___x_2715_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2724_; 
v_reuseFailAlloc_2724_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2724_, 0, v___x_2721_);
v___x_2723_ = v_reuseFailAlloc_2724_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
return v___x_2723_;
}
}
}
else
{
lean_dec(v_snd_2711_);
lean_dec(v_fst_2709_);
lean_dec(v_fst_2708_);
return v___x_2712_;
}
}
}
}
else
{
lean_object* v_a_2726_; lean_object* v___x_2728_; uint8_t v_isShared_2729_; uint8_t v_isSharedCheck_2733_; 
lean_dec(v_tail_2691_);
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
v_a_2726_ = lean_ctor_get(v___x_2698_, 0);
v_isSharedCheck_2733_ = !lean_is_exclusive(v___x_2698_);
if (v_isSharedCheck_2733_ == 0)
{
v___x_2728_ = v___x_2698_;
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
else
{
lean_inc(v_a_2726_);
lean_dec(v___x_2698_);
v___x_2728_ = lean_box(0);
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
v_resetjp_2727_:
{
lean_object* v___x_2731_; 
if (v_isShared_2729_ == 0)
{
v___x_2731_ = v___x_2728_;
goto v_reusejp_2730_;
}
else
{
lean_object* v_reuseFailAlloc_2732_; 
v_reuseFailAlloc_2732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2732_, 0, v_a_2726_);
v___x_2731_ = v_reuseFailAlloc_2732_;
goto v_reusejp_2730_;
}
v_reusejp_2730_:
{
return v___x_2731_;
}
}
}
}
case 1:
{
lean_object* v_tail_2734_; lean_object* v___x_2735_; uint8_t v___x_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___f_2739_; uint8_t v___x_2740_; lean_object* v___x_2741_; 
v_tail_2734_ = lean_ctor_get(v_path_2640_, 1);
lean_inc(v_tail_2734_);
lean_dec_ref_known(v_path_2640_, 2);
v___x_2735_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___closed__3);
v___x_2736_ = 0;
v___x_2737_ = lean_box(0);
v___x_2738_ = lean_box(v___x_2736_);
v___f_2739_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_construct___lam__1___boxed), 9, 4);
lean_closure_set(v___f_2739_, 0, v___x_2735_);
lean_closure_set(v___f_2739_, 1, v___x_2738_);
lean_closure_set(v___f_2739_, 2, v___x_2737_);
lean_closure_set(v___f_2739_, 3, v_goal_2638_);
v___x_2740_ = 0;
v___x_2741_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_2739_, v___x_2740_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2741_) == 0)
{
lean_object* v_a_2742_; lean_object* v_snd_2743_; lean_object* v_snd_2744_; uint8_t v___x_2745_; 
v_a_2742_ = lean_ctor_get(v___x_2741_, 0);
lean_inc(v_a_2742_);
lean_dec_ref_known(v___x_2741_, 1);
v_snd_2743_ = lean_ctor_get(v_a_2742_, 1);
lean_inc(v_snd_2743_);
v_snd_2744_ = lean_ctor_get(v_snd_2743_, 1);
v___x_2745_ = lean_unbox(v_snd_2744_);
if (v___x_2745_ == 0)
{
lean_object* v___x_2746_; lean_object* v___x_2747_; 
lean_dec(v_snd_2743_);
lean_dec(v_a_2742_);
lean_dec(v_tail_2734_);
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
v___x_2746_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__13, &lp_mathlib_ExistsAndEq_construct___closed__13_once, _init_lp_mathlib_ExistsAndEq_construct___closed__13);
v___x_2747_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2746_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2747_;
}
else
{
if (lean_obj_tag(v_leaves_2641_) == 0)
{
lean_object* v___x_2748_; lean_object* v___x_2749_; 
lean_dec(v_snd_2743_);
lean_dec(v_a_2742_);
lean_dec(v_tail_2734_);
lean_dec(v_exs_2639_);
v___x_2748_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__15, &lp_mathlib_ExistsAndEq_construct___closed__15_once, _init_lp_mathlib_ExistsAndEq_construct___closed__15);
v___x_2749_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2748_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2749_;
}
else
{
lean_object* v_head_2750_; lean_object* v_fst_2751_; lean_object* v_fst_2752_; lean_object* v_tail_2753_; lean_object* v_snd_2754_; lean_object* v___x_2755_; 
v_head_2750_ = lean_ctor_get(v_leaves_2641_, 0);
lean_inc(v_head_2750_);
v_fst_2751_ = lean_ctor_get(v_a_2742_, 0);
lean_inc(v_fst_2751_);
lean_dec(v_a_2742_);
v_fst_2752_ = lean_ctor_get(v_snd_2743_, 0);
lean_inc_n(v_fst_2752_, 2);
lean_dec(v_snd_2743_);
v_tail_2753_ = lean_ctor_get(v_leaves_2641_, 1);
lean_inc(v_tail_2753_);
lean_dec_ref_known(v_leaves_2641_, 2);
v_snd_2754_ = lean_ctor_get(v_head_2750_, 1);
lean_inc(v_snd_2754_);
lean_dec(v_head_2750_);
v___x_2755_ = lp_mathlib_ExistsAndEq_construct(v_fst_2752_, v_exs_2639_, v_tail_2734_, v_tail_2753_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2755_) == 0)
{
lean_object* v_a_2756_; lean_object* v___x_2758_; uint8_t v_isShared_2759_; uint8_t v_isSharedCheck_2768_; 
v_a_2756_ = lean_ctor_get(v___x_2755_, 0);
v_isSharedCheck_2768_ = !lean_is_exclusive(v___x_2755_);
if (v_isSharedCheck_2768_ == 0)
{
v___x_2758_ = v___x_2755_;
v_isShared_2759_ = v_isSharedCheck_2768_;
goto v_resetjp_2757_;
}
else
{
lean_inc(v_a_2756_);
lean_dec(v___x_2755_);
v___x_2758_ = lean_box(0);
v_isShared_2759_ = v_isSharedCheck_2768_;
goto v_resetjp_2757_;
}
v_resetjp_2757_:
{
lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; lean_object* v___x_2766_; 
v___x_2760_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__11, &lp_mathlib_ExistsAndEq_construct___closed__11_once, _init_lp_mathlib_ExistsAndEq_construct___closed__11);
v___x_2761_ = l_Lean_Expr_app___override(v___x_2760_, v_fst_2751_);
v___x_2762_ = l_Lean_Expr_app___override(v___x_2761_, v_fst_2752_);
v___x_2763_ = l_Lean_Expr_app___override(v___x_2762_, v_snd_2754_);
v___x_2764_ = l_Lean_Expr_app___override(v___x_2763_, v_a_2756_);
if (v_isShared_2759_ == 0)
{
lean_ctor_set(v___x_2758_, 0, v___x_2764_);
v___x_2766_ = v___x_2758_;
goto v_reusejp_2765_;
}
else
{
lean_object* v_reuseFailAlloc_2767_; 
v_reuseFailAlloc_2767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2767_, 0, v___x_2764_);
v___x_2766_ = v_reuseFailAlloc_2767_;
goto v_reusejp_2765_;
}
v_reusejp_2765_:
{
return v___x_2766_;
}
}
}
else
{
lean_dec(v_snd_2754_);
lean_dec(v_fst_2752_);
lean_dec(v_fst_2751_);
return v___x_2755_;
}
}
}
}
else
{
lean_object* v_a_2769_; lean_object* v___x_2771_; uint8_t v_isShared_2772_; uint8_t v_isSharedCheck_2776_; 
lean_dec(v_tail_2734_);
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
v_a_2769_ = lean_ctor_get(v___x_2741_, 0);
v_isSharedCheck_2776_ = !lean_is_exclusive(v___x_2741_);
if (v_isSharedCheck_2776_ == 0)
{
v___x_2771_ = v___x_2741_;
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
else
{
lean_inc(v_a_2769_);
lean_dec(v___x_2741_);
v___x_2771_ = lean_box(0);
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
v_resetjp_2770_:
{
lean_object* v___x_2774_; 
if (v_isShared_2772_ == 0)
{
v___x_2774_ = v___x_2771_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2775_; 
v_reuseFailAlloc_2775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2775_, 0, v_a_2769_);
v___x_2774_ = v_reuseFailAlloc_2775_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
return v___x_2774_;
}
}
}
}
case 2:
{
lean_object* v___x_2777_; lean_object* v___x_2778_; 
lean_dec_ref_known(v_path_2640_, 2);
lean_dec(v_leaves_2641_);
lean_dec(v_exs_2639_);
lean_dec_ref(v_goal_2638_);
v___x_2777_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__16, &lp_mathlib_ExistsAndEq_construct___closed__16_once, _init_lp_mathlib_ExistsAndEq_construct___closed__16);
v___x_2778_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2777_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2778_;
}
default: 
{
if (lean_obj_tag(v_exs_2639_) == 0)
{
lean_object* v___x_2779_; lean_object* v___x_2780_; 
lean_dec_ref_known(v_path_2640_, 2);
lean_dec(v_leaves_2641_);
lean_dec_ref(v_goal_2638_);
v___x_2779_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__17, &lp_mathlib_ExistsAndEq_construct___closed__17_once, _init_lp_mathlib_ExistsAndEq_construct___closed__17);
v___x_2780_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2779_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2780_;
}
else
{
lean_object* v_head_2781_; lean_object* v_snd_2782_; lean_object* v_tail_2783_; lean_object* v___x_2785_; uint8_t v_isShared_2786_; uint8_t v_isSharedCheck_2843_; 
v_head_2781_ = lean_ctor_get(v_exs_2639_, 0);
lean_inc(v_head_2781_);
v_snd_2782_ = lean_ctor_get(v_head_2781_, 1);
lean_inc(v_snd_2782_);
v_tail_2783_ = lean_ctor_get(v_path_2640_, 1);
v_isSharedCheck_2843_ = !lean_is_exclusive(v_path_2640_);
if (v_isSharedCheck_2843_ == 0)
{
lean_object* v_unused_2844_; 
v_unused_2844_ = lean_ctor_get(v_path_2640_, 0);
lean_dec(v_unused_2844_);
v___x_2785_ = v_path_2640_;
v_isShared_2786_ = v_isSharedCheck_2843_;
goto v_resetjp_2784_;
}
else
{
lean_inc(v_tail_2783_);
lean_dec(v_path_2640_);
v___x_2785_ = lean_box(0);
v_isShared_2786_ = v_isSharedCheck_2843_;
goto v_resetjp_2784_;
}
v_resetjp_2784_:
{
lean_object* v_tail_2787_; lean_object* v___x_2789_; uint8_t v_isShared_2790_; uint8_t v_isSharedCheck_2841_; 
v_tail_2787_ = lean_ctor_get(v_exs_2639_, 1);
v_isSharedCheck_2841_ = !lean_is_exclusive(v_exs_2639_);
if (v_isSharedCheck_2841_ == 0)
{
lean_object* v_unused_2842_; 
v_unused_2842_ = lean_ctor_get(v_exs_2639_, 0);
lean_dec(v_unused_2842_);
v___x_2789_ = v_exs_2639_;
v_isShared_2790_ = v_isSharedCheck_2841_;
goto v_resetjp_2788_;
}
else
{
lean_inc(v_tail_2787_);
lean_dec(v_exs_2639_);
v___x_2789_ = lean_box(0);
v_isShared_2790_ = v_isSharedCheck_2841_;
goto v_resetjp_2788_;
}
v_resetjp_2788_:
{
lean_object* v_fst_2791_; lean_object* v_snd_2792_; lean_object* v___x_2793_; lean_object* v___x_2794_; uint8_t v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; lean_object* v___f_2798_; uint8_t v___x_2799_; lean_object* v___x_2800_; 
v_fst_2791_ = lean_ctor_get(v_head_2781_, 0);
lean_inc_n(v_fst_2791_, 3);
lean_dec(v_head_2781_);
v_snd_2792_ = lean_ctor_get(v_snd_2782_, 1);
lean_inc(v_snd_2792_);
lean_dec(v_snd_2782_);
v___x_2793_ = l_Lean_Expr_sort___override(v_fst_2791_);
v___x_2794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2794_, 0, v___x_2793_);
v___x_2795_ = 0;
v___x_2796_ = lean_box(0);
v___x_2797_ = lean_box(v___x_2795_);
v___f_2798_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_construct___lam__3___boxed), 10, 5);
lean_closure_set(v___f_2798_, 0, v___x_2794_);
lean_closure_set(v___f_2798_, 1, v___x_2797_);
lean_closure_set(v___f_2798_, 2, v___x_2796_);
lean_closure_set(v___f_2798_, 3, v_fst_2791_);
lean_closure_set(v___f_2798_, 4, v_goal_2638_);
v___x_2799_ = 0;
v___x_2800_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_2798_, v___x_2799_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2800_) == 0)
{
lean_object* v_a_2801_; lean_object* v_snd_2802_; lean_object* v_snd_2803_; uint8_t v___x_2804_; 
v_a_2801_ = lean_ctor_get(v___x_2800_, 0);
lean_inc(v_a_2801_);
lean_dec_ref_known(v___x_2800_, 1);
v_snd_2802_ = lean_ctor_get(v_a_2801_, 1);
lean_inc(v_snd_2802_);
v_snd_2803_ = lean_ctor_get(v_snd_2802_, 1);
v___x_2804_ = lean_unbox(v_snd_2803_);
if (v___x_2804_ == 0)
{
lean_object* v___x_2805_; lean_object* v___x_2806_; 
lean_dec(v_snd_2802_);
lean_dec(v_a_2801_);
lean_dec(v_snd_2792_);
lean_dec(v_fst_2791_);
lean_del_object(v___x_2789_);
lean_dec(v_tail_2787_);
lean_del_object(v___x_2785_);
lean_dec(v_tail_2783_);
lean_dec(v_leaves_2641_);
v___x_2805_ = lean_obj_once(&lp_mathlib_ExistsAndEq_construct___closed__19, &lp_mathlib_ExistsAndEq_construct___closed__19_once, _init_lp_mathlib_ExistsAndEq_construct___closed__19);
v___x_2806_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_2805_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
return v___x_2806_;
}
else
{
lean_object* v_fst_2807_; lean_object* v_fst_2808_; lean_object* v___x_2809_; lean_object* v___x_2811_; 
v_fst_2807_ = lean_ctor_get(v_a_2801_, 0);
lean_inc(v_fst_2807_);
lean_dec(v_a_2801_);
v_fst_2808_ = lean_ctor_get(v_snd_2802_, 0);
lean_inc(v_fst_2808_);
lean_dec(v_snd_2802_);
v___x_2809_ = lean_box(0);
lean_inc(v_snd_2792_);
if (v_isShared_2790_ == 0)
{
lean_ctor_set(v___x_2789_, 1, v___x_2809_);
lean_ctor_set(v___x_2789_, 0, v_snd_2792_);
v___x_2811_ = v___x_2789_;
goto v_reusejp_2810_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v_snd_2792_);
lean_ctor_set(v_reuseFailAlloc_2832_, 1, v___x_2809_);
v___x_2811_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2810_;
}
v_reusejp_2810_:
{
lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; 
v___x_2812_ = lean_array_mk(v___x_2811_);
lean_inc(v_fst_2808_);
v___x_2813_ = l_Lean_Expr_betaRev(v_fst_2808_, v___x_2812_, v___x_2799_, v___x_2799_);
lean_dec_ref(v___x_2812_);
v___x_2814_ = lp_mathlib_ExistsAndEq_construct(v___x_2813_, v_tail_2787_, v_tail_2783_, v_leaves_2641_, v_a_2642_, v_a_2643_, v_a_2644_, v_a_2645_);
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v_a_2815_; lean_object* v___x_2817_; uint8_t v_isShared_2818_; uint8_t v_isSharedCheck_2831_; 
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
v_isSharedCheck_2831_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2817_ = v___x_2814_;
v_isShared_2818_ = v_isSharedCheck_2831_;
goto v_resetjp_2816_;
}
else
{
lean_inc(v_a_2815_);
lean_dec(v___x_2814_);
v___x_2817_ = lean_box(0);
v_isShared_2818_ = v_isSharedCheck_2831_;
goto v_resetjp_2816_;
}
v_resetjp_2816_:
{
lean_object* v___x_2819_; lean_object* v___x_2821_; 
v___x_2819_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__20));
if (v_isShared_2786_ == 0)
{
lean_ctor_set(v___x_2785_, 1, v___x_2809_);
lean_ctor_set(v___x_2785_, 0, v_fst_2791_);
v___x_2821_ = v___x_2785_;
goto v_reusejp_2820_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v_fst_2791_);
lean_ctor_set(v_reuseFailAlloc_2830_, 1, v___x_2809_);
v___x_2821_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2820_;
}
v_reusejp_2820_:
{
lean_object* v___x_2822_; lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2828_; 
v___x_2822_ = l_Lean_Expr_const___override(v___x_2819_, v___x_2821_);
v___x_2823_ = l_Lean_Expr_app___override(v___x_2822_, v_fst_2807_);
v___x_2824_ = l_Lean_Expr_app___override(v___x_2823_, v_fst_2808_);
v___x_2825_ = l_Lean_Expr_app___override(v___x_2824_, v_snd_2792_);
v___x_2826_ = l_Lean_Expr_app___override(v___x_2825_, v_a_2815_);
if (v_isShared_2818_ == 0)
{
lean_ctor_set(v___x_2817_, 0, v___x_2826_);
v___x_2828_ = v___x_2817_;
goto v_reusejp_2827_;
}
else
{
lean_object* v_reuseFailAlloc_2829_; 
v_reuseFailAlloc_2829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2829_, 0, v___x_2826_);
v___x_2828_ = v_reuseFailAlloc_2829_;
goto v_reusejp_2827_;
}
v_reusejp_2827_:
{
return v___x_2828_;
}
}
}
}
else
{
lean_dec(v_fst_2808_);
lean_dec(v_fst_2807_);
lean_dec(v_snd_2792_);
lean_dec(v_fst_2791_);
lean_del_object(v___x_2785_);
return v___x_2814_;
}
}
}
}
else
{
lean_object* v_a_2833_; lean_object* v___x_2835_; uint8_t v_isShared_2836_; uint8_t v_isSharedCheck_2840_; 
lean_dec(v_snd_2792_);
lean_dec(v_fst_2791_);
lean_del_object(v___x_2789_);
lean_dec(v_tail_2787_);
lean_del_object(v___x_2785_);
lean_dec(v_tail_2783_);
lean_dec(v_leaves_2641_);
v_a_2833_ = lean_ctor_get(v___x_2800_, 0);
v_isSharedCheck_2840_ = !lean_is_exclusive(v___x_2800_);
if (v_isSharedCheck_2840_ == 0)
{
v___x_2835_ = v___x_2800_;
v_isShared_2836_ = v_isSharedCheck_2840_;
goto v_resetjp_2834_;
}
else
{
lean_inc(v_a_2833_);
lean_dec(v___x_2800_);
v___x_2835_ = lean_box(0);
v_isShared_2836_ = v_isSharedCheck_2840_;
goto v_resetjp_2834_;
}
v_resetjp_2834_:
{
lean_object* v___x_2838_; 
if (v_isShared_2836_ == 0)
{
v___x_2838_ = v___x_2835_;
goto v_reusejp_2837_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v_a_2833_);
v___x_2838_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2837_;
}
v_reusejp_2837_:
{
return v___x_2838_;
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
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_construct___boxed(lean_object* v_goal_2845_, lean_object* v_exs_2846_, lean_object* v_path_2847_, lean_object* v_leaves_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_, lean_object* v_a_2852_, lean_object* v_a_2853_){
_start:
{
lean_object* v_res_2854_; 
v_res_2854_ = lp_mathlib_ExistsAndEq_construct(v_goal_2845_, v_exs_2846_, v_path_2847_, v_leaves_2848_, v_a_2849_, v_a_2850_, v_a_2851_, v_a_2852_);
lean_dec(v_a_2852_);
lean_dec_ref(v_a_2851_);
lean_dec(v_a_2850_);
lean_dec_ref(v_a_2849_);
return v_res_2854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg(lean_object* v_mvarId_2855_, lean_object* v_x_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_, lean_object* v___y_2859_, lean_object* v___y_2860_){
_start:
{
lean_object* v___x_2862_; 
v___x_2862_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2855_, v_x_2856_, v___y_2857_, v___y_2858_, v___y_2859_, v___y_2860_);
if (lean_obj_tag(v___x_2862_) == 0)
{
lean_object* v_a_2863_; lean_object* v___x_2865_; uint8_t v_isShared_2866_; uint8_t v_isSharedCheck_2870_; 
v_a_2863_ = lean_ctor_get(v___x_2862_, 0);
v_isSharedCheck_2870_ = !lean_is_exclusive(v___x_2862_);
if (v_isSharedCheck_2870_ == 0)
{
v___x_2865_ = v___x_2862_;
v_isShared_2866_ = v_isSharedCheck_2870_;
goto v_resetjp_2864_;
}
else
{
lean_inc(v_a_2863_);
lean_dec(v___x_2862_);
v___x_2865_ = lean_box(0);
v_isShared_2866_ = v_isSharedCheck_2870_;
goto v_resetjp_2864_;
}
v_resetjp_2864_:
{
lean_object* v___x_2868_; 
if (v_isShared_2866_ == 0)
{
v___x_2868_ = v___x_2865_;
goto v_reusejp_2867_;
}
else
{
lean_object* v_reuseFailAlloc_2869_; 
v_reuseFailAlloc_2869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2869_, 0, v_a_2863_);
v___x_2868_ = v_reuseFailAlloc_2869_;
goto v_reusejp_2867_;
}
v_reusejp_2867_:
{
return v___x_2868_;
}
}
}
else
{
lean_object* v_a_2871_; lean_object* v___x_2873_; uint8_t v_isShared_2874_; uint8_t v_isSharedCheck_2878_; 
v_a_2871_ = lean_ctor_get(v___x_2862_, 0);
v_isSharedCheck_2878_ = !lean_is_exclusive(v___x_2862_);
if (v_isSharedCheck_2878_ == 0)
{
v___x_2873_ = v___x_2862_;
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
else
{
lean_inc(v_a_2871_);
lean_dec(v___x_2862_);
v___x_2873_ = lean_box(0);
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
v_resetjp_2872_:
{
lean_object* v___x_2876_; 
if (v_isShared_2874_ == 0)
{
v___x_2876_ = v___x_2873_;
goto v_reusejp_2875_;
}
else
{
lean_object* v_reuseFailAlloc_2877_; 
v_reuseFailAlloc_2877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2877_, 0, v_a_2871_);
v___x_2876_ = v_reuseFailAlloc_2877_;
goto v_reusejp_2875_;
}
v_reusejp_2875_:
{
return v___x_2876_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg___boxed(lean_object* v_mvarId_2879_, lean_object* v_x_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_, lean_object* v___y_2885_){
_start:
{
lean_object* v_res_2886_; 
v_res_2886_ = lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg(v_mvarId_2879_, v_x_2880_, v___y_2881_, v___y_2882_, v___y_2883_, v___y_2884_);
lean_dec(v___y_2884_);
lean_dec_ref(v___y_2883_);
lean_dec(v___y_2882_);
lean_dec_ref(v___y_2881_);
return v_res_2886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2(lean_object* v_00_u03b1_2887_, lean_object* v_mvarId_2888_, lean_object* v_x_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_){
_start:
{
lean_object* v___x_2895_; 
v___x_2895_ = lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg(v_mvarId_2888_, v_x_2889_, v___y_2890_, v___y_2891_, v___y_2892_, v___y_2893_);
return v___x_2895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___boxed(lean_object* v_00_u03b1_2896_, lean_object* v_mvarId_2897_, lean_object* v_x_2898_, lean_object* v___y_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_, lean_object* v___y_2903_){
_start:
{
lean_object* v_res_2904_; 
v_res_2904_ = lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2(v_00_u03b1_2896_, v_mvarId_2897_, v_x_2898_, v___y_2899_, v___y_2900_, v___y_2901_, v___y_2902_);
lean_dec(v___y_2902_);
lean_dec_ref(v___y_2901_);
lean_dec(v___y_2900_);
lean_dec_ref(v___y_2899_);
return v_res_2904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(lean_object* v___x_2905_, uint8_t v___x_2906_, lean_object* v___x_2907_, lean_object* v___x_2908_, lean_object* v_fst_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_){
_start:
{
lean_object* v___x_2915_; 
lean_inc(v___x_2907_);
v___x_2915_ = l_Lean_Meta_mkFreshExprMVar(v___x_2905_, v___x_2906_, v___x_2907_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_);
if (lean_obj_tag(v___x_2915_) == 0)
{
lean_object* v_a_2916_; lean_object* v___x_2917_; lean_object* v___x_2918_; 
v_a_2916_ = lean_ctor_get(v___x_2915_, 0);
lean_inc_n(v_a_2916_, 2);
lean_dec_ref_known(v___x_2915_, 1);
v___x_2917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2917_, 0, v_a_2916_);
lean_inc(v___x_2907_);
lean_inc_ref(v___x_2917_);
v___x_2918_ = l_Lean_Meta_mkFreshExprMVar(v___x_2917_, v___x_2906_, v___x_2907_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_);
if (lean_obj_tag(v___x_2918_) == 0)
{
lean_object* v_a_2919_; lean_object* v___x_2920_; 
v_a_2919_ = lean_ctor_get(v___x_2918_, 0);
lean_inc(v_a_2919_);
lean_dec_ref_known(v___x_2918_, 1);
v___x_2920_ = l_Lean_Meta_mkFreshExprMVar(v___x_2917_, v___x_2906_, v___x_2907_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_);
if (lean_obj_tag(v___x_2920_) == 0)
{
lean_object* v_a_2921_; lean_object* v_keyedConfig_2922_; uint8_t v_trackZetaDelta_2923_; lean_object* v_zetaDeltaSet_2924_; lean_object* v_lctx_2925_; lean_object* v_localInstances_2926_; lean_object* v_defEqCtx_x3f_2927_; lean_object* v_synthPendingDepth_2928_; lean_object* v_customCanUnfoldPredicate_x3f_2929_; uint8_t v_univApprox_2930_; uint8_t v_inTypeClassResolution_2931_; uint8_t v_cacheInferType_2932_; lean_object* v___x_2934_; uint8_t v_isShared_2935_; uint8_t v_isSharedCheck_2983_; 
v_a_2921_ = lean_ctor_get(v___x_2920_, 0);
lean_inc(v_a_2921_);
lean_dec_ref_known(v___x_2920_, 1);
v_keyedConfig_2922_ = lean_ctor_get(v___y_2910_, 0);
v_trackZetaDelta_2923_ = lean_ctor_get_uint8(v___y_2910_, sizeof(void*)*7);
v_zetaDeltaSet_2924_ = lean_ctor_get(v___y_2910_, 1);
v_lctx_2925_ = lean_ctor_get(v___y_2910_, 2);
v_localInstances_2926_ = lean_ctor_get(v___y_2910_, 3);
v_defEqCtx_x3f_2927_ = lean_ctor_get(v___y_2910_, 4);
v_synthPendingDepth_2928_ = lean_ctor_get(v___y_2910_, 5);
v_customCanUnfoldPredicate_x3f_2929_ = lean_ctor_get(v___y_2910_, 6);
v_univApprox_2930_ = lean_ctor_get_uint8(v___y_2910_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2931_ = lean_ctor_get_uint8(v___y_2910_, sizeof(void*)*7 + 2);
v_cacheInferType_2932_ = lean_ctor_get_uint8(v___y_2910_, sizeof(void*)*7 + 3);
v_isSharedCheck_2983_ = !lean_is_exclusive(v___y_2910_);
if (v_isSharedCheck_2983_ == 0)
{
v___x_2934_ = v___y_2910_;
v_isShared_2935_ = v_isSharedCheck_2983_;
goto v_resetjp_2933_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2929_);
lean_inc(v_synthPendingDepth_2928_);
lean_inc(v_defEqCtx_x3f_2927_);
lean_inc(v_localInstances_2926_);
lean_inc(v_lctx_2925_);
lean_inc(v_zetaDeltaSet_2924_);
lean_inc(v_keyedConfig_2922_);
lean_dec(v___y_2910_);
v___x_2934_ = lean_box(0);
v_isShared_2935_ = v_isSharedCheck_2983_;
goto v_resetjp_2933_;
}
v_resetjp_2933_:
{
lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; uint8_t v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2944_; 
v___x_2936_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__5));
v___x_2937_ = l_Lean_Expr_const___override(v___x_2936_, v___x_2908_);
lean_inc(v_a_2916_);
v___x_2938_ = l_Lean_Expr_app___override(v___x_2937_, v_a_2916_);
lean_inc(v_a_2919_);
v___x_2939_ = l_Lean_Expr_app___override(v___x_2938_, v_a_2919_);
lean_inc(v_a_2921_);
v___x_2940_ = l_Lean_Expr_app___override(v___x_2939_, v_a_2921_);
v___x_2941_ = 2;
v___x_2942_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2941_, v_keyedConfig_2922_);
if (v_isShared_2935_ == 0)
{
lean_ctor_set(v___x_2934_, 0, v___x_2942_);
v___x_2944_ = v___x_2934_;
goto v_reusejp_2943_;
}
else
{
lean_object* v_reuseFailAlloc_2982_; 
v_reuseFailAlloc_2982_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2982_, 0, v___x_2942_);
lean_ctor_set(v_reuseFailAlloc_2982_, 1, v_zetaDeltaSet_2924_);
lean_ctor_set(v_reuseFailAlloc_2982_, 2, v_lctx_2925_);
lean_ctor_set(v_reuseFailAlloc_2982_, 3, v_localInstances_2926_);
lean_ctor_set(v_reuseFailAlloc_2982_, 4, v_defEqCtx_x3f_2927_);
lean_ctor_set(v_reuseFailAlloc_2982_, 5, v_synthPendingDepth_2928_);
lean_ctor_set(v_reuseFailAlloc_2982_, 6, v_customCanUnfoldPredicate_x3f_2929_);
lean_ctor_set_uint8(v_reuseFailAlloc_2982_, sizeof(void*)*7, v_trackZetaDelta_2923_);
lean_ctor_set_uint8(v_reuseFailAlloc_2982_, sizeof(void*)*7 + 1, v_univApprox_2930_);
lean_ctor_set_uint8(v_reuseFailAlloc_2982_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2931_);
lean_ctor_set_uint8(v_reuseFailAlloc_2982_, sizeof(void*)*7 + 3, v_cacheInferType_2932_);
v___x_2944_ = v_reuseFailAlloc_2982_;
goto v_reusejp_2943_;
}
v_reusejp_2943_:
{
lean_object* v___x_2945_; 
v___x_2945_ = l_Lean_Meta_isExprDefEq(v___x_2940_, v_fst_2909_, v___x_2944_, v___y_2911_, v___y_2912_, v___y_2913_);
lean_dec_ref(v___x_2944_);
if (lean_obj_tag(v___x_2945_) == 0)
{
lean_object* v_a_2946_; lean_object* v___x_2948_; uint8_t v_isShared_2949_; uint8_t v_isSharedCheck_2973_; 
v_a_2946_ = lean_ctor_get(v___x_2945_, 0);
v_isSharedCheck_2973_ = !lean_is_exclusive(v___x_2945_);
if (v_isSharedCheck_2973_ == 0)
{
v___x_2948_ = v___x_2945_;
v_isShared_2949_ = v_isSharedCheck_2973_;
goto v_resetjp_2947_;
}
else
{
lean_inc(v_a_2946_);
lean_dec(v___x_2945_);
v___x_2948_ = lean_box(0);
v_isShared_2949_ = v_isSharedCheck_2973_;
goto v_resetjp_2947_;
}
v_resetjp_2947_:
{
uint8_t v___x_2950_; 
v___x_2950_ = lean_unbox(v_a_2946_);
if (v___x_2950_ == 0)
{
lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2953_; lean_object* v___x_2955_; 
v___x_2951_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2951_, 0, v_a_2921_);
lean_ctor_set(v___x_2951_, 1, v_a_2946_);
v___x_2952_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2952_, 0, v_a_2919_);
lean_ctor_set(v___x_2952_, 1, v___x_2951_);
v___x_2953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2953_, 0, v_a_2916_);
lean_ctor_set(v___x_2953_, 1, v___x_2952_);
if (v_isShared_2949_ == 0)
{
lean_ctor_set(v___x_2948_, 0, v___x_2953_);
v___x_2955_ = v___x_2948_;
goto v_reusejp_2954_;
}
else
{
lean_object* v_reuseFailAlloc_2956_; 
v_reuseFailAlloc_2956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2956_, 0, v___x_2953_);
v___x_2955_ = v_reuseFailAlloc_2956_;
goto v_reusejp_2954_;
}
v_reusejp_2954_:
{
return v___x_2955_;
}
}
else
{
lean_object* v___x_2957_; lean_object* v_a_2958_; lean_object* v___x_2959_; lean_object* v_a_2960_; lean_object* v___x_2961_; lean_object* v_a_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_2972_; 
lean_del_object(v___x_2948_);
v___x_2957_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2916_, v___y_2911_);
v_a_2958_ = lean_ctor_get(v___x_2957_, 0);
lean_inc(v_a_2958_);
lean_dec_ref(v___x_2957_);
v___x_2959_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2919_, v___y_2911_);
v_a_2960_ = lean_ctor_get(v___x_2959_, 0);
lean_inc(v_a_2960_);
lean_dec_ref(v___x_2959_);
v___x_2961_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_2921_, v___y_2911_);
v_a_2962_ = lean_ctor_get(v___x_2961_, 0);
v_isSharedCheck_2972_ = !lean_is_exclusive(v___x_2961_);
if (v_isSharedCheck_2972_ == 0)
{
v___x_2964_ = v___x_2961_;
v_isShared_2965_ = v_isSharedCheck_2972_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_a_2962_);
lean_dec(v___x_2961_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_2972_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2970_; 
v___x_2966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2966_, 0, v_a_2962_);
lean_ctor_set(v___x_2966_, 1, v_a_2946_);
v___x_2967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2967_, 0, v_a_2960_);
lean_ctor_set(v___x_2967_, 1, v___x_2966_);
v___x_2968_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2968_, 0, v_a_2958_);
lean_ctor_set(v___x_2968_, 1, v___x_2967_);
if (v_isShared_2965_ == 0)
{
lean_ctor_set(v___x_2964_, 0, v___x_2968_);
v___x_2970_ = v___x_2964_;
goto v_reusejp_2969_;
}
else
{
lean_object* v_reuseFailAlloc_2971_; 
v_reuseFailAlloc_2971_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2971_, 0, v___x_2968_);
v___x_2970_ = v_reuseFailAlloc_2971_;
goto v_reusejp_2969_;
}
v_reusejp_2969_:
{
return v___x_2970_;
}
}
}
}
}
else
{
lean_object* v_a_2974_; lean_object* v___x_2976_; uint8_t v_isShared_2977_; uint8_t v_isSharedCheck_2981_; 
lean_dec(v_a_2921_);
lean_dec(v_a_2919_);
lean_dec(v_a_2916_);
v_a_2974_ = lean_ctor_get(v___x_2945_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2945_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2976_ = v___x_2945_;
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
else
{
lean_inc(v_a_2974_);
lean_dec(v___x_2945_);
v___x_2976_ = lean_box(0);
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
v_resetjp_2975_:
{
lean_object* v___x_2979_; 
if (v_isShared_2977_ == 0)
{
v___x_2979_ = v___x_2976_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_a_2974_);
v___x_2979_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
return v___x_2979_;
}
}
}
}
}
}
else
{
lean_object* v_a_2984_; lean_object* v___x_2986_; uint8_t v_isShared_2987_; uint8_t v_isSharedCheck_2991_; 
lean_dec(v_a_2919_);
lean_dec(v_a_2916_);
lean_dec_ref(v___y_2910_);
lean_dec_ref(v_fst_2909_);
lean_dec(v___x_2908_);
v_a_2984_ = lean_ctor_get(v___x_2920_, 0);
v_isSharedCheck_2991_ = !lean_is_exclusive(v___x_2920_);
if (v_isSharedCheck_2991_ == 0)
{
v___x_2986_ = v___x_2920_;
v_isShared_2987_ = v_isSharedCheck_2991_;
goto v_resetjp_2985_;
}
else
{
lean_inc(v_a_2984_);
lean_dec(v___x_2920_);
v___x_2986_ = lean_box(0);
v_isShared_2987_ = v_isSharedCheck_2991_;
goto v_resetjp_2985_;
}
v_resetjp_2985_:
{
lean_object* v___x_2989_; 
if (v_isShared_2987_ == 0)
{
v___x_2989_ = v___x_2986_;
goto v_reusejp_2988_;
}
else
{
lean_object* v_reuseFailAlloc_2990_; 
v_reuseFailAlloc_2990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2990_, 0, v_a_2984_);
v___x_2989_ = v_reuseFailAlloc_2990_;
goto v_reusejp_2988_;
}
v_reusejp_2988_:
{
return v___x_2989_;
}
}
}
}
else
{
lean_object* v_a_2992_; lean_object* v___x_2994_; uint8_t v_isShared_2995_; uint8_t v_isSharedCheck_2999_; 
lean_dec_ref_known(v___x_2917_, 1);
lean_dec(v_a_2916_);
lean_dec_ref(v___y_2910_);
lean_dec_ref(v_fst_2909_);
lean_dec(v___x_2908_);
lean_dec(v___x_2907_);
v_a_2992_ = lean_ctor_get(v___x_2918_, 0);
v_isSharedCheck_2999_ = !lean_is_exclusive(v___x_2918_);
if (v_isSharedCheck_2999_ == 0)
{
v___x_2994_ = v___x_2918_;
v_isShared_2995_ = v_isSharedCheck_2999_;
goto v_resetjp_2993_;
}
else
{
lean_inc(v_a_2992_);
lean_dec(v___x_2918_);
v___x_2994_ = lean_box(0);
v_isShared_2995_ = v_isSharedCheck_2999_;
goto v_resetjp_2993_;
}
v_resetjp_2993_:
{
lean_object* v___x_2997_; 
if (v_isShared_2995_ == 0)
{
v___x_2997_ = v___x_2994_;
goto v_reusejp_2996_;
}
else
{
lean_object* v_reuseFailAlloc_2998_; 
v_reuseFailAlloc_2998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2998_, 0, v_a_2992_);
v___x_2997_ = v_reuseFailAlloc_2998_;
goto v_reusejp_2996_;
}
v_reusejp_2996_:
{
return v___x_2997_;
}
}
}
}
else
{
lean_object* v_a_3000_; lean_object* v___x_3002_; uint8_t v_isShared_3003_; uint8_t v_isSharedCheck_3007_; 
lean_dec_ref(v___y_2910_);
lean_dec_ref(v_fst_2909_);
lean_dec(v___x_2908_);
lean_dec(v___x_2907_);
v_a_3000_ = lean_ctor_get(v___x_2915_, 0);
v_isSharedCheck_3007_ = !lean_is_exclusive(v___x_2915_);
if (v_isSharedCheck_3007_ == 0)
{
v___x_3002_ = v___x_2915_;
v_isShared_3003_ = v_isSharedCheck_3007_;
goto v_resetjp_3001_;
}
else
{
lean_inc(v_a_3000_);
lean_dec(v___x_2915_);
v___x_3002_ = lean_box(0);
v_isShared_3003_ = v_isSharedCheck_3007_;
goto v_resetjp_3001_;
}
v_resetjp_3001_:
{
lean_object* v___x_3005_; 
if (v_isShared_3003_ == 0)
{
v___x_3005_ = v___x_3002_;
goto v_reusejp_3004_;
}
else
{
lean_object* v_reuseFailAlloc_3006_; 
v_reuseFailAlloc_3006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3006_, 0, v_a_3000_);
v___x_3005_ = v_reuseFailAlloc_3006_;
goto v_reusejp_3004_;
}
v_reusejp_3004_:
{
return v___x_3005_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed(lean_object* v___x_3008_, lean_object* v___x_3009_, lean_object* v___x_3010_, lean_object* v___x_3011_, lean_object* v_fst_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_){
_start:
{
uint8_t v___x_4106__boxed_3018_; lean_object* v_res_3019_; 
v___x_4106__boxed_3018_ = lean_unbox(v___x_3009_);
v_res_3019_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0(v___x_3008_, v___x_4106__boxed_3018_, v___x_3010_, v___x_3011_, v_fst_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v___y_3014_);
return v_res_3019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0(lean_object* v_fst_3020_, lean_object* v_x_3021_, lean_object* v_x_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_){
_start:
{
if (lean_obj_tag(v_x_3021_) == 0)
{
lean_object* v___x_3028_; lean_object* v___x_3029_; 
lean_dec(v_fst_3020_);
v___x_3028_ = l_List_reverse___redArg(v_x_3022_);
v___x_3029_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3029_, 0, v___x_3028_);
return v___x_3029_;
}
else
{
lean_object* v_head_3030_; lean_object* v_tail_3031_; lean_object* v___x_3033_; uint8_t v_isShared_3034_; uint8_t v_isSharedCheck_3059_; 
v_head_3030_ = lean_ctor_get(v_x_3021_, 0);
v_tail_3031_ = lean_ctor_get(v_x_3021_, 1);
v_isSharedCheck_3059_ = !lean_is_exclusive(v_x_3021_);
if (v_isSharedCheck_3059_ == 0)
{
v___x_3033_ = v_x_3021_;
v_isShared_3034_ = v_isSharedCheck_3059_;
goto v_resetjp_3032_;
}
else
{
lean_inc(v_tail_3031_);
lean_inc(v_head_3030_);
lean_dec(v_x_3021_);
v___x_3033_ = lean_box(0);
v_isShared_3034_ = v_isSharedCheck_3059_;
goto v_resetjp_3032_;
}
v_resetjp_3032_:
{
lean_object* v_snd_3035_; lean_object* v___x_3037_; uint8_t v_isShared_3038_; uint8_t v_isSharedCheck_3057_; 
v_snd_3035_ = lean_ctor_get(v_head_3030_, 1);
v_isSharedCheck_3057_ = !lean_is_exclusive(v_head_3030_);
if (v_isSharedCheck_3057_ == 0)
{
lean_object* v_unused_3058_; 
v_unused_3058_ = lean_ctor_get(v_head_3030_, 0);
lean_dec(v_unused_3058_);
v___x_3037_ = v_head_3030_;
v_isShared_3038_ = v_isSharedCheck_3057_;
goto v_resetjp_3036_;
}
else
{
lean_inc(v_snd_3035_);
lean_dec(v_head_3030_);
v___x_3037_ = lean_box(0);
v_isShared_3038_ = v_isSharedCheck_3057_;
goto v_resetjp_3036_;
}
v_resetjp_3036_:
{
lean_object* v___x_3039_; lean_object* v___x_3040_; 
lean_inc(v_fst_3020_);
v___x_3039_ = l_Lean_Meta_FVarSubst_apply(v_fst_3020_, v_snd_3035_);
lean_dec(v_snd_3035_);
lean_inc(v___y_3026_);
lean_inc_ref(v___y_3025_);
lean_inc(v___y_3024_);
lean_inc_ref(v___y_3023_);
lean_inc_ref(v___x_3039_);
v___x_3040_ = lean_infer_type(v___x_3039_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3040_) == 0)
{
lean_object* v_a_3041_; lean_object* v___x_3043_; 
v_a_3041_ = lean_ctor_get(v___x_3040_, 0);
lean_inc(v_a_3041_);
lean_dec_ref_known(v___x_3040_, 1);
if (v_isShared_3038_ == 0)
{
lean_ctor_set(v___x_3037_, 1, v___x_3039_);
lean_ctor_set(v___x_3037_, 0, v_a_3041_);
v___x_3043_ = v___x_3037_;
goto v_reusejp_3042_;
}
else
{
lean_object* v_reuseFailAlloc_3048_; 
v_reuseFailAlloc_3048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3048_, 0, v_a_3041_);
lean_ctor_set(v_reuseFailAlloc_3048_, 1, v___x_3039_);
v___x_3043_ = v_reuseFailAlloc_3048_;
goto v_reusejp_3042_;
}
v_reusejp_3042_:
{
lean_object* v___x_3045_; 
if (v_isShared_3034_ == 0)
{
lean_ctor_set(v___x_3033_, 1, v_x_3022_);
lean_ctor_set(v___x_3033_, 0, v___x_3043_);
v___x_3045_ = v___x_3033_;
goto v_reusejp_3044_;
}
else
{
lean_object* v_reuseFailAlloc_3047_; 
v_reuseFailAlloc_3047_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3047_, 0, v___x_3043_);
lean_ctor_set(v_reuseFailAlloc_3047_, 1, v_x_3022_);
v___x_3045_ = v_reuseFailAlloc_3047_;
goto v_reusejp_3044_;
}
v_reusejp_3044_:
{
v_x_3021_ = v_tail_3031_;
v_x_3022_ = v___x_3045_;
goto _start;
}
}
}
else
{
lean_object* v_a_3049_; lean_object* v___x_3051_; uint8_t v_isShared_3052_; uint8_t v_isSharedCheck_3056_; 
lean_dec_ref(v___x_3039_);
lean_del_object(v___x_3037_);
lean_del_object(v___x_3033_);
lean_dec(v_tail_3031_);
lean_dec(v_x_3022_);
lean_dec(v_fst_3020_);
v_a_3049_ = lean_ctor_get(v___x_3040_, 0);
v_isSharedCheck_3056_ = !lean_is_exclusive(v___x_3040_);
if (v_isSharedCheck_3056_ == 0)
{
v___x_3051_ = v___x_3040_;
v_isShared_3052_ = v_isSharedCheck_3056_;
goto v_resetjp_3050_;
}
else
{
lean_inc(v_a_3049_);
lean_dec(v___x_3040_);
v___x_3051_ = lean_box(0);
v_isShared_3052_ = v_isSharedCheck_3056_;
goto v_resetjp_3050_;
}
v_resetjp_3050_:
{
lean_object* v___x_3054_; 
if (v_isShared_3052_ == 0)
{
v___x_3054_ = v___x_3051_;
goto v_reusejp_3053_;
}
else
{
lean_object* v_reuseFailAlloc_3055_; 
v_reuseFailAlloc_3055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3055_, 0, v_a_3049_);
v___x_3054_ = v_reuseFailAlloc_3055_;
goto v_reusejp_3053_;
}
v_reusejp_3053_:
{
return v___x_3054_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0___boxed(lean_object* v_fst_3060_, lean_object* v_x_3061_, lean_object* v_x_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_){
_start:
{
lean_object* v_res_3068_; 
v_res_3068_ = lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0(v_fst_3060_, v_x_3061_, v_x_3062_, v___y_3063_, v___y_3064_, v___y_3065_, v___y_3066_);
lean_dec(v___y_3066_);
lean_dec_ref(v___y_3065_);
lean_dec(v___y_3064_);
lean_dec_ref(v___y_3063_);
return v_res_3068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_x_3069_, lean_object* v_x_3070_, lean_object* v_x_3071_, lean_object* v_x_3072_){
_start:
{
lean_object* v_ks_3073_; lean_object* v_vs_3074_; lean_object* v___x_3076_; uint8_t v_isShared_3077_; uint8_t v_isSharedCheck_3098_; 
v_ks_3073_ = lean_ctor_get(v_x_3069_, 0);
v_vs_3074_ = lean_ctor_get(v_x_3069_, 1);
v_isSharedCheck_3098_ = !lean_is_exclusive(v_x_3069_);
if (v_isSharedCheck_3098_ == 0)
{
v___x_3076_ = v_x_3069_;
v_isShared_3077_ = v_isSharedCheck_3098_;
goto v_resetjp_3075_;
}
else
{
lean_inc(v_vs_3074_);
lean_inc(v_ks_3073_);
lean_dec(v_x_3069_);
v___x_3076_ = lean_box(0);
v_isShared_3077_ = v_isSharedCheck_3098_;
goto v_resetjp_3075_;
}
v_resetjp_3075_:
{
lean_object* v___x_3078_; uint8_t v___x_3079_; 
v___x_3078_ = lean_array_get_size(v_ks_3073_);
v___x_3079_ = lean_nat_dec_lt(v_x_3070_, v___x_3078_);
if (v___x_3079_ == 0)
{
lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3083_; 
lean_dec(v_x_3070_);
v___x_3080_ = lean_array_push(v_ks_3073_, v_x_3071_);
v___x_3081_ = lean_array_push(v_vs_3074_, v_x_3072_);
if (v_isShared_3077_ == 0)
{
lean_ctor_set(v___x_3076_, 1, v___x_3081_);
lean_ctor_set(v___x_3076_, 0, v___x_3080_);
v___x_3083_ = v___x_3076_;
goto v_reusejp_3082_;
}
else
{
lean_object* v_reuseFailAlloc_3084_; 
v_reuseFailAlloc_3084_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3084_, 0, v___x_3080_);
lean_ctor_set(v_reuseFailAlloc_3084_, 1, v___x_3081_);
v___x_3083_ = v_reuseFailAlloc_3084_;
goto v_reusejp_3082_;
}
v_reusejp_3082_:
{
return v___x_3083_;
}
}
else
{
lean_object* v_k_x27_3085_; uint8_t v___x_3086_; 
v_k_x27_3085_ = lean_array_fget_borrowed(v_ks_3073_, v_x_3070_);
v___x_3086_ = l_Lean_instBEqMVarId_beq(v_x_3071_, v_k_x27_3085_);
if (v___x_3086_ == 0)
{
lean_object* v___x_3088_; 
if (v_isShared_3077_ == 0)
{
v___x_3088_ = v___x_3076_;
goto v_reusejp_3087_;
}
else
{
lean_object* v_reuseFailAlloc_3092_; 
v_reuseFailAlloc_3092_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3092_, 0, v_ks_3073_);
lean_ctor_set(v_reuseFailAlloc_3092_, 1, v_vs_3074_);
v___x_3088_ = v_reuseFailAlloc_3092_;
goto v_reusejp_3087_;
}
v_reusejp_3087_:
{
lean_object* v___x_3089_; lean_object* v___x_3090_; 
v___x_3089_ = lean_unsigned_to_nat(1u);
v___x_3090_ = lean_nat_add(v_x_3070_, v___x_3089_);
lean_dec(v_x_3070_);
v_x_3069_ = v___x_3088_;
v_x_3070_ = v___x_3090_;
goto _start;
}
}
else
{
lean_object* v___x_3093_; lean_object* v___x_3094_; lean_object* v___x_3096_; 
v___x_3093_ = lean_array_fset(v_ks_3073_, v_x_3070_, v_x_3071_);
v___x_3094_ = lean_array_fset(v_vs_3074_, v_x_3070_, v_x_3072_);
lean_dec(v_x_3070_);
if (v_isShared_3077_ == 0)
{
lean_ctor_set(v___x_3076_, 1, v___x_3094_);
lean_ctor_set(v___x_3076_, 0, v___x_3093_);
v___x_3096_ = v___x_3076_;
goto v_reusejp_3095_;
}
else
{
lean_object* v_reuseFailAlloc_3097_; 
v_reuseFailAlloc_3097_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3097_, 0, v___x_3093_);
lean_ctor_set(v_reuseFailAlloc_3097_, 1, v___x_3094_);
v___x_3096_ = v_reuseFailAlloc_3097_;
goto v_reusejp_3095_;
}
v_reusejp_3095_:
{
return v___x_3096_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4___redArg(lean_object* v_n_3099_, lean_object* v_k_3100_, lean_object* v_v_3101_){
_start:
{
lean_object* v___x_3102_; lean_object* v___x_3103_; 
v___x_3102_ = lean_unsigned_to_nat(0u);
v___x_3103_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_n_3099_, v___x_3102_, v_k_3100_, v_v_3101_);
return v___x_3103_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_3104_; 
v___x_3104_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_3104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(lean_object* v_x_3105_, size_t v_x_3106_, size_t v_x_3107_, lean_object* v_x_3108_, lean_object* v_x_3109_){
_start:
{
if (lean_obj_tag(v_x_3105_) == 0)
{
lean_object* v_es_3110_; size_t v___x_3111_; size_t v___x_3112_; lean_object* v_j_3113_; lean_object* v___x_3114_; uint8_t v___x_3115_; 
v_es_3110_ = lean_ctor_get(v_x_3105_, 0);
v___x_3111_ = ((size_t)31ULL);
v___x_3112_ = lean_usize_land(v_x_3106_, v___x_3111_);
v_j_3113_ = lean_usize_to_nat(v___x_3112_);
v___x_3114_ = lean_array_get_size(v_es_3110_);
v___x_3115_ = lean_nat_dec_lt(v_j_3113_, v___x_3114_);
if (v___x_3115_ == 0)
{
lean_dec(v_j_3113_);
lean_dec(v_x_3109_);
lean_dec(v_x_3108_);
return v_x_3105_;
}
else
{
lean_object* v___x_3117_; uint8_t v_isShared_3118_; uint8_t v_isSharedCheck_3154_; 
lean_inc_ref(v_es_3110_);
v_isSharedCheck_3154_ = !lean_is_exclusive(v_x_3105_);
if (v_isSharedCheck_3154_ == 0)
{
lean_object* v_unused_3155_; 
v_unused_3155_ = lean_ctor_get(v_x_3105_, 0);
lean_dec(v_unused_3155_);
v___x_3117_ = v_x_3105_;
v_isShared_3118_ = v_isSharedCheck_3154_;
goto v_resetjp_3116_;
}
else
{
lean_dec(v_x_3105_);
v___x_3117_ = lean_box(0);
v_isShared_3118_ = v_isSharedCheck_3154_;
goto v_resetjp_3116_;
}
v_resetjp_3116_:
{
lean_object* v_v_3119_; lean_object* v___x_3120_; lean_object* v_xs_x27_3121_; lean_object* v___y_3123_; 
v_v_3119_ = lean_array_fget(v_es_3110_, v_j_3113_);
v___x_3120_ = lean_box(0);
v_xs_x27_3121_ = lean_array_fset(v_es_3110_, v_j_3113_, v___x_3120_);
switch(lean_obj_tag(v_v_3119_))
{
case 0:
{
lean_object* v_key_3128_; lean_object* v_val_3129_; lean_object* v___x_3131_; uint8_t v_isShared_3132_; uint8_t v_isSharedCheck_3139_; 
v_key_3128_ = lean_ctor_get(v_v_3119_, 0);
v_val_3129_ = lean_ctor_get(v_v_3119_, 1);
v_isSharedCheck_3139_ = !lean_is_exclusive(v_v_3119_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3131_ = v_v_3119_;
v_isShared_3132_ = v_isSharedCheck_3139_;
goto v_resetjp_3130_;
}
else
{
lean_inc(v_val_3129_);
lean_inc(v_key_3128_);
lean_dec(v_v_3119_);
v___x_3131_ = lean_box(0);
v_isShared_3132_ = v_isSharedCheck_3139_;
goto v_resetjp_3130_;
}
v_resetjp_3130_:
{
uint8_t v___x_3133_; 
v___x_3133_ = l_Lean_instBEqMVarId_beq(v_x_3108_, v_key_3128_);
if (v___x_3133_ == 0)
{
lean_object* v___x_3134_; lean_object* v___x_3135_; 
lean_del_object(v___x_3131_);
v___x_3134_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_3128_, v_val_3129_, v_x_3108_, v_x_3109_);
v___x_3135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3135_, 0, v___x_3134_);
v___y_3123_ = v___x_3135_;
goto v___jp_3122_;
}
else
{
lean_object* v___x_3137_; 
lean_dec(v_val_3129_);
lean_dec(v_key_3128_);
if (v_isShared_3132_ == 0)
{
lean_ctor_set(v___x_3131_, 1, v_x_3109_);
lean_ctor_set(v___x_3131_, 0, v_x_3108_);
v___x_3137_ = v___x_3131_;
goto v_reusejp_3136_;
}
else
{
lean_object* v_reuseFailAlloc_3138_; 
v_reuseFailAlloc_3138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3138_, 0, v_x_3108_);
lean_ctor_set(v_reuseFailAlloc_3138_, 1, v_x_3109_);
v___x_3137_ = v_reuseFailAlloc_3138_;
goto v_reusejp_3136_;
}
v_reusejp_3136_:
{
v___y_3123_ = v___x_3137_;
goto v___jp_3122_;
}
}
}
}
case 1:
{
lean_object* v_node_3140_; lean_object* v___x_3142_; uint8_t v_isShared_3143_; uint8_t v_isSharedCheck_3152_; 
v_node_3140_ = lean_ctor_get(v_v_3119_, 0);
v_isSharedCheck_3152_ = !lean_is_exclusive(v_v_3119_);
if (v_isSharedCheck_3152_ == 0)
{
v___x_3142_ = v_v_3119_;
v_isShared_3143_ = v_isSharedCheck_3152_;
goto v_resetjp_3141_;
}
else
{
lean_inc(v_node_3140_);
lean_dec(v_v_3119_);
v___x_3142_ = lean_box(0);
v_isShared_3143_ = v_isSharedCheck_3152_;
goto v_resetjp_3141_;
}
v_resetjp_3141_:
{
size_t v___x_3144_; size_t v___x_3145_; size_t v___x_3146_; size_t v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3150_; 
v___x_3144_ = ((size_t)5ULL);
v___x_3145_ = lean_usize_shift_right(v_x_3106_, v___x_3144_);
v___x_3146_ = ((size_t)1ULL);
v___x_3147_ = lean_usize_add(v_x_3107_, v___x_3146_);
v___x_3148_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(v_node_3140_, v___x_3145_, v___x_3147_, v_x_3108_, v_x_3109_);
if (v_isShared_3143_ == 0)
{
lean_ctor_set(v___x_3142_, 0, v___x_3148_);
v___x_3150_ = v___x_3142_;
goto v_reusejp_3149_;
}
else
{
lean_object* v_reuseFailAlloc_3151_; 
v_reuseFailAlloc_3151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3151_, 0, v___x_3148_);
v___x_3150_ = v_reuseFailAlloc_3151_;
goto v_reusejp_3149_;
}
v_reusejp_3149_:
{
v___y_3123_ = v___x_3150_;
goto v___jp_3122_;
}
}
}
default: 
{
lean_object* v___x_3153_; 
v___x_3153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3153_, 0, v_x_3108_);
lean_ctor_set(v___x_3153_, 1, v_x_3109_);
v___y_3123_ = v___x_3153_;
goto v___jp_3122_;
}
}
v___jp_3122_:
{
lean_object* v___x_3124_; lean_object* v___x_3126_; 
v___x_3124_ = lean_array_fset(v_xs_x27_3121_, v_j_3113_, v___y_3123_);
lean_dec(v_j_3113_);
if (v_isShared_3118_ == 0)
{
lean_ctor_set(v___x_3117_, 0, v___x_3124_);
v___x_3126_ = v___x_3117_;
goto v_reusejp_3125_;
}
else
{
lean_object* v_reuseFailAlloc_3127_; 
v_reuseFailAlloc_3127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3127_, 0, v___x_3124_);
v___x_3126_ = v_reuseFailAlloc_3127_;
goto v_reusejp_3125_;
}
v_reusejp_3125_:
{
return v___x_3126_;
}
}
}
}
}
else
{
lean_object* v_ks_3156_; lean_object* v_vs_3157_; lean_object* v___x_3159_; uint8_t v_isShared_3160_; uint8_t v_isSharedCheck_3177_; 
v_ks_3156_ = lean_ctor_get(v_x_3105_, 0);
v_vs_3157_ = lean_ctor_get(v_x_3105_, 1);
v_isSharedCheck_3177_ = !lean_is_exclusive(v_x_3105_);
if (v_isSharedCheck_3177_ == 0)
{
v___x_3159_ = v_x_3105_;
v_isShared_3160_ = v_isSharedCheck_3177_;
goto v_resetjp_3158_;
}
else
{
lean_inc(v_vs_3157_);
lean_inc(v_ks_3156_);
lean_dec(v_x_3105_);
v___x_3159_ = lean_box(0);
v_isShared_3160_ = v_isSharedCheck_3177_;
goto v_resetjp_3158_;
}
v_resetjp_3158_:
{
lean_object* v___x_3162_; 
if (v_isShared_3160_ == 0)
{
v___x_3162_ = v___x_3159_;
goto v_reusejp_3161_;
}
else
{
lean_object* v_reuseFailAlloc_3176_; 
v_reuseFailAlloc_3176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3176_, 0, v_ks_3156_);
lean_ctor_set(v_reuseFailAlloc_3176_, 1, v_vs_3157_);
v___x_3162_ = v_reuseFailAlloc_3176_;
goto v_reusejp_3161_;
}
v_reusejp_3161_:
{
lean_object* v_newNode_3163_; uint8_t v___y_3165_; size_t v___x_3171_; uint8_t v___x_3172_; 
v_newNode_3163_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4___redArg(v___x_3162_, v_x_3108_, v_x_3109_);
v___x_3171_ = ((size_t)7ULL);
v___x_3172_ = lean_usize_dec_le(v___x_3171_, v_x_3107_);
if (v___x_3172_ == 0)
{
lean_object* v___x_3173_; lean_object* v___x_3174_; uint8_t v___x_3175_; 
v___x_3173_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_3163_);
v___x_3174_ = lean_unsigned_to_nat(4u);
v___x_3175_ = lean_nat_dec_lt(v___x_3173_, v___x_3174_);
lean_dec(v___x_3173_);
v___y_3165_ = v___x_3175_;
goto v___jp_3164_;
}
else
{
v___y_3165_ = v___x_3172_;
goto v___jp_3164_;
}
v___jp_3164_:
{
if (v___y_3165_ == 0)
{
lean_object* v_ks_3166_; lean_object* v_vs_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; 
v_ks_3166_ = lean_ctor_get(v_newNode_3163_, 0);
lean_inc_ref(v_ks_3166_);
v_vs_3167_ = lean_ctor_get(v_newNode_3163_, 1);
lean_inc_ref(v_vs_3167_);
lean_dec_ref(v_newNode_3163_);
v___x_3168_ = lean_unsigned_to_nat(0u);
v___x_3169_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_3170_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg(v_x_3107_, v_ks_3166_, v_vs_3167_, v___x_3168_, v___x_3169_);
lean_dec_ref(v_vs_3167_);
lean_dec_ref(v_ks_3166_);
return v___x_3170_;
}
else
{
return v_newNode_3163_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg(size_t v_depth_3178_, lean_object* v_keys_3179_, lean_object* v_vals_3180_, lean_object* v_i_3181_, lean_object* v_entries_3182_){
_start:
{
lean_object* v___x_3183_; uint8_t v___x_3184_; 
v___x_3183_ = lean_array_get_size(v_keys_3179_);
v___x_3184_ = lean_nat_dec_lt(v_i_3181_, v___x_3183_);
if (v___x_3184_ == 0)
{
lean_dec(v_i_3181_);
return v_entries_3182_;
}
else
{
lean_object* v_k_3185_; lean_object* v_v_3186_; uint64_t v___x_3187_; size_t v_h_3188_; size_t v___x_3189_; lean_object* v___x_3190_; size_t v___x_3191_; size_t v___x_3192_; size_t v___x_3193_; size_t v_h_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; 
v_k_3185_ = lean_array_fget_borrowed(v_keys_3179_, v_i_3181_);
v_v_3186_ = lean_array_fget_borrowed(v_vals_3180_, v_i_3181_);
v___x_3187_ = l_Lean_instHashableMVarId_hash(v_k_3185_);
v_h_3188_ = lean_uint64_to_usize(v___x_3187_);
v___x_3189_ = ((size_t)5ULL);
v___x_3190_ = lean_unsigned_to_nat(1u);
v___x_3191_ = ((size_t)1ULL);
v___x_3192_ = lean_usize_sub(v_depth_3178_, v___x_3191_);
v___x_3193_ = lean_usize_mul(v___x_3189_, v___x_3192_);
v_h_3194_ = lean_usize_shift_right(v_h_3188_, v___x_3193_);
v___x_3195_ = lean_nat_add(v_i_3181_, v___x_3190_);
lean_dec(v_i_3181_);
lean_inc(v_v_3186_);
lean_inc(v_k_3185_);
v___x_3196_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(v_entries_3182_, v_h_3194_, v_depth_3178_, v_k_3185_, v_v_3186_);
v_i_3181_ = v___x_3195_;
v_entries_3182_ = v___x_3196_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_depth_3198_, lean_object* v_keys_3199_, lean_object* v_vals_3200_, lean_object* v_i_3201_, lean_object* v_entries_3202_){
_start:
{
size_t v_depth_boxed_3203_; lean_object* v_res_3204_; 
v_depth_boxed_3203_ = lean_unbox_usize(v_depth_3198_);
lean_dec(v_depth_3198_);
v_res_3204_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_boxed_3203_, v_keys_3199_, v_vals_3200_, v_i_3201_, v_entries_3202_);
lean_dec_ref(v_vals_3200_);
lean_dec_ref(v_keys_3199_);
return v_res_3204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_x_3205_, lean_object* v_x_3206_, lean_object* v_x_3207_, lean_object* v_x_3208_, lean_object* v_x_3209_){
_start:
{
size_t v_x_4469__boxed_3210_; size_t v_x_4470__boxed_3211_; lean_object* v_res_3212_; 
v_x_4469__boxed_3210_ = lean_unbox_usize(v_x_3206_);
lean_dec(v_x_3206_);
v_x_4470__boxed_3211_ = lean_unbox_usize(v_x_3207_);
lean_dec(v_x_3207_);
v_res_3212_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(v_x_3205_, v_x_4469__boxed_3210_, v_x_4470__boxed_3211_, v_x_3208_, v_x_3209_);
return v_res_3212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1___redArg(lean_object* v_x_3213_, lean_object* v_x_3214_, lean_object* v_x_3215_){
_start:
{
uint64_t v___x_3216_; size_t v___x_3217_; size_t v___x_3218_; lean_object* v___x_3219_; 
v___x_3216_ = l_Lean_instHashableMVarId_hash(v_x_3214_);
v___x_3217_ = lean_uint64_to_usize(v___x_3216_);
v___x_3218_ = ((size_t)1ULL);
v___x_3219_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(v_x_3213_, v___x_3217_, v___x_3218_, v_x_3214_, v_x_3215_);
return v___x_3219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg(lean_object* v_mvarId_3220_, lean_object* v_val_3221_, lean_object* v___y_3222_){
_start:
{
lean_object* v___x_3224_; lean_object* v_mctx_3225_; lean_object* v_cache_3226_; lean_object* v_zetaDeltaFVarIds_3227_; lean_object* v_postponed_3228_; lean_object* v_diag_3229_; lean_object* v___x_3231_; uint8_t v_isShared_3232_; uint8_t v_isSharedCheck_3257_; 
v___x_3224_ = lean_st_ref_take(v___y_3222_);
v_mctx_3225_ = lean_ctor_get(v___x_3224_, 0);
v_cache_3226_ = lean_ctor_get(v___x_3224_, 1);
v_zetaDeltaFVarIds_3227_ = lean_ctor_get(v___x_3224_, 2);
v_postponed_3228_ = lean_ctor_get(v___x_3224_, 3);
v_diag_3229_ = lean_ctor_get(v___x_3224_, 4);
v_isSharedCheck_3257_ = !lean_is_exclusive(v___x_3224_);
if (v_isSharedCheck_3257_ == 0)
{
v___x_3231_ = v___x_3224_;
v_isShared_3232_ = v_isSharedCheck_3257_;
goto v_resetjp_3230_;
}
else
{
lean_inc(v_diag_3229_);
lean_inc(v_postponed_3228_);
lean_inc(v_zetaDeltaFVarIds_3227_);
lean_inc(v_cache_3226_);
lean_inc(v_mctx_3225_);
lean_dec(v___x_3224_);
v___x_3231_ = lean_box(0);
v_isShared_3232_ = v_isSharedCheck_3257_;
goto v_resetjp_3230_;
}
v_resetjp_3230_:
{
lean_object* v_depth_3233_; lean_object* v_levelAssignDepth_3234_; lean_object* v_lmvarCounter_3235_; lean_object* v_mvarCounter_3236_; lean_object* v_lDecls_3237_; lean_object* v_decls_3238_; lean_object* v_userNames_3239_; lean_object* v_lAssignment_3240_; lean_object* v_eAssignment_3241_; lean_object* v_dAssignment_3242_; lean_object* v___x_3244_; uint8_t v_isShared_3245_; uint8_t v_isSharedCheck_3256_; 
v_depth_3233_ = lean_ctor_get(v_mctx_3225_, 0);
v_levelAssignDepth_3234_ = lean_ctor_get(v_mctx_3225_, 1);
v_lmvarCounter_3235_ = lean_ctor_get(v_mctx_3225_, 2);
v_mvarCounter_3236_ = lean_ctor_get(v_mctx_3225_, 3);
v_lDecls_3237_ = lean_ctor_get(v_mctx_3225_, 4);
v_decls_3238_ = lean_ctor_get(v_mctx_3225_, 5);
v_userNames_3239_ = lean_ctor_get(v_mctx_3225_, 6);
v_lAssignment_3240_ = lean_ctor_get(v_mctx_3225_, 7);
v_eAssignment_3241_ = lean_ctor_get(v_mctx_3225_, 8);
v_dAssignment_3242_ = lean_ctor_get(v_mctx_3225_, 9);
v_isSharedCheck_3256_ = !lean_is_exclusive(v_mctx_3225_);
if (v_isSharedCheck_3256_ == 0)
{
v___x_3244_ = v_mctx_3225_;
v_isShared_3245_ = v_isSharedCheck_3256_;
goto v_resetjp_3243_;
}
else
{
lean_inc(v_dAssignment_3242_);
lean_inc(v_eAssignment_3241_);
lean_inc(v_lAssignment_3240_);
lean_inc(v_userNames_3239_);
lean_inc(v_decls_3238_);
lean_inc(v_lDecls_3237_);
lean_inc(v_mvarCounter_3236_);
lean_inc(v_lmvarCounter_3235_);
lean_inc(v_levelAssignDepth_3234_);
lean_inc(v_depth_3233_);
lean_dec(v_mctx_3225_);
v___x_3244_ = lean_box(0);
v_isShared_3245_ = v_isSharedCheck_3256_;
goto v_resetjp_3243_;
}
v_resetjp_3243_:
{
lean_object* v___x_3246_; lean_object* v___x_3248_; 
v___x_3246_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1___redArg(v_eAssignment_3241_, v_mvarId_3220_, v_val_3221_);
if (v_isShared_3245_ == 0)
{
lean_ctor_set(v___x_3244_, 8, v___x_3246_);
v___x_3248_ = v___x_3244_;
goto v_reusejp_3247_;
}
else
{
lean_object* v_reuseFailAlloc_3255_; 
v_reuseFailAlloc_3255_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3255_, 0, v_depth_3233_);
lean_ctor_set(v_reuseFailAlloc_3255_, 1, v_levelAssignDepth_3234_);
lean_ctor_set(v_reuseFailAlloc_3255_, 2, v_lmvarCounter_3235_);
lean_ctor_set(v_reuseFailAlloc_3255_, 3, v_mvarCounter_3236_);
lean_ctor_set(v_reuseFailAlloc_3255_, 4, v_lDecls_3237_);
lean_ctor_set(v_reuseFailAlloc_3255_, 5, v_decls_3238_);
lean_ctor_set(v_reuseFailAlloc_3255_, 6, v_userNames_3239_);
lean_ctor_set(v_reuseFailAlloc_3255_, 7, v_lAssignment_3240_);
lean_ctor_set(v_reuseFailAlloc_3255_, 8, v___x_3246_);
lean_ctor_set(v_reuseFailAlloc_3255_, 9, v_dAssignment_3242_);
v___x_3248_ = v_reuseFailAlloc_3255_;
goto v_reusejp_3247_;
}
v_reusejp_3247_:
{
lean_object* v___x_3250_; 
if (v_isShared_3232_ == 0)
{
lean_ctor_set(v___x_3231_, 0, v___x_3248_);
v___x_3250_ = v___x_3231_;
goto v_reusejp_3249_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v___x_3248_);
lean_ctor_set(v_reuseFailAlloc_3254_, 1, v_cache_3226_);
lean_ctor_set(v_reuseFailAlloc_3254_, 2, v_zetaDeltaFVarIds_3227_);
lean_ctor_set(v_reuseFailAlloc_3254_, 3, v_postponed_3228_);
lean_ctor_set(v_reuseFailAlloc_3254_, 4, v_diag_3229_);
v___x_3250_ = v_reuseFailAlloc_3254_;
goto v_reusejp_3249_;
}
v_reusejp_3249_:
{
lean_object* v___x_3251_; lean_object* v___x_3252_; lean_object* v___x_3253_; 
v___x_3251_ = lean_st_ref_set(v___y_3222_, v___x_3250_);
v___x_3252_ = lean_box(0);
v___x_3253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3253_, 0, v___x_3252_);
return v___x_3253_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg___boxed(lean_object* v_mvarId_3258_, lean_object* v_val_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_){
_start:
{
lean_object* v_res_3262_; 
v_res_3262_ = lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg(v_mvarId_3258_, v_val_3259_, v___y_3260_);
lean_dec(v___y_3260_);
return v_res_3262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(lean_object* v_fst_3263_, lean_object* v_leaves_3264_, lean_object* v___x_3265_, lean_object* v_path_3266_, lean_object* v_P_x27_3267_, lean_object* v_fvars_3268_, lean_object* v_snd_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_, lean_object* v___y_3272_, lean_object* v___y_3273_){
_start:
{
lean_object* v___x_3275_; 
v___x_3275_ = lp_mathlib_List_mapM_loop___at___00ExistsAndEq_mkBeforeToAfter_spec__0(v_fst_3263_, v_leaves_3264_, v___x_3265_, v___y_3270_, v___y_3271_, v___y_3272_, v___y_3273_);
if (lean_obj_tag(v___x_3275_) == 0)
{
lean_object* v_a_3276_; lean_object* v___x_3277_; lean_object* v___x_3278_; 
v_a_3276_ = lean_ctor_get(v___x_3275_, 0);
lean_inc(v_a_3276_);
lean_dec_ref_known(v___x_3275_, 1);
v___x_3277_ = lp_mathlib_ExistsAndEq_Path_forResult(v_path_3266_);
v___x_3278_ = lp_mathlib_ExistsAndEq_construct(v_P_x27_3267_, v_fvars_3268_, v___x_3277_, v_a_3276_, v___y_3270_, v___y_3271_, v___y_3272_, v___y_3273_);
if (lean_obj_tag(v___x_3278_) == 0)
{
lean_object* v_a_3279_; lean_object* v___x_3280_; 
v_a_3279_ = lean_ctor_get(v___x_3278_, 0);
lean_inc(v_a_3279_);
lean_dec_ref_known(v___x_3278_, 1);
v___x_3280_ = lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg(v_snd_3269_, v_a_3279_, v___y_3271_);
return v___x_3280_;
}
else
{
lean_object* v_a_3281_; lean_object* v___x_3283_; uint8_t v_isShared_3284_; uint8_t v_isSharedCheck_3288_; 
lean_dec(v_snd_3269_);
v_a_3281_ = lean_ctor_get(v___x_3278_, 0);
v_isSharedCheck_3288_ = !lean_is_exclusive(v___x_3278_);
if (v_isSharedCheck_3288_ == 0)
{
v___x_3283_ = v___x_3278_;
v_isShared_3284_ = v_isSharedCheck_3288_;
goto v_resetjp_3282_;
}
else
{
lean_inc(v_a_3281_);
lean_dec(v___x_3278_);
v___x_3283_ = lean_box(0);
v_isShared_3284_ = v_isSharedCheck_3288_;
goto v_resetjp_3282_;
}
v_resetjp_3282_:
{
lean_object* v___x_3286_; 
if (v_isShared_3284_ == 0)
{
v___x_3286_ = v___x_3283_;
goto v_reusejp_3285_;
}
else
{
lean_object* v_reuseFailAlloc_3287_; 
v_reuseFailAlloc_3287_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3287_, 0, v_a_3281_);
v___x_3286_ = v_reuseFailAlloc_3287_;
goto v_reusejp_3285_;
}
v_reusejp_3285_:
{
return v___x_3286_;
}
}
}
}
else
{
lean_object* v_a_3289_; lean_object* v___x_3291_; uint8_t v_isShared_3292_; uint8_t v_isSharedCheck_3296_; 
lean_dec(v_snd_3269_);
lean_dec(v_fvars_3268_);
lean_dec_ref(v_P_x27_3267_);
lean_dec(v_path_3266_);
v_a_3289_ = lean_ctor_get(v___x_3275_, 0);
v_isSharedCheck_3296_ = !lean_is_exclusive(v___x_3275_);
if (v_isSharedCheck_3296_ == 0)
{
v___x_3291_ = v___x_3275_;
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
else
{
lean_inc(v_a_3289_);
lean_dec(v___x_3275_);
v___x_3291_ = lean_box(0);
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
v_resetjp_3290_:
{
lean_object* v___x_3294_; 
if (v_isShared_3292_ == 0)
{
v___x_3294_ = v___x_3291_;
goto v_reusejp_3293_;
}
else
{
lean_object* v_reuseFailAlloc_3295_; 
v_reuseFailAlloc_3295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3295_, 0, v_a_3289_);
v___x_3294_ = v_reuseFailAlloc_3295_;
goto v_reusejp_3293_;
}
v_reusejp_3293_:
{
return v___x_3294_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed(lean_object* v_fst_3297_, lean_object* v_leaves_3298_, lean_object* v___x_3299_, lean_object* v_path_3300_, lean_object* v_P_x27_3301_, lean_object* v_fvars_3302_, lean_object* v_snd_3303_, lean_object* v___y_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_){
_start:
{
lean_object* v_res_3309_; 
v_res_3309_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1(v_fst_3297_, v_leaves_3298_, v___x_3299_, v_path_3300_, v_P_x27_3301_, v_fvars_3302_, v_snd_3303_, v___y_3304_, v___y_3305_, v___y_3306_, v___y_3307_);
lean_dec(v___y_3307_);
lean_dec_ref(v___y_3306_);
lean_dec(v___y_3305_);
lean_dec_ref(v___y_3304_);
return v_res_3309_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2(void){
_start:
{
lean_object* v___x_3312_; lean_object* v___x_3313_; lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v___x_3317_; 
v___x_3312_ = ((lean_object*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__1));
v___x_3313_ = lean_unsigned_to_nat(44u);
v___x_3314_ = lean_unsigned_to_nat(260u);
v___x_3315_ = ((lean_object*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__0));
v___x_3316_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_3317_ = l_mkPanicMessageWithDecl(v___x_3316_, v___x_3315_, v___x_3314_, v___x_3313_, v___x_3312_);
return v___x_3317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(lean_object* v___x_3318_, lean_object* v___x_3319_, lean_object* v___x_3320_, lean_object* v_P_x27_3321_, lean_object* v_path_3322_, lean_object* v_fvars_3323_, lean_object* v_a_3324_, lean_object* v_leaves_3325_, lean_object* v_x_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_){
_start:
{
lean_object* v_fst_3332_; lean_object* v_snd_3333_; lean_object* v___x_3334_; uint8_t v___x_3335_; lean_object* v___x_3336_; lean_object* v___f_3337_; uint8_t v___x_3338_; lean_object* v___x_3339_; 
v_fst_3332_ = lean_ctor_get(v_x_3326_, 0);
lean_inc(v_fst_3332_);
v_snd_3333_ = lean_ctor_get(v_x_3326_, 1);
lean_inc(v_snd_3333_);
lean_dec_ref(v_x_3326_);
v___x_3334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3334_, 0, v___x_3318_);
v___x_3335_ = 0;
v___x_3336_ = lean_box(v___x_3335_);
lean_inc(v___x_3319_);
v___f_3337_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__0___boxed), 10, 5);
lean_closure_set(v___f_3337_, 0, v___x_3334_);
lean_closure_set(v___f_3337_, 1, v___x_3336_);
lean_closure_set(v___f_3337_, 2, v___x_3319_);
lean_closure_set(v___f_3337_, 3, v___x_3320_);
lean_closure_set(v___f_3337_, 4, v_fst_3332_);
v___x_3338_ = 0;
v___x_3339_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__2___redArg(v___f_3337_, v___x_3338_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
if (lean_obj_tag(v___x_3339_) == 0)
{
lean_object* v_a_3340_; lean_object* v_snd_3341_; lean_object* v_snd_3342_; lean_object* v_snd_3343_; uint8_t v___x_3344_; 
v_a_3340_ = lean_ctor_get(v___x_3339_, 0);
lean_inc(v_a_3340_);
lean_dec_ref_known(v___x_3339_, 1);
v_snd_3341_ = lean_ctor_get(v_a_3340_, 1);
lean_inc(v_snd_3341_);
lean_dec(v_a_3340_);
v_snd_3342_ = lean_ctor_get(v_snd_3341_, 1);
v_snd_3343_ = lean_ctor_get(v_snd_3342_, 1);
lean_inc(v_snd_3343_);
v___x_3344_ = lean_unbox(v_snd_3343_);
if (v___x_3344_ == 0)
{
lean_object* v___x_3345_; lean_object* v___x_3346_; 
lean_dec(v_snd_3343_);
lean_dec(v_snd_3341_);
lean_dec(v_snd_3333_);
lean_dec(v_leaves_3325_);
lean_dec(v_fvars_3323_);
lean_dec(v_path_3322_);
lean_dec_ref(v_P_x27_3321_);
lean_dec(v___x_3319_);
v___x_3345_ = lean_obj_once(&lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2, &lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2_once, _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___closed__2);
v___x_3346_ = lp_mathlib_panic___at___00ExistsAndEq_destruct_spec__0(v___x_3345_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
return v___x_3346_;
}
else
{
lean_object* v_fst_3347_; lean_object* v___x_3348_; 
v_fst_3347_ = lean_ctor_get(v_snd_3341_, 0);
lean_inc(v_fst_3347_);
lean_dec(v_snd_3341_);
lean_inc_ref(v_P_x27_3321_);
v___x_3348_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_P_x27_3321_, v___x_3319_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
if (lean_obj_tag(v___x_3348_) == 0)
{
lean_object* v_a_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; uint8_t v___y_3353_; uint8_t v___x_3380_; 
v_a_3349_ = lean_ctor_get(v___x_3348_, 0);
lean_inc(v_a_3349_);
lean_dec_ref_known(v___x_3348_, 1);
v___x_3350_ = l_Lean_Expr_mvarId_x21(v_a_3349_);
v___x_3351_ = l_Lean_Expr_fvarId_x21(v_snd_3333_);
lean_dec(v_snd_3333_);
v___x_3380_ = lean_expr_eqv(v_fst_3347_, v_a_3324_);
lean_dec(v_fst_3347_);
if (v___x_3380_ == 0)
{
uint8_t v___x_3381_; 
v___x_3381_ = lean_unbox(v_snd_3343_);
v___y_3353_ = v___x_3381_;
goto v___jp_3352_;
}
else
{
v___y_3353_ = v___x_3338_;
goto v___jp_3352_;
}
v___jp_3352_:
{
lean_object* v___x_3354_; uint8_t v___x_3355_; lean_object* v___x_3356_; 
v___x_3354_ = lean_box(0);
v___x_3355_ = lean_unbox(v_snd_3343_);
lean_dec(v_snd_3343_);
v___x_3356_ = l_Lean_Meta_substCore(v___x_3350_, v___x_3351_, v___y_3353_, v___x_3354_, v___x_3355_, v___x_3338_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
if (lean_obj_tag(v___x_3356_) == 0)
{
lean_object* v_a_3357_; lean_object* v_fst_3358_; lean_object* v_snd_3359_; lean_object* v___x_3360_; lean_object* v___f_3361_; lean_object* v___x_3362_; 
v_a_3357_ = lean_ctor_get(v___x_3356_, 0);
lean_inc(v_a_3357_);
lean_dec_ref_known(v___x_3356_, 1);
v_fst_3358_ = lean_ctor_get(v_a_3357_, 0);
lean_inc(v_fst_3358_);
v_snd_3359_ = lean_ctor_get(v_a_3357_, 1);
lean_inc_n(v_snd_3359_, 2);
lean_dec(v_a_3357_);
v___x_3360_ = lean_box(0);
v___f_3361_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__1___boxed), 12, 7);
lean_closure_set(v___f_3361_, 0, v_fst_3358_);
lean_closure_set(v___f_3361_, 1, v_leaves_3325_);
lean_closure_set(v___f_3361_, 2, v___x_3360_);
lean_closure_set(v___f_3361_, 3, v_path_3322_);
lean_closure_set(v___f_3361_, 4, v_P_x27_3321_);
lean_closure_set(v___f_3361_, 5, v_fvars_3323_);
lean_closure_set(v___f_3361_, 6, v_snd_3359_);
v___x_3362_ = lp_mathlib_Lean_MVarId_withContext___at___00ExistsAndEq_mkBeforeToAfter_spec__2___redArg(v_snd_3359_, v___f_3361_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_);
if (lean_obj_tag(v___x_3362_) == 0)
{
lean_object* v___x_3363_; 
lean_dec_ref_known(v___x_3362_, 1);
v___x_3363_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_a_3349_, v___y_3328_);
return v___x_3363_;
}
else
{
lean_object* v_a_3364_; lean_object* v___x_3366_; uint8_t v_isShared_3367_; uint8_t v_isSharedCheck_3371_; 
lean_dec(v_a_3349_);
v_a_3364_ = lean_ctor_get(v___x_3362_, 0);
v_isSharedCheck_3371_ = !lean_is_exclusive(v___x_3362_);
if (v_isSharedCheck_3371_ == 0)
{
v___x_3366_ = v___x_3362_;
v_isShared_3367_ = v_isSharedCheck_3371_;
goto v_resetjp_3365_;
}
else
{
lean_inc(v_a_3364_);
lean_dec(v___x_3362_);
v___x_3366_ = lean_box(0);
v_isShared_3367_ = v_isSharedCheck_3371_;
goto v_resetjp_3365_;
}
v_resetjp_3365_:
{
lean_object* v___x_3369_; 
if (v_isShared_3367_ == 0)
{
v___x_3369_ = v___x_3366_;
goto v_reusejp_3368_;
}
else
{
lean_object* v_reuseFailAlloc_3370_; 
v_reuseFailAlloc_3370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3370_, 0, v_a_3364_);
v___x_3369_ = v_reuseFailAlloc_3370_;
goto v_reusejp_3368_;
}
v_reusejp_3368_:
{
return v___x_3369_;
}
}
}
}
else
{
lean_object* v_a_3372_; lean_object* v___x_3374_; uint8_t v_isShared_3375_; uint8_t v_isSharedCheck_3379_; 
lean_dec(v_a_3349_);
lean_dec(v_leaves_3325_);
lean_dec(v_fvars_3323_);
lean_dec(v_path_3322_);
lean_dec_ref(v_P_x27_3321_);
v_a_3372_ = lean_ctor_get(v___x_3356_, 0);
v_isSharedCheck_3379_ = !lean_is_exclusive(v___x_3356_);
if (v_isSharedCheck_3379_ == 0)
{
v___x_3374_ = v___x_3356_;
v_isShared_3375_ = v_isSharedCheck_3379_;
goto v_resetjp_3373_;
}
else
{
lean_inc(v_a_3372_);
lean_dec(v___x_3356_);
v___x_3374_ = lean_box(0);
v_isShared_3375_ = v_isSharedCheck_3379_;
goto v_resetjp_3373_;
}
v_resetjp_3373_:
{
lean_object* v___x_3377_; 
if (v_isShared_3375_ == 0)
{
v___x_3377_ = v___x_3374_;
goto v_reusejp_3376_;
}
else
{
lean_object* v_reuseFailAlloc_3378_; 
v_reuseFailAlloc_3378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3378_, 0, v_a_3372_);
v___x_3377_ = v_reuseFailAlloc_3378_;
goto v_reusejp_3376_;
}
v_reusejp_3376_:
{
return v___x_3377_;
}
}
}
}
}
else
{
lean_dec(v_fst_3347_);
lean_dec(v_snd_3343_);
lean_dec(v_snd_3333_);
lean_dec(v_leaves_3325_);
lean_dec(v_fvars_3323_);
lean_dec(v_path_3322_);
lean_dec_ref(v_P_x27_3321_);
return v___x_3348_;
}
}
}
else
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3389_; 
lean_dec(v_snd_3333_);
lean_dec(v_leaves_3325_);
lean_dec(v_fvars_3323_);
lean_dec(v_path_3322_);
lean_dec_ref(v_P_x27_3321_);
lean_dec(v___x_3319_);
v_a_3382_ = lean_ctor_get(v___x_3339_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3339_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3384_ = v___x_3339_;
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3339_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
lean_object* v___x_3387_; 
if (v_isShared_3385_ == 0)
{
v___x_3387_ = v___x_3384_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v_a_3382_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed(lean_object* v___x_3390_, lean_object* v___x_3391_, lean_object* v___x_3392_, lean_object* v_P_x27_3393_, lean_object* v_path_3394_, lean_object* v_fvars_3395_, lean_object* v_a_3396_, lean_object* v_leaves_3397_, lean_object* v_x_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_, lean_object* v___y_3403_){
_start:
{
lean_object* v_res_3404_; 
v_res_3404_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2(v___x_3390_, v___x_3391_, v___x_3392_, v_P_x27_3393_, v_path_3394_, v_fvars_3395_, v_a_3396_, v_leaves_3397_, v_x_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v___y_3402_);
lean_dec(v___y_3402_);
lean_dec_ref(v___y_3401_);
lean_dec(v___y_3400_);
lean_dec_ref(v___y_3399_);
lean_dec_ref(v_a_3396_);
return v_res_3404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(lean_object* v___x_3405_, lean_object* v_P_x27_3406_, lean_object* v_fvars_3407_, lean_object* v_path_3408_, lean_object* v___f_3409_, lean_object* v_a_3410_, uint8_t v___x_3411_, lean_object* v___x_3412_, lean_object* v___x_3413_, lean_object* v_00_u03b1_3414_, lean_object* v_p_3415_, lean_object* v_h_3416_, lean_object* v_ha_3417_, lean_object* v___y_3418_, lean_object* v___y_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_){
_start:
{
lean_object* v___x_3423_; lean_object* v___x_3424_; 
v___x_3423_ = lean_box(0);
lean_inc_ref(v_ha_3417_);
lean_inc_ref(v_P_x27_3406_);
v___x_3424_ = lp_mathlib_ExistsAndEq_destruct(v___x_3405_, v_P_x27_3406_, v_ha_3417_, v_fvars_3407_, v_path_3408_, v___x_3423_, v___f_3409_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
if (lean_obj_tag(v___x_3424_) == 0)
{
lean_object* v_a_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; uint8_t v___x_3430_; uint8_t v___x_3431_; lean_object* v___x_3432_; 
v_a_3425_ = lean_ctor_get(v___x_3424_, 0);
lean_inc(v_a_3425_);
lean_dec_ref_known(v___x_3424_, 1);
v___x_3426_ = lean_unsigned_to_nat(2u);
v___x_3427_ = lean_mk_empty_array_with_capacity(v___x_3426_);
v___x_3428_ = lean_array_push(v___x_3427_, v_a_3410_);
v___x_3429_ = lean_array_push(v___x_3428_, v_ha_3417_);
v___x_3430_ = 1;
v___x_3431_ = 1;
v___x_3432_ = l_Lean_Meta_mkLambdaFVars(v___x_3429_, v_a_3425_, v___x_3411_, v___x_3430_, v___x_3411_, v___x_3430_, v___x_3431_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
lean_dec_ref(v___x_3429_);
if (lean_obj_tag(v___x_3432_) == 0)
{
lean_object* v_a_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___x_3441_; lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3444_; lean_object* v___x_3445_; 
v_a_3433_ = lean_ctor_get(v___x_3432_, 0);
lean_inc(v_a_3433_);
lean_dec_ref_known(v___x_3432_, 1);
v___x_3434_ = ((lean_object*)(lp_mathlib_ExistsAndEq_destruct___lam__5___closed__0));
v___x_3435_ = l_Lean_Name_mkStr2(v___x_3412_, v___x_3434_);
v___x_3436_ = l_Lean_Expr_const___override(v___x_3435_, v___x_3413_);
v___x_3437_ = l_Lean_Expr_app___override(v___x_3436_, v_00_u03b1_3414_);
v___x_3438_ = l_Lean_Expr_app___override(v___x_3437_, v_p_3415_);
v___x_3439_ = l_Lean_Expr_app___override(v___x_3438_, v_P_x27_3406_);
lean_inc_ref(v_h_3416_);
v___x_3440_ = l_Lean_Expr_app___override(v___x_3439_, v_h_3416_);
v___x_3441_ = l_Lean_Expr_app___override(v___x_3440_, v_a_3433_);
v___x_3442_ = lean_unsigned_to_nat(1u);
v___x_3443_ = lean_mk_empty_array_with_capacity(v___x_3442_);
v___x_3444_ = lean_array_push(v___x_3443_, v_h_3416_);
v___x_3445_ = l_Lean_Meta_mkLambdaFVars(v___x_3444_, v___x_3441_, v___x_3411_, v___x_3430_, v___x_3411_, v___x_3430_, v___x_3431_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
lean_dec_ref(v___x_3444_);
return v___x_3445_;
}
else
{
lean_dec_ref(v_h_3416_);
lean_dec_ref(v_p_3415_);
lean_dec_ref(v_00_u03b1_3414_);
lean_dec(v___x_3413_);
lean_dec_ref(v___x_3412_);
lean_dec_ref(v_P_x27_3406_);
return v___x_3432_;
}
}
else
{
lean_dec_ref(v_ha_3417_);
lean_dec_ref(v_h_3416_);
lean_dec_ref(v_p_3415_);
lean_dec_ref(v_00_u03b1_3414_);
lean_dec(v___x_3413_);
lean_dec_ref(v___x_3412_);
lean_dec_ref(v_a_3410_);
lean_dec_ref(v_P_x27_3406_);
return v___x_3424_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed(lean_object** _args){
lean_object* v___x_3446_ = _args[0];
lean_object* v_P_x27_3447_ = _args[1];
lean_object* v_fvars_3448_ = _args[2];
lean_object* v_path_3449_ = _args[3];
lean_object* v___f_3450_ = _args[4];
lean_object* v_a_3451_ = _args[5];
lean_object* v___x_3452_ = _args[6];
lean_object* v___x_3453_ = _args[7];
lean_object* v___x_3454_ = _args[8];
lean_object* v_00_u03b1_3455_ = _args[9];
lean_object* v_p_3456_ = _args[10];
lean_object* v_h_3457_ = _args[11];
lean_object* v_ha_3458_ = _args[12];
lean_object* v___y_3459_ = _args[13];
lean_object* v___y_3460_ = _args[14];
lean_object* v___y_3461_ = _args[15];
lean_object* v___y_3462_ = _args[16];
lean_object* v___y_3463_ = _args[17];
_start:
{
uint8_t v___x_4915__boxed_3464_; lean_object* v_res_3465_; 
v___x_4915__boxed_3464_ = lean_unbox(v___x_3452_);
v_res_3465_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3(v___x_3446_, v_P_x27_3447_, v_fvars_3448_, v_path_3449_, v___f_3450_, v_a_3451_, v___x_4915__boxed_3464_, v___x_3453_, v___x_3454_, v_00_u03b1_3455_, v_p_3456_, v_h_3457_, v_ha_3458_, v___y_3459_, v___y_3460_, v___y_3461_, v___y_3462_);
lean_dec(v___y_3462_);
lean_dec_ref(v___y_3461_);
lean_dec(v___y_3460_);
lean_dec_ref(v___y_3459_);
return v_res_3465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(lean_object* v___x_3466_, lean_object* v___x_3467_, lean_object* v___x_3468_, lean_object* v_P_x27_3469_, lean_object* v_path_3470_, lean_object* v_fvars_3471_, lean_object* v___x_3472_, lean_object* v_p_3473_, lean_object* v___x_3474_, lean_object* v_00_u03b1_3475_, lean_object* v_h_3476_, uint8_t v___x_3477_, lean_object* v_a_3478_, lean_object* v___y_3479_, lean_object* v___y_3480_, lean_object* v___y_3481_, lean_object* v___y_3482_){
_start:
{
lean_object* v___f_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; uint8_t v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v___f_3490_; lean_object* v___x_3491_; 
lean_inc_ref_n(v_a_3478_, 2);
lean_inc(v_fvars_3471_);
lean_inc(v_path_3470_);
lean_inc_ref(v_P_x27_3469_);
lean_inc(v___x_3468_);
lean_inc(v___x_3467_);
v___f_3484_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__2___boxed), 14, 7);
lean_closure_set(v___f_3484_, 0, v___x_3466_);
lean_closure_set(v___f_3484_, 1, v___x_3467_);
lean_closure_set(v___f_3484_, 2, v___x_3468_);
lean_closure_set(v___f_3484_, 3, v_P_x27_3469_);
lean_closure_set(v___f_3484_, 4, v_path_3470_);
lean_closure_set(v___f_3484_, 5, v_fvars_3471_);
lean_closure_set(v___f_3484_, 6, v_a_3478_);
v___x_3485_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3485_, 0, v_a_3478_);
lean_ctor_set(v___x_3485_, 1, v___x_3472_);
v___x_3486_ = lean_array_mk(v___x_3485_);
v___x_3487_ = 0;
lean_inc_ref(v_p_3473_);
v___x_3488_ = l_Lean_Expr_betaRev(v_p_3473_, v___x_3486_, v___x_3487_, v___x_3487_);
lean_dec_ref(v___x_3486_);
v___x_3489_ = lean_box(v___x_3487_);
lean_inc_ref(v___x_3488_);
v___f_3490_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__3___boxed), 18, 12);
lean_closure_set(v___f_3490_, 0, v___x_3488_);
lean_closure_set(v___f_3490_, 1, v_P_x27_3469_);
lean_closure_set(v___f_3490_, 2, v_fvars_3471_);
lean_closure_set(v___f_3490_, 3, v_path_3470_);
lean_closure_set(v___f_3490_, 4, v___f_3484_);
lean_closure_set(v___f_3490_, 5, v_a_3478_);
lean_closure_set(v___f_3490_, 6, v___x_3489_);
lean_closure_set(v___f_3490_, 7, v___x_3474_);
lean_closure_set(v___f_3490_, 8, v___x_3468_);
lean_closure_set(v___f_3490_, 9, v_00_u03b1_3475_);
lean_closure_set(v___f_3490_, 10, v_p_3473_);
lean_closure_set(v___f_3490_, 11, v_h_3476_);
v___x_3491_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_3467_, v___x_3477_, v___x_3488_, v___f_3490_, v___y_3479_, v___y_3480_, v___y_3481_, v___y_3482_);
return v___x_3491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed(lean_object** _args){
lean_object* v___x_3492_ = _args[0];
lean_object* v___x_3493_ = _args[1];
lean_object* v___x_3494_ = _args[2];
lean_object* v_P_x27_3495_ = _args[3];
lean_object* v_path_3496_ = _args[4];
lean_object* v_fvars_3497_ = _args[5];
lean_object* v___x_3498_ = _args[6];
lean_object* v_p_3499_ = _args[7];
lean_object* v___x_3500_ = _args[8];
lean_object* v_00_u03b1_3501_ = _args[9];
lean_object* v_h_3502_ = _args[10];
lean_object* v___x_3503_ = _args[11];
lean_object* v_a_3504_ = _args[12];
lean_object* v___y_3505_ = _args[13];
lean_object* v___y_3506_ = _args[14];
lean_object* v___y_3507_ = _args[15];
lean_object* v___y_3508_ = _args[16];
lean_object* v___y_3509_ = _args[17];
_start:
{
uint8_t v___x_4996__boxed_3510_; lean_object* v_res_3511_; 
v___x_4996__boxed_3510_ = lean_unbox(v___x_3503_);
v_res_3511_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4(v___x_3492_, v___x_3493_, v___x_3494_, v_P_x27_3495_, v_path_3496_, v_fvars_3497_, v___x_3498_, v_p_3499_, v___x_3500_, v_00_u03b1_3501_, v_h_3502_, v___x_4996__boxed_3510_, v_a_3504_, v___y_3505_, v___y_3506_, v___y_3507_, v___y_3508_);
lean_dec(v___y_3508_);
lean_dec_ref(v___y_3507_);
lean_dec(v___y_3506_);
lean_dec_ref(v___y_3505_);
return v_res_3511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5(lean_object* v_u_3512_, lean_object* v___x_3513_, lean_object* v___x_3514_, lean_object* v_P_x27_3515_, lean_object* v_path_3516_, lean_object* v_fvars_3517_, lean_object* v___x_3518_, lean_object* v_p_3519_, lean_object* v___x_3520_, lean_object* v_00_u03b1_3521_, uint8_t v___x_3522_, lean_object* v_h_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_, lean_object* v___y_3527_){
_start:
{
lean_object* v___x_3529_; lean_object* v___x_3530_; lean_object* v___f_3531_; lean_object* v___x_3532_; 
v___x_3529_ = l_Lean_Expr_sort___override(v_u_3512_);
v___x_3530_ = lean_box(v___x_3522_);
lean_inc_ref(v_00_u03b1_3521_);
lean_inc(v___x_3513_);
v___f_3531_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__4___boxed), 18, 12);
lean_closure_set(v___f_3531_, 0, v___x_3529_);
lean_closure_set(v___f_3531_, 1, v___x_3513_);
lean_closure_set(v___f_3531_, 2, v___x_3514_);
lean_closure_set(v___f_3531_, 3, v_P_x27_3515_);
lean_closure_set(v___f_3531_, 4, v_path_3516_);
lean_closure_set(v___f_3531_, 5, v_fvars_3517_);
lean_closure_set(v___f_3531_, 6, v___x_3518_);
lean_closure_set(v___f_3531_, 7, v_p_3519_);
lean_closure_set(v___f_3531_, 8, v___x_3520_);
lean_closure_set(v___f_3531_, 9, v_00_u03b1_3521_);
lean_closure_set(v___f_3531_, 10, v_h_3523_);
lean_closure_set(v___f_3531_, 11, v___x_3530_);
v___x_3532_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_3513_, v___x_3522_, v_00_u03b1_3521_, v___f_3531_, v___y_3524_, v___y_3525_, v___y_3526_, v___y_3527_);
return v___x_3532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5___boxed(lean_object** _args){
lean_object* v_u_3533_ = _args[0];
lean_object* v___x_3534_ = _args[1];
lean_object* v___x_3535_ = _args[2];
lean_object* v_P_x27_3536_ = _args[3];
lean_object* v_path_3537_ = _args[4];
lean_object* v_fvars_3538_ = _args[5];
lean_object* v___x_3539_ = _args[6];
lean_object* v_p_3540_ = _args[7];
lean_object* v___x_3541_ = _args[8];
lean_object* v_00_u03b1_3542_ = _args[9];
lean_object* v___x_3543_ = _args[10];
lean_object* v_h_3544_ = _args[11];
lean_object* v___y_3545_ = _args[12];
lean_object* v___y_3546_ = _args[13];
lean_object* v___y_3547_ = _args[14];
lean_object* v___y_3548_ = _args[15];
lean_object* v___y_3549_ = _args[16];
_start:
{
uint8_t v___x_5044__boxed_3550_; lean_object* v_res_3551_; 
v___x_5044__boxed_3550_ = lean_unbox(v___x_3543_);
v_res_3551_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5(v_u_3533_, v___x_3534_, v___x_3535_, v_P_x27_3536_, v_path_3537_, v_fvars_3538_, v___x_3539_, v_p_3540_, v___x_3541_, v_00_u03b1_3542_, v___x_5044__boxed_3550_, v_h_3544_, v___y_3545_, v___y_3546_, v___y_3547_, v___y_3548_);
lean_dec(v___y_3548_);
lean_dec_ref(v___y_3547_);
lean_dec(v___y_3546_);
lean_dec_ref(v___y_3545_);
return v_res_3551_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2(void){
_start:
{
lean_object* v___x_3555_; lean_object* v___x_3556_; 
v___x_3555_ = lean_unsigned_to_nat(0u);
v___x_3556_ = l_Lean_Expr_bvar___override(v___x_3555_);
return v___x_3556_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3(void){
_start:
{
lean_object* v___x_3557_; lean_object* v___x_3558_; lean_object* v___x_3559_; 
v___x_3557_ = lean_box(0);
v___x_3558_ = lean_obj_once(&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2, &lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2_once, _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__2);
v___x_3559_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3559_, 0, v___x_3558_);
lean_ctor_set(v___x_3559_, 1, v___x_3557_);
return v___x_3559_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4(void){
_start:
{
lean_object* v___x_3560_; lean_object* v___x_3561_; 
v___x_3560_ = lean_obj_once(&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3, &lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3_once, _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__3);
v___x_3561_ = lean_array_mk(v___x_3560_);
return v___x_3561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter(lean_object* v_u_3562_, lean_object* v_00_u03b1_3563_, lean_object* v_p_3564_, lean_object* v_P_x27_3565_, lean_object* v_fvars_3566_, lean_object* v_path_3567_, lean_object* v_a_3568_, lean_object* v_a_3569_, lean_object* v_a_3570_, lean_object* v_a_3571_){
_start:
{
lean_object* v___x_3573_; uint8_t v___x_3574_; lean_object* v___x_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___f_3583_; lean_object* v___x_3584_; uint8_t v___x_3585_; lean_object* v___x_3586_; lean_object* v___x_3587_; lean_object* v___x_3588_; lean_object* v___x_3589_; 
v___x_3573_ = lean_box(0);
v___x_3574_ = 0;
v___x_3575_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0));
v___x_3576_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_3577_ = lean_box(0);
lean_inc(v_u_3562_);
v___x_3578_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3578_, 0, v_u_3562_);
lean_ctor_set(v___x_3578_, 1, v___x_3577_);
lean_inc_ref(v___x_3578_);
v___x_3579_ = l_Lean_Expr_const___override(v___x_3576_, v___x_3578_);
lean_inc_ref_n(v_00_u03b1_3563_, 2);
v___x_3580_ = l_Lean_Expr_app___override(v___x_3579_, v_00_u03b1_3563_);
v___x_3581_ = ((lean_object*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1));
v___x_3582_ = lean_box(v___x_3574_);
lean_inc_ref(v_p_3564_);
v___f_3583_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___lam__5___boxed), 17, 11);
lean_closure_set(v___f_3583_, 0, v_u_3562_);
lean_closure_set(v___f_3583_, 1, v___x_3573_);
lean_closure_set(v___f_3583_, 2, v___x_3578_);
lean_closure_set(v___f_3583_, 3, v_P_x27_3565_);
lean_closure_set(v___f_3583_, 4, v_path_3567_);
lean_closure_set(v___f_3583_, 5, v_fvars_3566_);
lean_closure_set(v___f_3583_, 6, v___x_3577_);
lean_closure_set(v___f_3583_, 7, v_p_3564_);
lean_closure_set(v___f_3583_, 8, v___x_3575_);
lean_closure_set(v___f_3583_, 9, v_00_u03b1_3563_);
lean_closure_set(v___f_3583_, 10, v___x_3582_);
v___x_3584_ = lean_obj_once(&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4, &lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4_once, _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4);
v___x_3585_ = 0;
v___x_3586_ = l_Lean_Expr_betaRev(v_p_3564_, v___x_3584_, v___x_3585_, v___x_3585_);
v___x_3587_ = l_Lean_Expr_lam___override(v___x_3581_, v_00_u03b1_3563_, v___x_3586_, v___x_3574_);
v___x_3588_ = l_Lean_Expr_app___override(v___x_3580_, v___x_3587_);
v___x_3589_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_3573_, v___x_3574_, v___x_3588_, v___f_3583_, v_a_3568_, v_a_3569_, v_a_3570_, v_a_3571_);
return v___x_3589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkBeforeToAfter___boxed(lean_object* v_u_3590_, lean_object* v_00_u03b1_3591_, lean_object* v_p_3592_, lean_object* v_P_x27_3593_, lean_object* v_fvars_3594_, lean_object* v_path_3595_, lean_object* v_a_3596_, lean_object* v_a_3597_, lean_object* v_a_3598_, lean_object* v_a_3599_, lean_object* v_a_3600_){
_start:
{
lean_object* v_res_3601_; 
v_res_3601_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter(v_u_3590_, v_00_u03b1_3591_, v_p_3592_, v_P_x27_3593_, v_fvars_3594_, v_path_3595_, v_a_3596_, v_a_3597_, v_a_3598_, v_a_3599_);
lean_dec(v_a_3599_);
lean_dec_ref(v_a_3598_);
lean_dec(v_a_3597_);
lean_dec_ref(v_a_3596_);
return v_res_3601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1(lean_object* v_mvarId_3602_, lean_object* v_val_3603_, lean_object* v___y_3604_, lean_object* v___y_3605_, lean_object* v___y_3606_, lean_object* v___y_3607_){
_start:
{
lean_object* v___x_3609_; 
v___x_3609_ = lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___redArg(v_mvarId_3602_, v_val_3603_, v___y_3605_);
return v___x_3609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1___boxed(lean_object* v_mvarId_3610_, lean_object* v_val_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_){
_start:
{
lean_object* v_res_3617_; 
v_res_3617_ = lp_mathlib_Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1(v_mvarId_3610_, v_val_3611_, v___y_3612_, v___y_3613_, v___y_3614_, v___y_3615_);
lean_dec(v___y_3615_);
lean_dec_ref(v___y_3614_);
lean_dec(v___y_3613_);
lean_dec_ref(v___y_3612_);
return v_res_3617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1(lean_object* v_00_u03b2_3618_, lean_object* v_x_3619_, lean_object* v_x_3620_, lean_object* v_x_3621_){
_start:
{
lean_object* v___x_3622_; 
v___x_3622_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1___redArg(v_x_3619_, v_x_3620_, v_x_3621_);
return v___x_3622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_3623_, lean_object* v_x_3624_, size_t v_x_3625_, size_t v_x_3626_, lean_object* v_x_3627_, lean_object* v_x_3628_){
_start:
{
lean_object* v___x_3629_; 
v___x_3629_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___redArg(v_x_3624_, v_x_3625_, v_x_3626_, v_x_3627_, v_x_3628_);
return v___x_3629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_3630_, lean_object* v_x_3631_, lean_object* v_x_3632_, lean_object* v_x_3633_, lean_object* v_x_3634_, lean_object* v_x_3635_){
_start:
{
size_t v_x_5174__boxed_3636_; size_t v_x_5175__boxed_3637_; lean_object* v_res_3638_; 
v_x_5174__boxed_3636_ = lean_unbox_usize(v_x_3632_);
lean_dec(v_x_3632_);
v_x_5175__boxed_3637_ = lean_unbox_usize(v_x_3633_);
lean_dec(v_x_3633_);
v_res_3638_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3(v_00_u03b2_3630_, v_x_3631_, v_x_5174__boxed_3636_, v_x_5175__boxed_3637_, v_x_3634_, v_x_3635_);
return v_res_3638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4(lean_object* v_00_u03b2_3639_, lean_object* v_n_3640_, lean_object* v_k_3641_, lean_object* v_v_3642_){
_start:
{
lean_object* v___x_3643_; 
v___x_3643_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4___redArg(v_n_3640_, v_k_3641_, v_v_3642_);
return v___x_3643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_3644_, size_t v_depth_3645_, lean_object* v_keys_3646_, lean_object* v_vals_3647_, lean_object* v_heq_3648_, lean_object* v_i_3649_, lean_object* v_entries_3650_){
_start:
{
lean_object* v___x_3651_; 
v___x_3651_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___redArg(v_depth_3645_, v_keys_3646_, v_vals_3647_, v_i_3649_, v_entries_3650_);
return v___x_3651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03b2_3652_, lean_object* v_depth_3653_, lean_object* v_keys_3654_, lean_object* v_vals_3655_, lean_object* v_heq_3656_, lean_object* v_i_3657_, lean_object* v_entries_3658_){
_start:
{
size_t v_depth_boxed_3659_; lean_object* v_res_3660_; 
v_depth_boxed_3659_ = lean_unbox_usize(v_depth_3653_);
lean_dec(v_depth_3653_);
v_res_3660_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__5(v_00_u03b2_3652_, v_depth_boxed_3659_, v_keys_3654_, v_vals_3655_, v_heq_3656_, v_i_3657_, v_entries_3658_);
lean_dec_ref(v_vals_3655_);
lean_dec_ref(v_keys_3654_);
return v_res_3660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_3661_, lean_object* v_x_3662_, lean_object* v_x_3663_, lean_object* v_x_3664_, lean_object* v_x_3665_){
_start:
{
lean_object* v___x_3666_; 
v___x_3666_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00ExistsAndEq_mkBeforeToAfter_spec__1_spec__1_spec__3_spec__4_spec__5___redArg(v_x_3662_, v_x_3663_, v_x_3664_, v_x_3665_);
return v___x_3666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(lean_object* v_a_x27_3667_, lean_object* v___x_3668_, lean_object* v_p_3669_, uint8_t v___x_3670_, lean_object* v_fvars_3671_, lean_object* v_path_3672_, lean_object* v___x_3673_, lean_object* v___x_3674_, lean_object* v_00_u03b1_3675_, lean_object* v___x_3676_, lean_object* v_leaves_3677_, lean_object* v_x_3678_, lean_object* v___y_3679_, lean_object* v___y_3680_, lean_object* v___y_3681_, lean_object* v___y_3682_){
_start:
{
lean_object* v___x_3684_; lean_object* v___x_3685_; lean_object* v___x_3686_; lean_object* v___x_3687_; 
lean_inc_ref(v_a_x27_3667_);
v___x_3684_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3684_, 0, v_a_x27_3667_);
lean_ctor_set(v___x_3684_, 1, v___x_3668_);
v___x_3685_ = lean_array_mk(v___x_3684_);
v___x_3686_ = l_Lean_Expr_betaRev(v_p_3669_, v___x_3685_, v___x_3670_, v___x_3670_);
lean_dec_ref(v___x_3685_);
v___x_3687_ = lp_mathlib_ExistsAndEq_construct(v___x_3686_, v_fvars_3671_, v_path_3672_, v_leaves_3677_, v___y_3679_, v___y_3680_, v___y_3681_, v___y_3682_);
if (lean_obj_tag(v___x_3687_) == 0)
{
lean_object* v_a_3688_; lean_object* v___x_3690_; uint8_t v_isShared_3691_; uint8_t v_isSharedCheck_3702_; 
v_a_3688_ = lean_ctor_get(v___x_3687_, 0);
v_isSharedCheck_3702_ = !lean_is_exclusive(v___x_3687_);
if (v_isSharedCheck_3702_ == 0)
{
v___x_3690_ = v___x_3687_;
v_isShared_3691_ = v_isSharedCheck_3702_;
goto v_resetjp_3689_;
}
else
{
lean_inc(v_a_3688_);
lean_dec(v___x_3687_);
v___x_3690_ = lean_box(0);
v_isShared_3691_ = v_isSharedCheck_3702_;
goto v_resetjp_3689_;
}
v_resetjp_3689_:
{
lean_object* v___x_3692_; lean_object* v___x_3693_; lean_object* v___x_3694_; lean_object* v___x_3695_; lean_object* v___x_3696_; lean_object* v___x_3697_; lean_object* v___x_3698_; lean_object* v___x_3700_; 
v___x_3692_ = ((lean_object*)(lp_mathlib_ExistsAndEq_construct___closed__9));
v___x_3693_ = l_Lean_Name_mkStr2(v___x_3673_, v___x_3692_);
v___x_3694_ = l_Lean_Expr_const___override(v___x_3693_, v___x_3674_);
v___x_3695_ = l_Lean_Expr_app___override(v___x_3694_, v_00_u03b1_3675_);
v___x_3696_ = l_Lean_Expr_app___override(v___x_3695_, v___x_3676_);
v___x_3697_ = l_Lean_Expr_app___override(v___x_3696_, v_a_x27_3667_);
v___x_3698_ = l_Lean_Expr_app___override(v___x_3697_, v_a_3688_);
if (v_isShared_3691_ == 0)
{
lean_ctor_set(v___x_3690_, 0, v___x_3698_);
v___x_3700_ = v___x_3690_;
goto v_reusejp_3699_;
}
else
{
lean_object* v_reuseFailAlloc_3701_; 
v_reuseFailAlloc_3701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3701_, 0, v___x_3698_);
v___x_3700_ = v_reuseFailAlloc_3701_;
goto v_reusejp_3699_;
}
v_reusejp_3699_:
{
return v___x_3700_;
}
}
}
else
{
lean_dec_ref(v___x_3676_);
lean_dec_ref(v_00_u03b1_3675_);
lean_dec(v___x_3674_);
lean_dec_ref(v___x_3673_);
lean_dec_ref(v_a_x27_3667_);
return v___x_3687_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed(lean_object** _args){
lean_object* v_a_x27_3703_ = _args[0];
lean_object* v___x_3704_ = _args[1];
lean_object* v_p_3705_ = _args[2];
lean_object* v___x_3706_ = _args[3];
lean_object* v_fvars_3707_ = _args[4];
lean_object* v_path_3708_ = _args[5];
lean_object* v___x_3709_ = _args[6];
lean_object* v___x_3710_ = _args[7];
lean_object* v_00_u03b1_3711_ = _args[8];
lean_object* v___x_3712_ = _args[9];
lean_object* v_leaves_3713_ = _args[10];
lean_object* v_x_3714_ = _args[11];
lean_object* v___y_3715_ = _args[12];
lean_object* v___y_3716_ = _args[13];
lean_object* v___y_3717_ = _args[14];
lean_object* v___y_3718_ = _args[15];
lean_object* v___y_3719_ = _args[16];
_start:
{
uint8_t v___x_378__boxed_3720_; lean_object* v_res_3721_; 
v___x_378__boxed_3720_ = lean_unbox(v___x_3706_);
v_res_3721_ = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0(v_a_x27_3703_, v___x_3704_, v_p_3705_, v___x_378__boxed_3720_, v_fvars_3707_, v_path_3708_, v___x_3709_, v___x_3710_, v_00_u03b1_3711_, v___x_3712_, v_leaves_3713_, v_x_3714_, v___y_3715_, v___y_3716_, v___y_3717_, v___y_3718_);
lean_dec(v___y_3718_);
lean_dec_ref(v___y_3717_);
lean_dec(v___y_3716_);
lean_dec_ref(v___y_3715_);
lean_dec_ref(v_x_3714_);
return v_res_3721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(lean_object* v_u_3722_, lean_object* v_00_u03b1_3723_, lean_object* v_p_3724_, uint8_t v___x_3725_, lean_object* v_a_x27_3726_, lean_object* v_fvars_3727_, lean_object* v_path_3728_, lean_object* v_P_x27_3729_, lean_object* v_h_3730_, lean_object* v___y_3731_, lean_object* v___y_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_){
_start:
{
lean_object* v___x_3736_; lean_object* v___x_3737_; lean_object* v___x_3738_; lean_object* v___x_3739_; lean_object* v___x_3740_; lean_object* v___x_3741_; lean_object* v___x_3742_; lean_object* v___x_3743_; uint8_t v___x_3744_; lean_object* v___x_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___f_3748_; lean_object* v___x_3749_; lean_object* v___x_3750_; lean_object* v___x_3751_; 
v___x_3736_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__0));
v___x_3737_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_3738_ = lean_box(0);
v___x_3739_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3739_, 0, v_u_3722_);
lean_ctor_set(v___x_3739_, 1, v___x_3738_);
lean_inc_ref(v___x_3739_);
v___x_3740_ = l_Lean_Expr_const___override(v___x_3737_, v___x_3739_);
lean_inc_ref_n(v_00_u03b1_3723_, 2);
v___x_3741_ = l_Lean_Expr_app___override(v___x_3740_, v_00_u03b1_3723_);
v___x_3742_ = ((lean_object*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1));
v___x_3743_ = lean_obj_once(&lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4, &lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4_once, _init_lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__4);
v___x_3744_ = 0;
lean_inc_ref(v_p_3724_);
v___x_3745_ = l_Lean_Expr_betaRev(v_p_3724_, v___x_3743_, v___x_3744_, v___x_3744_);
v___x_3746_ = l_Lean_Expr_lam___override(v___x_3742_, v_00_u03b1_3723_, v___x_3745_, v___x_3725_);
v___x_3747_ = lean_box(v___x_3744_);
lean_inc_ref(v___x_3746_);
lean_inc(v_path_3728_);
lean_inc(v_fvars_3727_);
v___f_3748_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__0___boxed), 17, 10);
lean_closure_set(v___f_3748_, 0, v_a_x27_3726_);
lean_closure_set(v___f_3748_, 1, v___x_3738_);
lean_closure_set(v___f_3748_, 2, v_p_3724_);
lean_closure_set(v___f_3748_, 3, v___x_3747_);
lean_closure_set(v___f_3748_, 4, v_fvars_3727_);
lean_closure_set(v___f_3748_, 5, v_path_3728_);
lean_closure_set(v___f_3748_, 6, v___x_3736_);
lean_closure_set(v___f_3748_, 7, v___x_3739_);
lean_closure_set(v___f_3748_, 8, v_00_u03b1_3723_);
lean_closure_set(v___f_3748_, 9, v___x_3746_);
v___x_3749_ = l_Lean_Expr_app___override(v___x_3741_, v___x_3746_);
v___x_3750_ = lp_mathlib_ExistsAndEq_Path_forResult(v_path_3728_);
lean_inc_ref(v_h_3730_);
v___x_3751_ = lp_mathlib_ExistsAndEq_destruct(v_P_x27_3729_, v___x_3749_, v_h_3730_, v_fvars_3727_, v___x_3750_, v___x_3738_, v___f_3748_, v___y_3731_, v___y_3732_, v___y_3733_, v___y_3734_);
if (lean_obj_tag(v___x_3751_) == 0)
{
lean_object* v_a_3752_; lean_object* v___x_3753_; lean_object* v___x_3754_; lean_object* v___x_3755_; uint8_t v___x_3756_; uint8_t v___x_3757_; lean_object* v___x_3758_; 
v_a_3752_ = lean_ctor_get(v___x_3751_, 0);
lean_inc(v_a_3752_);
lean_dec_ref_known(v___x_3751_, 1);
v___x_3753_ = lean_unsigned_to_nat(1u);
v___x_3754_ = lean_mk_empty_array_with_capacity(v___x_3753_);
v___x_3755_ = lean_array_push(v___x_3754_, v_h_3730_);
v___x_3756_ = 1;
v___x_3757_ = 1;
v___x_3758_ = l_Lean_Meta_mkLambdaFVars(v___x_3755_, v_a_3752_, v___x_3744_, v___x_3756_, v___x_3744_, v___x_3756_, v___x_3757_, v___y_3731_, v___y_3732_, v___y_3733_, v___y_3734_);
lean_dec_ref(v___x_3755_);
return v___x_3758_;
}
else
{
lean_dec_ref(v_h_3730_);
return v___x_3751_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed(lean_object* v_u_3759_, lean_object* v_00_u03b1_3760_, lean_object* v_p_3761_, lean_object* v___x_3762_, lean_object* v_a_x27_3763_, lean_object* v_fvars_3764_, lean_object* v_path_3765_, lean_object* v_P_x27_3766_, lean_object* v_h_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_){
_start:
{
uint8_t v___x_462__boxed_3773_; lean_object* v_res_3774_; 
v___x_462__boxed_3773_ = lean_unbox(v___x_3762_);
v_res_3774_ = lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1(v_u_3759_, v_00_u03b1_3760_, v_p_3761_, v___x_462__boxed_3773_, v_a_x27_3763_, v_fvars_3764_, v_path_3765_, v_P_x27_3766_, v_h_3767_, v___y_3768_, v___y_3769_, v___y_3770_, v___y_3771_);
lean_dec(v___y_3771_);
lean_dec_ref(v___y_3770_);
lean_dec(v___y_3769_);
lean_dec_ref(v___y_3768_);
return v_res_3774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore(lean_object* v_u_3775_, lean_object* v_00_u03b1_3776_, lean_object* v_p_3777_, lean_object* v_P_x27_3778_, lean_object* v_a_x27_3779_, lean_object* v_fvars_3780_, lean_object* v_path_3781_, lean_object* v_a_3782_, lean_object* v_a_3783_, lean_object* v_a_3784_, lean_object* v_a_3785_){
_start:
{
lean_object* v___x_3787_; uint8_t v___x_3788_; lean_object* v___x_3789_; lean_object* v___f_3790_; lean_object* v___x_3791_; 
v___x_3787_ = lean_box(0);
v___x_3788_ = 0;
v___x_3789_ = lean_box(v___x_3788_);
lean_inc_ref(v_P_x27_3778_);
v___f_3790_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_mkAfterToBefore___lam__1___boxed), 14, 8);
lean_closure_set(v___f_3790_, 0, v_u_3775_);
lean_closure_set(v___f_3790_, 1, v_00_u03b1_3776_);
lean_closure_set(v___f_3790_, 2, v_p_3777_);
lean_closure_set(v___f_3790_, 3, v___x_3789_);
lean_closure_set(v___f_3790_, 4, v_a_x27_3779_);
lean_closure_set(v___f_3790_, 5, v_fvars_3780_);
lean_closure_set(v___f_3790_, 6, v_path_3781_);
lean_closure_set(v___f_3790_, 7, v_P_x27_3778_);
v___x_3791_ = lp_mathlib_Qq_withLocalDeclQ___at___00ExistsAndEq_destruct_spec__1___redArg(v___x_3787_, v___x_3788_, v_P_x27_3778_, v___f_3790_, v_a_3782_, v_a_3783_, v_a_3784_, v_a_3785_);
return v___x_3791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_mkAfterToBefore___boxed(lean_object* v_u_3792_, lean_object* v_00_u03b1_3793_, lean_object* v_p_3794_, lean_object* v_P_x27_3795_, lean_object* v_a_x27_3796_, lean_object* v_fvars_3797_, lean_object* v_path_3798_, lean_object* v_a_3799_, lean_object* v_a_3800_, lean_object* v_a_3801_, lean_object* v_a_3802_, lean_object* v_a_3803_){
_start:
{
lean_object* v_res_3804_; 
v_res_3804_ = lp_mathlib_ExistsAndEq_mkAfterToBefore(v_u_3792_, v_00_u03b1_3793_, v_p_3794_, v_P_x27_3795_, v_a_x27_3796_, v_fvars_3797_, v_path_3798_, v_a_3799_, v_a_3800_, v_a_3801_, v_a_3802_);
lean_dec(v_a_3802_);
lean_dec_ref(v_a_3801_);
lean_dec(v_a_3800_);
lean_dec_ref(v_a_3799_);
return v_res_3804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0(lean_object* v_xs_3805_, lean_object* v_mvars_3806_, lean_object* v_t_3807_, lean_object* v___y_3808_, lean_object* v___y_3809_, lean_object* v___y_3810_, lean_object* v___y_3811_){
_start:
{
lean_object* v___x_3813_; lean_object* v___x_3814_; 
v___x_3813_ = l_Lean_Expr_replaceFVars(v_t_3807_, v_xs_3805_, v_mvars_3806_);
v___x_3814_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v___x_3813_, v___y_3809_);
return v___x_3814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0___boxed(lean_object* v_xs_3815_, lean_object* v_mvars_3816_, lean_object* v_t_3817_, lean_object* v___y_3818_, lean_object* v___y_3819_, lean_object* v___y_3820_, lean_object* v___y_3821_, lean_object* v___y_3822_){
_start:
{
lean_object* v_res_3823_; 
v_res_3823_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0(v_xs_3815_, v_mvars_3816_, v_t_3817_, v___y_3818_, v___y_3819_, v___y_3820_, v___y_3821_);
lean_dec(v___y_3821_);
lean_dec_ref(v___y_3820_);
lean_dec(v___y_3819_);
lean_dec_ref(v___y_3818_);
lean_dec_ref(v_t_3817_);
lean_dec_ref(v_mvars_3816_);
lean_dec_ref(v_xs_3815_);
return v_res_3823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(lean_object* v___f_3824_, lean_object* v_res_3825_, lean_object* v___y_3826_, lean_object* v___y_3827_, lean_object* v___y_3828_, lean_object* v___y_3829_){
_start:
{
lean_object* v_expr_3831_; lean_object* v_proof_x3f_3832_; uint8_t v_cache_3833_; lean_object* v___x_3835_; uint8_t v_isShared_3836_; uint8_t v_isSharedCheck_3877_; 
v_expr_3831_ = lean_ctor_get(v_res_3825_, 0);
v_proof_x3f_3832_ = lean_ctor_get(v_res_3825_, 1);
v_cache_3833_ = lean_ctor_get_uint8(v_res_3825_, sizeof(void*)*2);
v_isSharedCheck_3877_ = !lean_is_exclusive(v_res_3825_);
if (v_isSharedCheck_3877_ == 0)
{
v___x_3835_ = v_res_3825_;
v_isShared_3836_ = v_isSharedCheck_3877_;
goto v_resetjp_3834_;
}
else
{
lean_inc(v_proof_x3f_3832_);
lean_inc(v_expr_3831_);
lean_dec(v_res_3825_);
v___x_3835_ = lean_box(0);
v_isShared_3836_ = v_isSharedCheck_3877_;
goto v_resetjp_3834_;
}
v_resetjp_3834_:
{
lean_object* v___x_3837_; 
lean_inc_ref(v___f_3824_);
lean_inc(v___y_3829_);
lean_inc_ref(v___y_3828_);
lean_inc(v___y_3827_);
lean_inc_ref(v___y_3826_);
v___x_3837_ = lean_apply_6(v___f_3824_, v_expr_3831_, v___y_3826_, v___y_3827_, v___y_3828_, v___y_3829_, lean_box(0));
if (lean_obj_tag(v___x_3837_) == 0)
{
lean_object* v_a_3838_; lean_object* v___x_3840_; uint8_t v_isShared_3841_; uint8_t v_isSharedCheck_3868_; 
v_a_3838_ = lean_ctor_get(v___x_3837_, 0);
v_isSharedCheck_3868_ = !lean_is_exclusive(v___x_3837_);
if (v_isSharedCheck_3868_ == 0)
{
v___x_3840_ = v___x_3837_;
v_isShared_3841_ = v_isSharedCheck_3868_;
goto v_resetjp_3839_;
}
else
{
lean_inc(v_a_3838_);
lean_dec(v___x_3837_);
v___x_3840_ = lean_box(0);
v_isShared_3841_ = v_isSharedCheck_3868_;
goto v_resetjp_3839_;
}
v_resetjp_3839_:
{
lean_object* v_a_3843_; 
if (lean_obj_tag(v_proof_x3f_3832_) == 0)
{
lean_dec_ref(v___f_3824_);
v_a_3843_ = v_proof_x3f_3832_;
goto v___jp_3842_;
}
else
{
lean_object* v_val_3850_; lean_object* v___x_3852_; uint8_t v_isShared_3853_; uint8_t v_isSharedCheck_3867_; 
v_val_3850_ = lean_ctor_get(v_proof_x3f_3832_, 0);
v_isSharedCheck_3867_ = !lean_is_exclusive(v_proof_x3f_3832_);
if (v_isSharedCheck_3867_ == 0)
{
v___x_3852_ = v_proof_x3f_3832_;
v_isShared_3853_ = v_isSharedCheck_3867_;
goto v_resetjp_3851_;
}
else
{
lean_inc(v_val_3850_);
lean_dec(v_proof_x3f_3832_);
v___x_3852_ = lean_box(0);
v_isShared_3853_ = v_isSharedCheck_3867_;
goto v_resetjp_3851_;
}
v_resetjp_3851_:
{
lean_object* v___x_3854_; 
lean_inc(v___y_3829_);
lean_inc_ref(v___y_3828_);
lean_inc(v___y_3827_);
lean_inc_ref(v___y_3826_);
v___x_3854_ = lean_apply_6(v___f_3824_, v_val_3850_, v___y_3826_, v___y_3827_, v___y_3828_, v___y_3829_, lean_box(0));
if (lean_obj_tag(v___x_3854_) == 0)
{
lean_object* v_a_3855_; lean_object* v___x_3857_; 
v_a_3855_ = lean_ctor_get(v___x_3854_, 0);
lean_inc(v_a_3855_);
lean_dec_ref_known(v___x_3854_, 1);
if (v_isShared_3853_ == 0)
{
lean_ctor_set(v___x_3852_, 0, v_a_3855_);
v___x_3857_ = v___x_3852_;
goto v_reusejp_3856_;
}
else
{
lean_object* v_reuseFailAlloc_3858_; 
v_reuseFailAlloc_3858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3858_, 0, v_a_3855_);
v___x_3857_ = v_reuseFailAlloc_3858_;
goto v_reusejp_3856_;
}
v_reusejp_3856_:
{
v_a_3843_ = v___x_3857_;
goto v___jp_3842_;
}
}
else
{
lean_object* v_a_3859_; lean_object* v___x_3861_; uint8_t v_isShared_3862_; uint8_t v_isSharedCheck_3866_; 
lean_del_object(v___x_3852_);
lean_del_object(v___x_3840_);
lean_dec(v_a_3838_);
lean_del_object(v___x_3835_);
v_a_3859_ = lean_ctor_get(v___x_3854_, 0);
v_isSharedCheck_3866_ = !lean_is_exclusive(v___x_3854_);
if (v_isSharedCheck_3866_ == 0)
{
v___x_3861_ = v___x_3854_;
v_isShared_3862_ = v_isSharedCheck_3866_;
goto v_resetjp_3860_;
}
else
{
lean_inc(v_a_3859_);
lean_dec(v___x_3854_);
v___x_3861_ = lean_box(0);
v_isShared_3862_ = v_isSharedCheck_3866_;
goto v_resetjp_3860_;
}
v_resetjp_3860_:
{
lean_object* v___x_3864_; 
if (v_isShared_3862_ == 0)
{
v___x_3864_ = v___x_3861_;
goto v_reusejp_3863_;
}
else
{
lean_object* v_reuseFailAlloc_3865_; 
v_reuseFailAlloc_3865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3865_, 0, v_a_3859_);
v___x_3864_ = v_reuseFailAlloc_3865_;
goto v_reusejp_3863_;
}
v_reusejp_3863_:
{
return v___x_3864_;
}
}
}
}
}
v___jp_3842_:
{
lean_object* v___x_3845_; 
if (v_isShared_3836_ == 0)
{
lean_ctor_set(v___x_3835_, 1, v_a_3843_);
lean_ctor_set(v___x_3835_, 0, v_a_3838_);
v___x_3845_ = v___x_3835_;
goto v_reusejp_3844_;
}
else
{
lean_object* v_reuseFailAlloc_3849_; 
v_reuseFailAlloc_3849_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_3849_, 0, v_a_3838_);
lean_ctor_set(v_reuseFailAlloc_3849_, 1, v_a_3843_);
lean_ctor_set_uint8(v_reuseFailAlloc_3849_, sizeof(void*)*2, v_cache_3833_);
v___x_3845_ = v_reuseFailAlloc_3849_;
goto v_reusejp_3844_;
}
v_reusejp_3844_:
{
lean_object* v___x_3847_; 
if (v_isShared_3841_ == 0)
{
lean_ctor_set(v___x_3840_, 0, v___x_3845_);
v___x_3847_ = v___x_3840_;
goto v_reusejp_3846_;
}
else
{
lean_object* v_reuseFailAlloc_3848_; 
v_reuseFailAlloc_3848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3848_, 0, v___x_3845_);
v___x_3847_ = v_reuseFailAlloc_3848_;
goto v_reusejp_3846_;
}
v_reusejp_3846_:
{
return v___x_3847_;
}
}
}
}
}
else
{
lean_object* v_a_3869_; lean_object* v___x_3871_; uint8_t v_isShared_3872_; uint8_t v_isSharedCheck_3876_; 
lean_del_object(v___x_3835_);
lean_dec(v_proof_x3f_3832_);
lean_dec_ref(v___f_3824_);
v_a_3869_ = lean_ctor_get(v___x_3837_, 0);
v_isSharedCheck_3876_ = !lean_is_exclusive(v___x_3837_);
if (v_isSharedCheck_3876_ == 0)
{
v___x_3871_ = v___x_3837_;
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
else
{
lean_inc(v_a_3869_);
lean_dec(v___x_3837_);
v___x_3871_ = lean_box(0);
v_isShared_3872_ = v_isSharedCheck_3876_;
goto v_resetjp_3870_;
}
v_resetjp_3870_:
{
lean_object* v___x_3874_; 
if (v_isShared_3872_ == 0)
{
v___x_3874_ = v___x_3871_;
goto v_reusejp_3873_;
}
else
{
lean_object* v_reuseFailAlloc_3875_; 
v_reuseFailAlloc_3875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3875_, 0, v_a_3869_);
v___x_3874_ = v_reuseFailAlloc_3875_;
goto v_reusejp_3873_;
}
v_reusejp_3873_:
{
return v___x_3874_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1___boxed(lean_object* v___f_3878_, lean_object* v_res_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_){
_start:
{
lean_object* v_res_3885_; 
v_res_3885_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(v___f_3878_, v_res_3879_, v___y_3880_, v___y_3881_, v___y_3882_, v___y_3883_);
lean_dec(v___y_3883_);
lean_dec_ref(v___y_3882_);
lean_dec(v___y_3881_);
lean_dec_ref(v___y_3880_);
return v_res_3885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2(lean_object* v_k_3886_, lean_object* v_mvars_3887_, lean_object* v_xs_3888_, lean_object* v_e_x27_3889_, lean_object* v___y_3890_, lean_object* v___y_3891_, lean_object* v___y_3892_, lean_object* v___y_3893_){
_start:
{
lean_object* v_a_3896_; lean_object* v___x_3899_; 
lean_inc(v___y_3893_);
lean_inc_ref(v___y_3892_);
lean_inc(v___y_3891_);
lean_inc_ref(v___y_3890_);
v___x_3899_ = lean_apply_6(v_k_3886_, v_e_x27_3889_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_, lean_box(0));
if (lean_obj_tag(v___x_3899_) == 0)
{
lean_object* v_a_3900_; lean_object* v___f_3901_; 
v_a_3900_ = lean_ctor_get(v___x_3899_, 0);
lean_inc(v_a_3900_);
lean_dec_ref_known(v___x_3899_, 1);
v___f_3901_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withAbstractMVars___lam__0___boxed), 8, 2);
lean_closure_set(v___f_3901_, 0, v_xs_3888_);
lean_closure_set(v___f_3901_, 1, v_mvars_3887_);
switch(lean_obj_tag(v_a_3900_))
{
case 0:
{
lean_object* v_r_3902_; lean_object* v___x_3904_; uint8_t v_isShared_3905_; uint8_t v_isSharedCheck_3926_; 
v_r_3902_ = lean_ctor_get(v_a_3900_, 0);
v_isSharedCheck_3926_ = !lean_is_exclusive(v_a_3900_);
if (v_isSharedCheck_3926_ == 0)
{
v___x_3904_ = v_a_3900_;
v_isShared_3905_ = v_isSharedCheck_3926_;
goto v_resetjp_3903_;
}
else
{
lean_inc(v_r_3902_);
lean_dec(v_a_3900_);
v___x_3904_ = lean_box(0);
v_isShared_3905_ = v_isSharedCheck_3926_;
goto v_resetjp_3903_;
}
v_resetjp_3903_:
{
lean_object* v___x_3906_; 
v___x_3906_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(v___f_3901_, v_r_3902_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_);
if (lean_obj_tag(v___x_3906_) == 0)
{
lean_object* v_a_3907_; lean_object* v___x_3909_; uint8_t v_isShared_3910_; uint8_t v_isSharedCheck_3917_; 
v_a_3907_ = lean_ctor_get(v___x_3906_, 0);
v_isSharedCheck_3917_ = !lean_is_exclusive(v___x_3906_);
if (v_isSharedCheck_3917_ == 0)
{
v___x_3909_ = v___x_3906_;
v_isShared_3910_ = v_isSharedCheck_3917_;
goto v_resetjp_3908_;
}
else
{
lean_inc(v_a_3907_);
lean_dec(v___x_3906_);
v___x_3909_ = lean_box(0);
v_isShared_3910_ = v_isSharedCheck_3917_;
goto v_resetjp_3908_;
}
v_resetjp_3908_:
{
lean_object* v___x_3912_; 
if (v_isShared_3905_ == 0)
{
lean_ctor_set(v___x_3904_, 0, v_a_3907_);
v___x_3912_ = v___x_3904_;
goto v_reusejp_3911_;
}
else
{
lean_object* v_reuseFailAlloc_3916_; 
v_reuseFailAlloc_3916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3916_, 0, v_a_3907_);
v___x_3912_ = v_reuseFailAlloc_3916_;
goto v_reusejp_3911_;
}
v_reusejp_3911_:
{
lean_object* v___x_3914_; 
if (v_isShared_3910_ == 0)
{
lean_ctor_set(v___x_3909_, 0, v___x_3912_);
v___x_3914_ = v___x_3909_;
goto v_reusejp_3913_;
}
else
{
lean_object* v_reuseFailAlloc_3915_; 
v_reuseFailAlloc_3915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3915_, 0, v___x_3912_);
v___x_3914_ = v_reuseFailAlloc_3915_;
goto v_reusejp_3913_;
}
v_reusejp_3913_:
{
return v___x_3914_;
}
}
}
}
else
{
lean_object* v_a_3918_; lean_object* v___x_3920_; uint8_t v_isShared_3921_; uint8_t v_isSharedCheck_3925_; 
lean_del_object(v___x_3904_);
v_a_3918_ = lean_ctor_get(v___x_3906_, 0);
v_isSharedCheck_3925_ = !lean_is_exclusive(v___x_3906_);
if (v_isSharedCheck_3925_ == 0)
{
v___x_3920_ = v___x_3906_;
v_isShared_3921_ = v_isSharedCheck_3925_;
goto v_resetjp_3919_;
}
else
{
lean_inc(v_a_3918_);
lean_dec(v___x_3906_);
v___x_3920_ = lean_box(0);
v_isShared_3921_ = v_isSharedCheck_3925_;
goto v_resetjp_3919_;
}
v_resetjp_3919_:
{
lean_object* v___x_3923_; 
if (v_isShared_3921_ == 0)
{
v___x_3923_ = v___x_3920_;
goto v_reusejp_3922_;
}
else
{
lean_object* v_reuseFailAlloc_3924_; 
v_reuseFailAlloc_3924_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3924_, 0, v_a_3918_);
v___x_3923_ = v_reuseFailAlloc_3924_;
goto v_reusejp_3922_;
}
v_reusejp_3922_:
{
return v___x_3923_;
}
}
}
}
}
case 1:
{
lean_object* v_e_3927_; lean_object* v___x_3929_; uint8_t v_isShared_3930_; uint8_t v_isSharedCheck_3951_; 
v_e_3927_ = lean_ctor_get(v_a_3900_, 0);
v_isSharedCheck_3951_ = !lean_is_exclusive(v_a_3900_);
if (v_isSharedCheck_3951_ == 0)
{
v___x_3929_ = v_a_3900_;
v_isShared_3930_ = v_isSharedCheck_3951_;
goto v_resetjp_3928_;
}
else
{
lean_inc(v_e_3927_);
lean_dec(v_a_3900_);
v___x_3929_ = lean_box(0);
v_isShared_3930_ = v_isSharedCheck_3951_;
goto v_resetjp_3928_;
}
v_resetjp_3928_:
{
lean_object* v___x_3931_; 
v___x_3931_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(v___f_3901_, v_e_3927_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_);
if (lean_obj_tag(v___x_3931_) == 0)
{
lean_object* v_a_3932_; lean_object* v___x_3934_; uint8_t v_isShared_3935_; uint8_t v_isSharedCheck_3942_; 
v_a_3932_ = lean_ctor_get(v___x_3931_, 0);
v_isSharedCheck_3942_ = !lean_is_exclusive(v___x_3931_);
if (v_isSharedCheck_3942_ == 0)
{
v___x_3934_ = v___x_3931_;
v_isShared_3935_ = v_isSharedCheck_3942_;
goto v_resetjp_3933_;
}
else
{
lean_inc(v_a_3932_);
lean_dec(v___x_3931_);
v___x_3934_ = lean_box(0);
v_isShared_3935_ = v_isSharedCheck_3942_;
goto v_resetjp_3933_;
}
v_resetjp_3933_:
{
lean_object* v___x_3937_; 
if (v_isShared_3930_ == 0)
{
lean_ctor_set(v___x_3929_, 0, v_a_3932_);
v___x_3937_ = v___x_3929_;
goto v_reusejp_3936_;
}
else
{
lean_object* v_reuseFailAlloc_3941_; 
v_reuseFailAlloc_3941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3941_, 0, v_a_3932_);
v___x_3937_ = v_reuseFailAlloc_3941_;
goto v_reusejp_3936_;
}
v_reusejp_3936_:
{
lean_object* v___x_3939_; 
if (v_isShared_3935_ == 0)
{
lean_ctor_set(v___x_3934_, 0, v___x_3937_);
v___x_3939_ = v___x_3934_;
goto v_reusejp_3938_;
}
else
{
lean_object* v_reuseFailAlloc_3940_; 
v_reuseFailAlloc_3940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3940_, 0, v___x_3937_);
v___x_3939_ = v_reuseFailAlloc_3940_;
goto v_reusejp_3938_;
}
v_reusejp_3938_:
{
return v___x_3939_;
}
}
}
}
else
{
lean_object* v_a_3943_; lean_object* v___x_3945_; uint8_t v_isShared_3946_; uint8_t v_isSharedCheck_3950_; 
lean_del_object(v___x_3929_);
v_a_3943_ = lean_ctor_get(v___x_3931_, 0);
v_isSharedCheck_3950_ = !lean_is_exclusive(v___x_3931_);
if (v_isSharedCheck_3950_ == 0)
{
v___x_3945_ = v___x_3931_;
v_isShared_3946_ = v_isSharedCheck_3950_;
goto v_resetjp_3944_;
}
else
{
lean_inc(v_a_3943_);
lean_dec(v___x_3931_);
v___x_3945_ = lean_box(0);
v_isShared_3946_ = v_isSharedCheck_3950_;
goto v_resetjp_3944_;
}
v_resetjp_3944_:
{
lean_object* v___x_3948_; 
if (v_isShared_3946_ == 0)
{
v___x_3948_ = v___x_3945_;
goto v_reusejp_3947_;
}
else
{
lean_object* v_reuseFailAlloc_3949_; 
v_reuseFailAlloc_3949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3949_, 0, v_a_3943_);
v___x_3948_ = v_reuseFailAlloc_3949_;
goto v_reusejp_3947_;
}
v_reusejp_3947_:
{
return v___x_3948_;
}
}
}
}
}
default: 
{
lean_object* v_e_x3f_3952_; 
v_e_x3f_3952_ = lean_ctor_get(v_a_3900_, 0);
lean_inc(v_e_x3f_3952_);
lean_dec_ref_known(v_a_3900_, 1);
if (lean_obj_tag(v_e_x3f_3952_) == 0)
{
lean_dec_ref(v___f_3901_);
v_a_3896_ = v_e_x3f_3952_;
goto v___jp_3895_;
}
else
{
lean_object* v_val_3953_; lean_object* v___x_3955_; uint8_t v_isShared_3956_; uint8_t v_isSharedCheck_3970_; 
v_val_3953_ = lean_ctor_get(v_e_x3f_3952_, 0);
v_isSharedCheck_3970_ = !lean_is_exclusive(v_e_x3f_3952_);
if (v_isSharedCheck_3970_ == 0)
{
v___x_3955_ = v_e_x3f_3952_;
v_isShared_3956_ = v_isSharedCheck_3970_;
goto v_resetjp_3954_;
}
else
{
lean_inc(v_val_3953_);
lean_dec(v_e_x3f_3952_);
v___x_3955_ = lean_box(0);
v_isShared_3956_ = v_isSharedCheck_3970_;
goto v_resetjp_3954_;
}
v_resetjp_3954_:
{
lean_object* v___x_3957_; 
v___x_3957_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__1(v___f_3901_, v_val_3953_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_);
if (lean_obj_tag(v___x_3957_) == 0)
{
lean_object* v_a_3958_; lean_object* v___x_3960_; 
v_a_3958_ = lean_ctor_get(v___x_3957_, 0);
lean_inc(v_a_3958_);
lean_dec_ref_known(v___x_3957_, 1);
if (v_isShared_3956_ == 0)
{
lean_ctor_set(v___x_3955_, 0, v_a_3958_);
v___x_3960_ = v___x_3955_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3961_; 
v_reuseFailAlloc_3961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3961_, 0, v_a_3958_);
v___x_3960_ = v_reuseFailAlloc_3961_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
v_a_3896_ = v___x_3960_;
goto v___jp_3895_;
}
}
else
{
lean_object* v_a_3962_; lean_object* v___x_3964_; uint8_t v_isShared_3965_; uint8_t v_isSharedCheck_3969_; 
lean_del_object(v___x_3955_);
v_a_3962_ = lean_ctor_get(v___x_3957_, 0);
v_isSharedCheck_3969_ = !lean_is_exclusive(v___x_3957_);
if (v_isSharedCheck_3969_ == 0)
{
v___x_3964_ = v___x_3957_;
v_isShared_3965_ = v_isSharedCheck_3969_;
goto v_resetjp_3963_;
}
else
{
lean_inc(v_a_3962_);
lean_dec(v___x_3957_);
v___x_3964_ = lean_box(0);
v_isShared_3965_ = v_isSharedCheck_3969_;
goto v_resetjp_3963_;
}
v_resetjp_3963_:
{
lean_object* v___x_3967_; 
if (v_isShared_3965_ == 0)
{
v___x_3967_ = v___x_3964_;
goto v_reusejp_3966_;
}
else
{
lean_object* v_reuseFailAlloc_3968_; 
v_reuseFailAlloc_3968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3968_, 0, v_a_3962_);
v___x_3967_ = v_reuseFailAlloc_3968_;
goto v_reusejp_3966_;
}
v_reusejp_3966_:
{
return v___x_3967_;
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
lean_dec_ref(v_xs_3888_);
lean_dec_ref(v_mvars_3887_);
return v___x_3899_;
}
v___jp_3895_:
{
lean_object* v___x_3897_; lean_object* v___x_3898_; 
v___x_3897_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_3897_, 0, v_a_3896_);
v___x_3898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3898_, 0, v___x_3897_);
return v___x_3898_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2___boxed(lean_object* v_k_3971_, lean_object* v_mvars_3972_, lean_object* v_xs_3973_, lean_object* v_e_x27_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_, lean_object* v___y_3978_, lean_object* v___y_3979_){
_start:
{
lean_object* v_res_3980_; 
v_res_3980_ = lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2(v_k_3971_, v_mvars_3972_, v_xs_3973_, v_e_x27_3974_, v___y_3975_, v___y_3976_, v___y_3977_, v___y_3978_);
lean_dec(v___y_3978_);
lean_dec_ref(v___y_3977_);
lean_dec(v___y_3976_);
lean_dec_ref(v___y_3975_);
return v_res_3980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars(lean_object* v_e_3981_, lean_object* v_k_3982_, lean_object* v_a_3983_, lean_object* v_a_3984_, lean_object* v_a_3985_, lean_object* v_a_3986_){
_start:
{
lean_object* v___x_3988_; lean_object* v_a_3989_; uint8_t v___x_3990_; 
v___x_3988_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__1___redArg(v_e_3981_, v_a_3984_);
v_a_3989_ = lean_ctor_get(v___x_3988_, 0);
lean_inc(v_a_3989_);
lean_dec_ref(v___x_3988_);
v___x_3990_ = l_Lean_Expr_hasMVar(v_a_3989_);
if (v___x_3990_ == 0)
{
lean_object* v___x_3991_; 
lean_inc(v_a_3986_);
lean_inc_ref(v_a_3985_);
lean_inc(v_a_3984_);
lean_inc_ref(v_a_3983_);
v___x_3991_ = lean_apply_6(v_k_3982_, v_a_3989_, v_a_3983_, v_a_3984_, v_a_3985_, v_a_3986_, lean_box(0));
return v___x_3991_;
}
else
{
uint8_t v___x_3992_; lean_object* v___x_3993_; 
v___x_3992_ = 0;
v___x_3993_ = l_Lean_Meta_abstractMVars(v_a_3989_, v___x_3992_, v_a_3983_, v_a_3984_, v_a_3985_, v_a_3986_);
if (lean_obj_tag(v___x_3993_) == 0)
{
lean_object* v_a_3994_; lean_object* v_mvars_3995_; lean_object* v_expr_3996_; lean_object* v___f_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; 
v_a_3994_ = lean_ctor_get(v___x_3993_, 0);
lean_inc(v_a_3994_);
lean_dec_ref_known(v___x_3993_, 1);
v_mvars_3995_ = lean_ctor_get(v_a_3994_, 1);
v_expr_3996_ = lean_ctor_get(v_a_3994_, 2);
lean_inc_ref(v_expr_3996_);
lean_inc_ref(v_mvars_3995_);
v___f_3997_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_withAbstractMVars___lam__2___boxed), 9, 2);
lean_closure_set(v___f_3997_, 0, v_k_3982_);
lean_closure_set(v___f_3997_, 1, v_mvars_3995_);
v___x_3998_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v_a_3994_);
lean_dec(v_a_3994_);
v___x_3999_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(v_expr_3996_, v___x_3998_, v___f_3997_, v___x_3992_, v_a_3983_, v_a_3984_, v_a_3985_, v_a_3986_);
return v___x_3999_;
}
else
{
lean_object* v_a_4000_; lean_object* v___x_4002_; uint8_t v_isShared_4003_; uint8_t v_isSharedCheck_4007_; 
lean_dec_ref(v_k_3982_);
v_a_4000_ = lean_ctor_get(v___x_3993_, 0);
v_isSharedCheck_4007_ = !lean_is_exclusive(v___x_3993_);
if (v_isSharedCheck_4007_ == 0)
{
v___x_4002_ = v___x_3993_;
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
else
{
lean_inc(v_a_4000_);
lean_dec(v___x_3993_);
v___x_4002_ = lean_box(0);
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
v_resetjp_4001_:
{
lean_object* v___x_4005_; 
if (v_isShared_4003_ == 0)
{
v___x_4005_ = v___x_4002_;
goto v_reusejp_4004_;
}
else
{
lean_object* v_reuseFailAlloc_4006_; 
v_reuseFailAlloc_4006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4006_, 0, v_a_4000_);
v___x_4005_ = v_reuseFailAlloc_4006_;
goto v_reusejp_4004_;
}
v_reusejp_4004_:
{
return v___x_4005_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_withAbstractMVars___boxed(lean_object* v_e_4008_, lean_object* v_k_4009_, lean_object* v_a_4010_, lean_object* v_a_4011_, lean_object* v_a_4012_, lean_object* v_a_4013_, lean_object* v_a_4014_){
_start:
{
lean_object* v_res_4015_; 
v_res_4015_ = lp_mathlib_ExistsAndEq_withAbstractMVars(v_e_4008_, v_k_4009_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
lean_dec(v_a_4013_);
lean_dec_ref(v_a_4012_);
lean_dec(v_a_4011_);
lean_dec_ref(v_a_4010_);
return v_res_4015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg(lean_object* v_lctx_4016_, lean_object* v_x_4017_, lean_object* v___y_4018_, lean_object* v___y_4019_, lean_object* v___y_4020_, lean_object* v___y_4021_){
_start:
{
lean_object* v_keyedConfig_4023_; uint8_t v_trackZetaDelta_4024_; lean_object* v_zetaDeltaSet_4025_; lean_object* v_localInstances_4026_; lean_object* v_defEqCtx_x3f_4027_; lean_object* v_synthPendingDepth_4028_; lean_object* v_customCanUnfoldPredicate_x3f_4029_; uint8_t v_univApprox_4030_; uint8_t v_inTypeClassResolution_4031_; uint8_t v_cacheInferType_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; 
v_keyedConfig_4023_ = lean_ctor_get(v___y_4018_, 0);
v_trackZetaDelta_4024_ = lean_ctor_get_uint8(v___y_4018_, sizeof(void*)*7);
v_zetaDeltaSet_4025_ = lean_ctor_get(v___y_4018_, 1);
v_localInstances_4026_ = lean_ctor_get(v___y_4018_, 3);
v_defEqCtx_x3f_4027_ = lean_ctor_get(v___y_4018_, 4);
v_synthPendingDepth_4028_ = lean_ctor_get(v___y_4018_, 5);
v_customCanUnfoldPredicate_x3f_4029_ = lean_ctor_get(v___y_4018_, 6);
v_univApprox_4030_ = lean_ctor_get_uint8(v___y_4018_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_4031_ = lean_ctor_get_uint8(v___y_4018_, sizeof(void*)*7 + 2);
v_cacheInferType_4032_ = lean_ctor_get_uint8(v___y_4018_, sizeof(void*)*7 + 3);
lean_inc(v_customCanUnfoldPredicate_x3f_4029_);
lean_inc(v_synthPendingDepth_4028_);
lean_inc(v_defEqCtx_x3f_4027_);
lean_inc_ref(v_localInstances_4026_);
lean_inc(v_zetaDeltaSet_4025_);
lean_inc_ref(v_keyedConfig_4023_);
v___x_4033_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_4033_, 0, v_keyedConfig_4023_);
lean_ctor_set(v___x_4033_, 1, v_zetaDeltaSet_4025_);
lean_ctor_set(v___x_4033_, 2, v_lctx_4016_);
lean_ctor_set(v___x_4033_, 3, v_localInstances_4026_);
lean_ctor_set(v___x_4033_, 4, v_defEqCtx_x3f_4027_);
lean_ctor_set(v___x_4033_, 5, v_synthPendingDepth_4028_);
lean_ctor_set(v___x_4033_, 6, v_customCanUnfoldPredicate_x3f_4029_);
lean_ctor_set_uint8(v___x_4033_, sizeof(void*)*7, v_trackZetaDelta_4024_);
lean_ctor_set_uint8(v___x_4033_, sizeof(void*)*7 + 1, v_univApprox_4030_);
lean_ctor_set_uint8(v___x_4033_, sizeof(void*)*7 + 2, v_inTypeClassResolution_4031_);
lean_ctor_set_uint8(v___x_4033_, sizeof(void*)*7 + 3, v_cacheInferType_4032_);
lean_inc(v___y_4021_);
lean_inc_ref(v___y_4020_);
lean_inc(v___y_4019_);
v___x_4034_ = lean_apply_5(v_x_4017_, v___x_4033_, v___y_4019_, v___y_4020_, v___y_4021_, lean_box(0));
return v___x_4034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg___boxed(lean_object* v_lctx_4035_, lean_object* v_x_4036_, lean_object* v___y_4037_, lean_object* v___y_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_){
_start:
{
lean_object* v_res_4042_; 
v_res_4042_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg(v_lctx_4035_, v_x_4036_, v___y_4037_, v___y_4038_, v___y_4039_, v___y_4040_);
lean_dec(v___y_4040_);
lean_dec_ref(v___y_4039_);
lean_dec(v___y_4038_);
lean_dec_ref(v___y_4037_);
return v_res_4042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0(lean_object* v_00_u03b1_4043_, lean_object* v_lctx_4044_, lean_object* v_x_4045_, lean_object* v___y_4046_, lean_object* v___y_4047_, lean_object* v___y_4048_, lean_object* v___y_4049_){
_start:
{
lean_object* v___x_4051_; 
v___x_4051_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg(v_lctx_4044_, v_x_4045_, v___y_4046_, v___y_4047_, v___y_4048_, v___y_4049_);
return v___x_4051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___boxed(lean_object* v_00_u03b1_4052_, lean_object* v_lctx_4053_, lean_object* v_x_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_, lean_object* v___y_4059_){
_start:
{
lean_object* v_res_4060_; 
v_res_4060_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0(v_00_u03b1_4052_, v_lctx_4053_, v_x_4054_, v___y_4055_, v___y_4056_, v___y_4057_, v___y_4058_);
lean_dec(v___y_4058_);
lean_dec_ref(v___y_4057_);
lean_dec(v___y_4056_);
lean_dec_ref(v___y_4055_);
return v_res_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1(lean_object* v_msg_4061_, lean_object* v___y_4062_, lean_object* v___y_4063_, lean_object* v___y_4064_, lean_object* v___y_4065_){
_start:
{
lean_object* v___f_4067_; lean_object* v___x_3013__overap_4068_; lean_object* v___x_4069_; 
v___f_4067_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__3___closed__0));
v___x_3013__overap_4068_ = lean_panic_fn_borrowed(v___f_4067_, v_msg_4061_);
lean_inc(v___y_4065_);
lean_inc_ref(v___y_4064_);
lean_inc(v___y_4063_);
lean_inc_ref(v___y_4062_);
v___x_4069_ = lean_apply_5(v___x_3013__overap_4068_, v___y_4062_, v___y_4063_, v___y_4064_, v___y_4065_, lean_box(0));
return v___x_4069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1___boxed(lean_object* v_msg_4070_, lean_object* v___y_4071_, lean_object* v___y_4072_, lean_object* v___y_4073_, lean_object* v___y_4074_, lean_object* v___y_4075_){
_start:
{
lean_object* v_res_4076_; 
v_res_4076_ = lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1(v_msg_4070_, v___y_4071_, v___y_4072_, v___y_4073_, v___y_4074_);
lean_dec(v___y_4074_);
lean_dec_ref(v___y_4073_);
lean_dec(v___y_4072_);
lean_dec_ref(v___y_4071_);
return v_res_4076_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2(void){
_start:
{
lean_object* v___x_4080_; lean_object* v___x_4081_; lean_object* v___x_4082_; 
v___x_4080_ = lean_box(0);
v___x_4081_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__1));
v___x_4082_ = l_Lean_Expr_const___override(v___x_4081_, v___x_4080_);
return v___x_4082_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5(void){
_start:
{
lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; 
v___x_4087_ = lean_box(0);
v___x_4088_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__4));
v___x_4089_ = l_Lean_Expr_const___override(v___x_4088_, v___x_4087_);
return v___x_4089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0(lean_object* v_fst_4090_, lean_object* v___x_4091_, lean_object* v_val_4092_, lean_object* v_arg_4093_, lean_object* v_arg_4094_, lean_object* v_val_4095_, lean_object* v_snd_4096_, lean_object* v___x_4097_, lean_object* v___x_4098_, uint8_t v___x_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_){
_start:
{
lean_object* v___x_4105_; 
lean_inc(v_fst_4090_);
v___x_4105_ = lp_mathlib_ExistsAndEq_mkNestedExists(v_fst_4090_, v___x_4091_, v___y_4100_, v___y_4101_, v___y_4102_, v___y_4103_);
if (lean_obj_tag(v___x_4105_) == 0)
{
lean_object* v_a_4106_; lean_object* v___x_4107_; 
v_a_4106_ = lean_ctor_get(v___x_4105_, 0);
lean_inc_n(v_a_4106_, 2);
lean_dec_ref_known(v___x_4105_, 1);
lean_inc(v_val_4095_);
lean_inc(v_fst_4090_);
lean_inc_ref(v_arg_4094_);
lean_inc_ref(v_arg_4093_);
lean_inc(v_val_4092_);
v___x_4107_ = lp_mathlib_ExistsAndEq_mkBeforeToAfter(v_val_4092_, v_arg_4093_, v_arg_4094_, v_a_4106_, v_fst_4090_, v_val_4095_, v___y_4100_, v___y_4101_, v___y_4102_, v___y_4103_);
if (lean_obj_tag(v___x_4107_) == 0)
{
lean_object* v_a_4108_; lean_object* v___x_4109_; 
v_a_4108_ = lean_ctor_get(v___x_4107_, 0);
lean_inc(v_a_4108_);
lean_dec_ref_known(v___x_4107_, 1);
lean_inc(v_a_4106_);
lean_inc_ref(v_arg_4094_);
lean_inc_ref(v_arg_4093_);
lean_inc(v_val_4092_);
v___x_4109_ = lp_mathlib_ExistsAndEq_mkAfterToBefore(v_val_4092_, v_arg_4093_, v_arg_4094_, v_a_4106_, v_snd_4096_, v_fst_4090_, v_val_4095_, v___y_4100_, v___y_4101_, v___y_4102_, v___y_4103_);
if (lean_obj_tag(v___x_4109_) == 0)
{
lean_object* v_a_4110_; lean_object* v___x_4112_; uint8_t v_isShared_4113_; uint8_t v_isSharedCheck_4142_; 
v_a_4110_ = lean_ctor_get(v___x_4109_, 0);
v_isSharedCheck_4142_ = !lean_is_exclusive(v___x_4109_);
if (v_isSharedCheck_4142_ == 0)
{
v___x_4112_ = v___x_4109_;
v_isShared_4113_ = v_isSharedCheck_4142_;
goto v_resetjp_4111_;
}
else
{
lean_inc(v_a_4110_);
lean_dec(v___x_4109_);
v___x_4112_ = lean_box(0);
v_isShared_4113_ = v_isSharedCheck_4142_;
goto v_resetjp_4111_;
}
v_resetjp_4111_:
{
lean_object* v___x_4114_; lean_object* v___x_4115_; lean_object* v___x_4116_; lean_object* v___x_4117_; lean_object* v___x_4118_; lean_object* v___x_4119_; lean_object* v___x_4120_; lean_object* v___x_4121_; uint8_t v___x_4122_; lean_object* v___x_4123_; uint8_t v___x_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; lean_object* v___x_4130_; lean_object* v___x_4131_; lean_object* v___x_4132_; lean_object* v___x_4133_; lean_object* v___x_4134_; lean_object* v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; lean_object* v___x_4138_; lean_object* v___x_4140_; 
v___x_4114_ = lean_box(0);
v___x_4115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4115_, 0, v_val_4092_);
lean_ctor_set(v___x_4115_, 1, v___x_4114_);
v___x_4116_ = l_Lean_Expr_const___override(v___x_4097_, v___x_4115_);
lean_inc_ref(v_arg_4093_);
v___x_4117_ = l_Lean_Expr_app___override(v___x_4116_, v_arg_4093_);
v___x_4118_ = ((lean_object*)(lp_mathlib_ExistsAndEq_mkBeforeToAfter___closed__1));
v___x_4119_ = l_Lean_Expr_bvar___override(v___x_4098_);
v___x_4120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4120_, 0, v___x_4119_);
lean_ctor_set(v___x_4120_, 1, v___x_4114_);
v___x_4121_ = lean_array_mk(v___x_4120_);
v___x_4122_ = 0;
v___x_4123_ = l_Lean_Expr_betaRev(v_arg_4094_, v___x_4121_, v___x_4122_, v___x_4122_);
lean_dec_ref(v___x_4121_);
v___x_4124_ = 0;
v___x_4125_ = l_Lean_Expr_lam___override(v___x_4118_, v_arg_4093_, v___x_4123_, v___x_4124_);
v___x_4126_ = l_Lean_Expr_app___override(v___x_4117_, v___x_4125_);
v___x_4127_ = lean_obj_once(&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2, &lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2_once, _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__2);
lean_inc_ref(v___x_4126_);
v___x_4128_ = l_Lean_Expr_app___override(v___x_4127_, v___x_4126_);
lean_inc_n(v_a_4106_, 2);
v___x_4129_ = l_Lean_Expr_app___override(v___x_4128_, v_a_4106_);
v___x_4130_ = lean_obj_once(&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5, &lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5_once, _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___closed__5);
v___x_4131_ = l_Lean_Expr_app___override(v___x_4130_, v___x_4126_);
v___x_4132_ = l_Lean_Expr_app___override(v___x_4131_, v_a_4106_);
v___x_4133_ = l_Lean_Expr_app___override(v___x_4132_, v_a_4108_);
v___x_4134_ = l_Lean_Expr_app___override(v___x_4133_, v_a_4110_);
v___x_4135_ = l_Lean_Expr_app___override(v___x_4129_, v___x_4134_);
v___x_4136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4136_, 0, v___x_4135_);
v___x_4137_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4137_, 0, v_a_4106_);
lean_ctor_set(v___x_4137_, 1, v___x_4136_);
lean_ctor_set_uint8(v___x_4137_, sizeof(void*)*2, v___x_4099_);
v___x_4138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4138_, 0, v___x_4137_);
if (v_isShared_4113_ == 0)
{
lean_ctor_set(v___x_4112_, 0, v___x_4138_);
v___x_4140_ = v___x_4112_;
goto v_reusejp_4139_;
}
else
{
lean_object* v_reuseFailAlloc_4141_; 
v_reuseFailAlloc_4141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4141_, 0, v___x_4138_);
v___x_4140_ = v_reuseFailAlloc_4141_;
goto v_reusejp_4139_;
}
v_reusejp_4139_:
{
return v___x_4140_;
}
}
}
else
{
lean_object* v_a_4143_; lean_object* v___x_4145_; uint8_t v_isShared_4146_; uint8_t v_isSharedCheck_4150_; 
lean_dec(v_a_4108_);
lean_dec(v_a_4106_);
lean_dec(v___x_4098_);
lean_dec(v___x_4097_);
lean_dec_ref(v_arg_4094_);
lean_dec_ref(v_arg_4093_);
lean_dec(v_val_4092_);
v_a_4143_ = lean_ctor_get(v___x_4109_, 0);
v_isSharedCheck_4150_ = !lean_is_exclusive(v___x_4109_);
if (v_isSharedCheck_4150_ == 0)
{
v___x_4145_ = v___x_4109_;
v_isShared_4146_ = v_isSharedCheck_4150_;
goto v_resetjp_4144_;
}
else
{
lean_inc(v_a_4143_);
lean_dec(v___x_4109_);
v___x_4145_ = lean_box(0);
v_isShared_4146_ = v_isSharedCheck_4150_;
goto v_resetjp_4144_;
}
v_resetjp_4144_:
{
lean_object* v___x_4148_; 
if (v_isShared_4146_ == 0)
{
v___x_4148_ = v___x_4145_;
goto v_reusejp_4147_;
}
else
{
lean_object* v_reuseFailAlloc_4149_; 
v_reuseFailAlloc_4149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4149_, 0, v_a_4143_);
v___x_4148_ = v_reuseFailAlloc_4149_;
goto v_reusejp_4147_;
}
v_reusejp_4147_:
{
return v___x_4148_;
}
}
}
}
else
{
lean_object* v_a_4151_; lean_object* v___x_4153_; uint8_t v_isShared_4154_; uint8_t v_isSharedCheck_4158_; 
lean_dec(v_a_4106_);
lean_dec(v___x_4098_);
lean_dec(v___x_4097_);
lean_dec_ref(v_snd_4096_);
lean_dec(v_val_4095_);
lean_dec_ref(v_arg_4094_);
lean_dec_ref(v_arg_4093_);
lean_dec(v_val_4092_);
lean_dec(v_fst_4090_);
v_a_4151_ = lean_ctor_get(v___x_4107_, 0);
v_isSharedCheck_4158_ = !lean_is_exclusive(v___x_4107_);
if (v_isSharedCheck_4158_ == 0)
{
v___x_4153_ = v___x_4107_;
v_isShared_4154_ = v_isSharedCheck_4158_;
goto v_resetjp_4152_;
}
else
{
lean_inc(v_a_4151_);
lean_dec(v___x_4107_);
v___x_4153_ = lean_box(0);
v_isShared_4154_ = v_isSharedCheck_4158_;
goto v_resetjp_4152_;
}
v_resetjp_4152_:
{
lean_object* v___x_4156_; 
if (v_isShared_4154_ == 0)
{
v___x_4156_ = v___x_4153_;
goto v_reusejp_4155_;
}
else
{
lean_object* v_reuseFailAlloc_4157_; 
v_reuseFailAlloc_4157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4157_, 0, v_a_4151_);
v___x_4156_ = v_reuseFailAlloc_4157_;
goto v_reusejp_4155_;
}
v_reusejp_4155_:
{
return v___x_4156_;
}
}
}
}
else
{
lean_object* v_a_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4166_; 
lean_dec(v___x_4098_);
lean_dec(v___x_4097_);
lean_dec_ref(v_snd_4096_);
lean_dec(v_val_4095_);
lean_dec_ref(v_arg_4094_);
lean_dec_ref(v_arg_4093_);
lean_dec(v_val_4092_);
lean_dec(v_fst_4090_);
v_a_4159_ = lean_ctor_get(v___x_4105_, 0);
v_isSharedCheck_4166_ = !lean_is_exclusive(v___x_4105_);
if (v_isSharedCheck_4166_ == 0)
{
v___x_4161_ = v___x_4105_;
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_a_4159_);
lean_dec(v___x_4105_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
lean_object* v___x_4164_; 
if (v_isShared_4162_ == 0)
{
v___x_4164_ = v___x_4161_;
goto v_reusejp_4163_;
}
else
{
lean_object* v_reuseFailAlloc_4165_; 
v_reuseFailAlloc_4165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4165_, 0, v_a_4159_);
v___x_4164_ = v_reuseFailAlloc_4165_;
goto v_reusejp_4163_;
}
v_reusejp_4163_:
{
return v___x_4164_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___boxed(lean_object* v_fst_4167_, lean_object* v___x_4168_, lean_object* v_val_4169_, lean_object* v_arg_4170_, lean_object* v_arg_4171_, lean_object* v_val_4172_, lean_object* v_snd_4173_, lean_object* v___x_4174_, lean_object* v___x_4175_, lean_object* v___x_4176_, lean_object* v___y_4177_, lean_object* v___y_4178_, lean_object* v___y_4179_, lean_object* v___y_4180_, lean_object* v___y_4181_){
_start:
{
uint8_t v___x_3473__boxed_4182_; lean_object* v_res_4183_; 
v___x_3473__boxed_4182_ = lean_unbox(v___x_4176_);
v_res_4183_ = lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0(v_fst_4167_, v___x_4168_, v_val_4169_, v_arg_4170_, v_arg_4171_, v_val_4172_, v_snd_4173_, v___x_4174_, v___x_4175_, v___x_3473__boxed_4182_, v___y_4177_, v___y_4178_, v___y_4179_, v___y_4180_);
lean_dec(v___y_4180_);
lean_dec_ref(v___y_4179_);
lean_dec(v___y_4178_);
lean_dec_ref(v___y_4177_);
return v_res_4183_;
}
}
static lean_object* _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2(void){
_start:
{
lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; lean_object* v___x_4191_; lean_object* v___x_4192_; 
v___x_4187_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__2));
v___x_4188_ = lean_unsigned_to_nat(39u);
v___x_4189_ = lean_unsigned_to_nat(340u);
v___x_4190_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__1));
v___x_4191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg___lam__2___closed__0));
v___x_4192_ = l_mkPanicMessageWithDecl(v___x_4191_, v___x_4190_, v___x_4189_, v___x_4188_, v___x_4187_);
return v___x_4192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1(lean_object* v___x_4193_, lean_object* v_arg_4194_, lean_object* v_arg_4195_, lean_object* v___x_4196_, uint8_t v___x_4197_, lean_object* v_xs_4198_, lean_object* v_body_4199_, lean_object* v___y_4200_, lean_object* v___y_4201_, lean_object* v___y_4202_, lean_object* v___y_4203_){
_start:
{
lean_object* v___x_4205_; lean_object* v___x_4206_; lean_object* v___x_4207_; 
v___x_4205_ = l_Lean_Expr_constLevels_x21(v___x_4193_);
v___x_4206_ = lean_unsigned_to_nat(0u);
v___x_4207_ = l_List_get_x3fInternal___redArg(v___x_4205_, v___x_4206_);
lean_dec(v___x_4205_);
if (lean_obj_tag(v___x_4207_) == 1)
{
lean_object* v_val_4208_; lean_object* v___x_4210_; uint8_t v_isShared_4211_; uint8_t v_isSharedCheck_4258_; 
v_val_4208_ = lean_ctor_get(v___x_4207_, 0);
v_isSharedCheck_4258_ = !lean_is_exclusive(v___x_4207_);
if (v_isSharedCheck_4258_ == 0)
{
v___x_4210_ = v___x_4207_;
v_isShared_4211_ = v_isSharedCheck_4258_;
goto v_resetjp_4209_;
}
else
{
lean_inc(v_val_4208_);
lean_dec(v___x_4207_);
v___x_4210_ = lean_box(0);
v_isShared_4211_ = v_isSharedCheck_4258_;
goto v_resetjp_4209_;
}
v_resetjp_4209_:
{
lean_object* v___x_4212_; uint8_t v___x_4213_; 
v___x_4212_ = lean_array_get_size(v_xs_4198_);
v___x_4213_ = lean_nat_dec_lt(v___x_4206_, v___x_4212_);
if (v___x_4213_ == 0)
{
lean_object* v___x_4214_; lean_object* v___x_4216_; 
lean_dec(v_val_4208_);
lean_dec_ref(v_body_4199_);
lean_dec(v___x_4196_);
lean_dec_ref(v_arg_4195_);
lean_dec_ref(v_arg_4194_);
v___x_4214_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0));
if (v_isShared_4211_ == 0)
{
lean_ctor_set_tag(v___x_4210_, 0);
lean_ctor_set(v___x_4210_, 0, v___x_4214_);
v___x_4216_ = v___x_4210_;
goto v_reusejp_4215_;
}
else
{
lean_object* v_reuseFailAlloc_4217_; 
v_reuseFailAlloc_4217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4217_, 0, v___x_4214_);
v___x_4216_ = v_reuseFailAlloc_4217_;
goto v_reusejp_4215_;
}
v_reusejp_4215_:
{
return v___x_4216_;
}
}
else
{
lean_object* v___x_4218_; lean_object* v___x_4219_; 
lean_del_object(v___x_4210_);
v___x_4218_ = lean_array_fget_borrowed(v_xs_4198_, v___x_4206_);
lean_inc_ref(v_body_4199_);
v___x_4219_ = lp_mathlib_ExistsAndEq_findEqPath___redArg(v___x_4218_, v_body_4199_, v___y_4201_);
if (lean_obj_tag(v___x_4219_) == 0)
{
lean_object* v_a_4220_; lean_object* v___x_4222_; uint8_t v_isShared_4223_; uint8_t v_isSharedCheck_4249_; 
v_a_4220_ = lean_ctor_get(v___x_4219_, 0);
v_isSharedCheck_4249_ = !lean_is_exclusive(v___x_4219_);
if (v_isSharedCheck_4249_ == 0)
{
v___x_4222_ = v___x_4219_;
v_isShared_4223_ = v_isSharedCheck_4249_;
goto v_resetjp_4221_;
}
else
{
lean_inc(v_a_4220_);
lean_dec(v___x_4219_);
v___x_4222_ = lean_box(0);
v_isShared_4223_ = v_isSharedCheck_4249_;
goto v_resetjp_4221_;
}
v_resetjp_4221_:
{
if (lean_obj_tag(v_a_4220_) == 1)
{
lean_object* v_val_4224_; lean_object* v___x_4225_; 
lean_del_object(v___x_4222_);
v_val_4224_ = lean_ctor_get(v_a_4220_, 0);
lean_inc_n(v_val_4224_, 2);
lean_dec_ref_known(v_a_4220_, 1);
lean_inc(v___x_4218_);
lean_inc(v_val_4208_);
v___x_4225_ = lp_mathlib___private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go___redArg(v_val_4208_, v___x_4218_, v_body_4199_, v_val_4224_, v___y_4200_, v___y_4201_, v___y_4202_, v___y_4203_);
if (lean_obj_tag(v___x_4225_) == 0)
{
lean_object* v_a_4226_; lean_object* v_snd_4227_; lean_object* v_snd_4228_; lean_object* v_fst_4229_; lean_object* v_fst_4230_; lean_object* v_fst_4231_; lean_object* v_snd_4232_; lean_object* v___x_4233_; lean_object* v___x_4234_; lean_object* v___f_4235_; lean_object* v___x_4236_; 
v_a_4226_ = lean_ctor_get(v___x_4225_, 0);
lean_inc(v_a_4226_);
lean_dec_ref_known(v___x_4225_, 1);
v_snd_4227_ = lean_ctor_get(v_a_4226_, 1);
lean_inc(v_snd_4227_);
v_snd_4228_ = lean_ctor_get(v_snd_4227_, 1);
lean_inc(v_snd_4228_);
v_fst_4229_ = lean_ctor_get(v_a_4226_, 0);
lean_inc(v_fst_4229_);
lean_dec(v_a_4226_);
v_fst_4230_ = lean_ctor_get(v_snd_4227_, 0);
lean_inc(v_fst_4230_);
lean_dec(v_snd_4227_);
v_fst_4231_ = lean_ctor_get(v_snd_4228_, 0);
lean_inc(v_fst_4231_);
v_snd_4232_ = lean_ctor_get(v_snd_4228_, 1);
lean_inc(v_snd_4232_);
lean_dec(v_snd_4228_);
lean_inc(v___x_4218_);
v___x_4233_ = l_Lean_Expr_replaceFVar(v_fst_4231_, v___x_4218_, v_snd_4232_);
lean_dec(v_fst_4231_);
v___x_4234_ = lean_box(v___x_4197_);
v___f_4235_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__0___boxed), 15, 10);
lean_closure_set(v___f_4235_, 0, v_fst_4229_);
lean_closure_set(v___f_4235_, 1, v___x_4233_);
lean_closure_set(v___f_4235_, 2, v_val_4208_);
lean_closure_set(v___f_4235_, 3, v_arg_4194_);
lean_closure_set(v___f_4235_, 4, v_arg_4195_);
lean_closure_set(v___f_4235_, 5, v_val_4224_);
lean_closure_set(v___f_4235_, 6, v_snd_4232_);
lean_closure_set(v___f_4235_, 7, v___x_4196_);
lean_closure_set(v___f_4235_, 8, v___x_4206_);
lean_closure_set(v___f_4235_, 9, v___x_4234_);
v___x_4236_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00ExistsAndEq_existsAndEqCore_spec__0___redArg(v_fst_4230_, v___f_4235_, v___y_4200_, v___y_4201_, v___y_4202_, v___y_4203_);
return v___x_4236_;
}
else
{
lean_object* v_a_4237_; lean_object* v___x_4239_; uint8_t v_isShared_4240_; uint8_t v_isSharedCheck_4244_; 
lean_dec(v_val_4224_);
lean_dec(v_val_4208_);
lean_dec(v___x_4196_);
lean_dec_ref(v_arg_4195_);
lean_dec_ref(v_arg_4194_);
v_a_4237_ = lean_ctor_get(v___x_4225_, 0);
v_isSharedCheck_4244_ = !lean_is_exclusive(v___x_4225_);
if (v_isSharedCheck_4244_ == 0)
{
v___x_4239_ = v___x_4225_;
v_isShared_4240_ = v_isSharedCheck_4244_;
goto v_resetjp_4238_;
}
else
{
lean_inc(v_a_4237_);
lean_dec(v___x_4225_);
v___x_4239_ = lean_box(0);
v_isShared_4240_ = v_isSharedCheck_4244_;
goto v_resetjp_4238_;
}
v_resetjp_4238_:
{
lean_object* v___x_4242_; 
if (v_isShared_4240_ == 0)
{
v___x_4242_ = v___x_4239_;
goto v_reusejp_4241_;
}
else
{
lean_object* v_reuseFailAlloc_4243_; 
v_reuseFailAlloc_4243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4243_, 0, v_a_4237_);
v___x_4242_ = v_reuseFailAlloc_4243_;
goto v_reusejp_4241_;
}
v_reusejp_4241_:
{
return v___x_4242_;
}
}
}
}
else
{
lean_object* v___x_4245_; lean_object* v___x_4247_; 
lean_dec(v_a_4220_);
lean_dec(v_val_4208_);
lean_dec_ref(v_body_4199_);
lean_dec(v___x_4196_);
lean_dec_ref(v_arg_4195_);
lean_dec_ref(v_arg_4194_);
v___x_4245_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0));
if (v_isShared_4223_ == 0)
{
lean_ctor_set(v___x_4222_, 0, v___x_4245_);
v___x_4247_ = v___x_4222_;
goto v_reusejp_4246_;
}
else
{
lean_object* v_reuseFailAlloc_4248_; 
v_reuseFailAlloc_4248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4248_, 0, v___x_4245_);
v___x_4247_ = v_reuseFailAlloc_4248_;
goto v_reusejp_4246_;
}
v_reusejp_4246_:
{
return v___x_4247_;
}
}
}
}
else
{
lean_object* v_a_4250_; lean_object* v___x_4252_; uint8_t v_isShared_4253_; uint8_t v_isSharedCheck_4257_; 
lean_dec(v_val_4208_);
lean_dec_ref(v_body_4199_);
lean_dec(v___x_4196_);
lean_dec_ref(v_arg_4195_);
lean_dec_ref(v_arg_4194_);
v_a_4250_ = lean_ctor_get(v___x_4219_, 0);
v_isSharedCheck_4257_ = !lean_is_exclusive(v___x_4219_);
if (v_isSharedCheck_4257_ == 0)
{
v___x_4252_ = v___x_4219_;
v_isShared_4253_ = v_isSharedCheck_4257_;
goto v_resetjp_4251_;
}
else
{
lean_inc(v_a_4250_);
lean_dec(v___x_4219_);
v___x_4252_ = lean_box(0);
v_isShared_4253_ = v_isSharedCheck_4257_;
goto v_resetjp_4251_;
}
v_resetjp_4251_:
{
lean_object* v___x_4255_; 
if (v_isShared_4253_ == 0)
{
v___x_4255_ = v___x_4252_;
goto v_reusejp_4254_;
}
else
{
lean_object* v_reuseFailAlloc_4256_; 
v_reuseFailAlloc_4256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4256_, 0, v_a_4250_);
v___x_4255_ = v_reuseFailAlloc_4256_;
goto v_reusejp_4254_;
}
v_reusejp_4254_:
{
return v___x_4255_;
}
}
}
}
}
}
else
{
lean_object* v___x_4259_; lean_object* v___x_4260_; 
lean_dec(v___x_4207_);
lean_dec_ref(v_body_4199_);
lean_dec(v___x_4196_);
lean_dec_ref(v_arg_4195_);
lean_dec_ref(v_arg_4194_);
v___x_4259_ = lean_obj_once(&lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2, &lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2_once, _init_lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__2);
v___x_4260_ = lp_mathlib_panic___at___00ExistsAndEq_existsAndEqCore_spec__1(v___x_4259_, v___y_4200_, v___y_4201_, v___y_4202_, v___y_4203_);
return v___x_4260_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___boxed(lean_object* v___x_4261_, lean_object* v_arg_4262_, lean_object* v_arg_4263_, lean_object* v___x_4264_, lean_object* v___x_4265_, lean_object* v_xs_4266_, lean_object* v_body_4267_, lean_object* v___y_4268_, lean_object* v___y_4269_, lean_object* v___y_4270_, lean_object* v___y_4271_, lean_object* v___y_4272_){
_start:
{
uint8_t v___x_3672__boxed_4273_; lean_object* v_res_4274_; 
v___x_3672__boxed_4273_ = lean_unbox(v___x_4265_);
v_res_4274_ = lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1(v___x_4261_, v_arg_4262_, v_arg_4263_, v___x_4264_, v___x_3672__boxed_4273_, v_xs_4266_, v_body_4267_, v___y_4268_, v___y_4269_, v___y_4270_, v___y_4271_);
lean_dec(v___y_4271_);
lean_dec_ref(v___y_4270_);
lean_dec(v___y_4269_);
lean_dec_ref(v___y_4268_);
lean_dec_ref(v_xs_4266_);
lean_dec_ref(v___x_4261_);
return v_res_4274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore(lean_object* v_e_4275_, lean_object* v_a_4276_, lean_object* v_a_4277_, lean_object* v_a_4278_, lean_object* v_a_4279_){
_start:
{
lean_object* v___x_4284_; uint8_t v___x_4285_; 
v___x_4284_ = l_Lean_Expr_cleanupAnnotations(v_e_4275_);
v___x_4285_ = l_Lean_Expr_isApp(v___x_4284_);
if (v___x_4285_ == 0)
{
lean_dec_ref(v___x_4284_);
goto v___jp_4281_;
}
else
{
lean_object* v_arg_4286_; lean_object* v___x_4287_; uint8_t v___x_4288_; 
v_arg_4286_ = lean_ctor_get(v___x_4284_, 1);
lean_inc_ref(v_arg_4286_);
v___x_4287_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4284_);
v___x_4288_ = l_Lean_Expr_isApp(v___x_4287_);
if (v___x_4288_ == 0)
{
lean_dec_ref(v___x_4287_);
lean_dec_ref(v_arg_4286_);
goto v___jp_4281_;
}
else
{
lean_object* v_arg_4289_; lean_object* v___x_4290_; lean_object* v___x_4291_; uint8_t v___x_4292_; 
v_arg_4289_ = lean_ctor_get(v___x_4287_, 1);
lean_inc_ref(v_arg_4289_);
v___x_4290_ = l_Lean_Expr_appFnCleanup___redArg(v___x_4287_);
v___x_4291_ = ((lean_object*)(lp_mathlib_ExistsAndEq_findEqPath___redArg___closed__1));
v___x_4292_ = l_Lean_Expr_isConstOf(v___x_4290_, v___x_4291_);
if (v___x_4292_ == 0)
{
lean_dec_ref(v___x_4290_);
lean_dec_ref(v_arg_4289_);
lean_dec_ref(v_arg_4286_);
goto v___jp_4281_;
}
else
{
lean_object* v___x_4293_; lean_object* v___f_4294_; lean_object* v___x_4295_; uint8_t v___x_4296_; lean_object* v___x_4297_; 
v___x_4293_ = lean_box(v___x_4292_);
lean_inc_ref(v_arg_4286_);
v___f_4294_ = lean_alloc_closure((void*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___boxed), 12, 5);
lean_closure_set(v___f_4294_, 0, v___x_4290_);
lean_closure_set(v___f_4294_, 1, v_arg_4289_);
lean_closure_set(v___f_4294_, 2, v_arg_4286_);
lean_closure_set(v___f_4294_, 3, v___x_4291_);
lean_closure_set(v___f_4294_, 4, v___x_4293_);
v___x_4295_ = lean_unsigned_to_nat(1u);
v___x_4296_ = 0;
v___x_4297_ = lp_mathlib_Lean_Meta_lambdaBoundedTelescope___at___00__private_Mathlib_Tactic_Simproc_ExistsAndEq_0__ExistsAndEq_findEq_go_spec__4___redArg(v_arg_4286_, v___x_4295_, v___f_4294_, v___x_4296_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_);
return v___x_4297_;
}
}
}
v___jp_4281_:
{
lean_object* v___x_4282_; lean_object* v___x_4283_; 
v___x_4282_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEqCore___lam__1___closed__0));
v___x_4283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4283_, 0, v___x_4282_);
return v___x_4283_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEqCore___boxed(lean_object* v_e_4298_, lean_object* v_a_4299_, lean_object* v_a_4300_, lean_object* v_a_4301_, lean_object* v_a_4302_, lean_object* v_a_4303_){
_start:
{
lean_object* v_res_4304_; 
v_res_4304_ = lp_mathlib_ExistsAndEq_existsAndEqCore(v_e_4298_, v_a_4299_, v_a_4300_, v_a_4301_, v_a_4302_);
lean_dec(v_a_4302_);
lean_dec_ref(v_a_4301_);
lean_dec(v_a_4300_);
lean_dec_ref(v_a_4299_);
return v_res_4304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___redArg(lean_object* v_e_4306_, lean_object* v_a_4307_, lean_object* v_a_4308_, lean_object* v_a_4309_, lean_object* v_a_4310_){
_start:
{
lean_object* v___x_4312_; lean_object* v___x_4313_; 
v___x_4312_ = ((lean_object*)(lp_mathlib_ExistsAndEq_existsAndEq___redArg___closed__0));
v___x_4313_ = lp_mathlib_ExistsAndEq_withAbstractMVars(v_e_4306_, v___x_4312_, v_a_4307_, v_a_4308_, v_a_4309_, v_a_4310_);
return v___x_4313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___redArg___boxed(lean_object* v_e_4314_, lean_object* v_a_4315_, lean_object* v_a_4316_, lean_object* v_a_4317_, lean_object* v_a_4318_, lean_object* v_a_4319_){
_start:
{
lean_object* v_res_4320_; 
v_res_4320_ = lp_mathlib_ExistsAndEq_existsAndEq___redArg(v_e_4314_, v_a_4315_, v_a_4316_, v_a_4317_, v_a_4318_);
lean_dec(v_a_4318_);
lean_dec_ref(v_a_4317_);
lean_dec(v_a_4316_);
lean_dec_ref(v_a_4315_);
return v_res_4320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq(lean_object* v_e_4321_, lean_object* v_a_4322_, lean_object* v_a_4323_, lean_object* v_a_4324_, lean_object* v_a_4325_, lean_object* v_a_4326_, lean_object* v_a_4327_, lean_object* v_a_4328_){
_start:
{
lean_object* v___x_4330_; 
v___x_4330_ = lp_mathlib_ExistsAndEq_existsAndEq___redArg(v_e_4321_, v_a_4325_, v_a_4326_, v_a_4327_, v_a_4328_);
return v___x_4330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ExistsAndEq_existsAndEq___boxed(lean_object* v_e_4331_, lean_object* v_a_4332_, lean_object* v_a_4333_, lean_object* v_a_4334_, lean_object* v_a_4335_, lean_object* v_a_4336_, lean_object* v_a_4337_, lean_object* v_a_4338_, lean_object* v_a_4339_){
_start:
{
lean_object* v_res_4340_; 
v_res_4340_ = lp_mathlib_ExistsAndEq_existsAndEq(v_e_4331_, v_a_4332_, v_a_4333_, v_a_4334_, v_a_4335_, v_a_4336_, v_a_4337_, v_a_4338_);
lean_dec(v_a_4338_);
lean_dec_ref(v_a_4337_);
lean_dec(v_a_4336_);
lean_dec_ref(v_a_4335_);
lean_dec(v_a_4334_);
lean_dec_ref(v_a_4333_);
lean_dec(v_a_4332_);
return v_res_4340_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin) {
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
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ExistsAndEq_instInhabitedGoTo_default = _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo_default();
lp_mathlib_ExistsAndEq_instInhabitedGoTo = _init_lp_mathlib_ExistsAndEq_instInhabitedGoTo();
lp_mathlib_ExistsAndEq_instInhabitedVarQ = _init_lp_mathlib_ExistsAndEq_instInhabitedVarQ();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedVarQ);
lp_mathlib_ExistsAndEq_instInhabitedHypQ = _init_lp_mathlib_ExistsAndEq_instInhabitedHypQ();
lean_mark_persistent(lp_mathlib_ExistsAndEq_instInhabitedHypQ);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin) {
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
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
}
#ifdef __cplusplus
}
#endif
