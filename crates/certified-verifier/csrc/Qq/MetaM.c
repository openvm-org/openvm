// Lean compiler output
// Module: Qq.MetaM
// Imports: public import Init public meta import Init public import Qq.Delab import Lean.Meta.SynthInstance import Lean.Elab.Term.TermElabM
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
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalDecl___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_isLevelDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalDeclD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_inferTypeQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not a type"};
static const lean_object* lp_Qq_Qq_inferTypeQ___closed__0 = (const lean_object*)&lp_Qq_Qq_inferTypeQ___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_inferTypeQ___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_inferTypeQ___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq_inferTypeQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_inferTypeQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "defEq _"};
static const lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__0_value)}};
static const lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "notDefEq"};
static const lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__2_value;
static const lean_ctor_object lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__2_value)}};
static const lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__3 = (const lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Qq_instReprMaybeDefEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_instReprMaybeDefEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_instReprMaybeDefEq___closed__0 = (const lean_object*)&lp_Qq_Qq_instReprMaybeDefEq___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_assertDefEqQ___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " is not definitionally equal to"};
static const lean_object* lp_Qq_Qq_assertDefEqQ___redArg___closed__0 = (const lean_object*)&lp_Qq_Qq_assertDefEqQ___redArg___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_assertDefEqQ___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_assertDefEqQ___redArg___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Qq_instReprMaybeLevelDefEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___closed__0 = (const lean_object*)&lp_Qq_Qq_instReprMaybeLevelDefEq___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isLevelDefEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_isLevelDefEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_assertLevelDefEqQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " and "};
static const lean_object* lp_Qq_Qq_assertLevelDefEqQ___closed__0 = (const lean_object*)&lp_Qq_Qq_assertLevelDefEqQ___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_assertLevelDefEqQ___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_assertLevelDefEqQ___closed__1;
static const lean_string_object lp_Qq_Qq_assertLevelDefEqQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = " are not definitionally equal"};
static const lean_object* lp_Qq_Qq_assertLevelDefEqQ___closed__2 = (const lean_object*)&lp_Qq_Qq_assertLevelDefEqQ___closed__2_value;
static lean_once_cell_t lp_Qq_Qq_assertLevelDefEqQ___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_assertLevelDefEqQ___closed__3;
LEAN_EXPORT lean_object* lp_Qq_Qq_assertLevelDefEqQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_assertLevelDefEqQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object* v_ty_1_, uint8_t v_kind_2_, lean_object* v_userName_3_, lean_object* v_a_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_9_, 0, v_ty_1_);
v___x_10_ = l_Lean_Meta_mkFreshExprMVar(v___x_9_, v_kind_2_, v_userName_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg___boxed(lean_object* v_ty_11_, lean_object* v_kind_12_, lean_object* v_userName_13_, lean_object* v_a_14_, lean_object* v_a_15_, lean_object* v_a_16_, lean_object* v_a_17_, lean_object* v_a_18_){
_start:
{
uint8_t v_kind_boxed_19_; lean_object* v_res_20_; 
v_kind_boxed_19_ = lean_unbox(v_kind_12_);
v_res_20_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v_ty_11_, v_kind_boxed_19_, v_userName_13_, v_a_14_, v_a_15_, v_a_16_, v_a_17_);
lean_dec(v_a_17_);
lean_dec_ref(v_a_16_);
lean_dec(v_a_15_);
lean_dec_ref(v_a_14_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ(lean_object* v_u_21_, lean_object* v_ty_22_, uint8_t v_kind_23_, lean_object* v_userName_24_, lean_object* v_a_25_, lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v_ty_22_, v_kind_23_, v_userName_24_, v_a_25_, v_a_26_, v_a_27_, v_a_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_mkFreshExprMVarQ___boxed(lean_object* v_u_31_, lean_object* v_ty_32_, lean_object* v_kind_33_, lean_object* v_userName_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_, lean_object* v_a_39_){
_start:
{
uint8_t v_kind_boxed_40_; lean_object* v_res_41_; 
v_kind_boxed_40_ = lean_unbox(v_kind_33_);
v_res_41_ = lp_Qq_Qq_mkFreshExprMVarQ(v_u_31_, v_ty_32_, v_kind_boxed_40_, v_userName_34_, v_a_35_, v_a_36_, v_a_37_, v_a_38_);
lean_dec(v_a_38_);
lean_dec_ref(v_a_37_);
lean_dec(v_a_36_);
lean_dec_ref(v_a_35_);
lean_dec(v_u_31_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ___redArg(lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_name_44_, lean_object* v_00_u03b2_45_, lean_object* v_k_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = l_Lean_Meta_withLocalDeclD___redArg(v_inst_43_, v_inst_42_, v_name_44_, v_00_u03b2_45_, v_k_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ(lean_object* v_n_48_, lean_object* v_u_49_, lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_name_53_, lean_object* v_00_u03b2_54_, lean_object* v_k_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = l_Lean_Meta_withLocalDeclD___redArg(v_inst_52_, v_inst_51_, v_name_53_, v_00_u03b2_54_, v_k_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclDQ___boxed(lean_object* v_n_57_, lean_object* v_u_58_, lean_object* v_00_u03b1_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_name_62_, lean_object* v_00_u03b2_63_, lean_object* v_k_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_Qq_Qq_withLocalDeclDQ(v_n_57_, v_u_58_, v_00_u03b1_59_, v_inst_60_, v_inst_61_, v_name_62_, v_00_u03b2_63_, v_k_64_);
lean_dec(v_u_58_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___redArg(lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_name_68_, uint8_t v_bi_69_, lean_object* v_00_u03b2_70_, lean_object* v_k_71_){
_start:
{
uint8_t v___x_72_; lean_object* v___x_73_; 
v___x_72_ = 0;
v___x_73_ = l_Lean_Meta_withLocalDecl___redArg(v_inst_67_, v_inst_66_, v_name_68_, v_bi_69_, v_00_u03b2_70_, v_k_71_, v___x_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___redArg___boxed(lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_name_76_, lean_object* v_bi_77_, lean_object* v_00_u03b2_78_, lean_object* v_k_79_){
_start:
{
uint8_t v_bi_boxed_80_; lean_object* v_res_81_; 
v_bi_boxed_80_ = lean_unbox(v_bi_77_);
v_res_81_ = lp_Qq_Qq_withLocalDeclQ___redArg(v_inst_74_, v_inst_75_, v_name_76_, v_bi_boxed_80_, v_00_u03b2_78_, v_k_79_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ(lean_object* v_n_82_, lean_object* v_u_83_, lean_object* v_00_u03b1_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_name_87_, uint8_t v_bi_88_, lean_object* v_00_u03b2_89_, lean_object* v_k_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_Qq_Qq_withLocalDeclQ___redArg(v_inst_85_, v_inst_86_, v_name_87_, v_bi_88_, v_00_u03b2_89_, v_k_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_withLocalDeclQ___boxed(lean_object* v_n_92_, lean_object* v_u_93_, lean_object* v_00_u03b1_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_name_97_, lean_object* v_bi_98_, lean_object* v_00_u03b2_99_, lean_object* v_k_100_){
_start:
{
uint8_t v_bi_boxed_101_; lean_object* v_res_102_; 
v_bi_boxed_101_ = lean_unbox(v_bi_98_);
v_res_102_ = lp_Qq_Qq_withLocalDeclQ(v_n_92_, v_u_93_, v_00_u03b1_94_, v_inst_95_, v_inst_96_, v_name_97_, v_bi_boxed_101_, v_00_u03b2_99_, v_k_100_);
lean_dec(v_u_93_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___redArg(lean_object* v_00_u03b1_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = lean_box(0);
v___x_110_ = l_Lean_Meta_synthInstance_x3f(v_00_u03b1_103_, v___x_109_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___redArg___boxed(lean_object* v_00_u03b1_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_, lean_object* v_a_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_Qq_Qq_synthInstanceQ_x3f___redArg(v_00_u03b1_111_, v_a_112_, v_a_113_, v_a_114_, v_a_115_);
lean_dec(v_a_115_);
lean_dec_ref(v_a_114_);
lean_dec(v_a_113_);
lean_dec_ref(v_a_112_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f(lean_object* v_u_118_, lean_object* v_00_u03b1_119_, lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_Qq_Qq_synthInstanceQ_x3f___redArg(v_00_u03b1_119_, v_a_120_, v_a_121_, v_a_122_, v_a_123_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ_x3f___boxed(lean_object* v_u_126_, lean_object* v_00_u03b1_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_Qq_Qq_synthInstanceQ_x3f(v_u_126_, v_00_u03b1_127_, v_a_128_, v_a_129_, v_a_130_, v_a_131_);
lean_dec(v_a_131_);
lean_dec_ref(v_a_130_);
lean_dec(v_a_129_);
lean_dec_ref(v_a_128_);
lean_dec(v_u_126_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg(lean_object* v_00_u03b1_134_, lean_object* v_a_135_, lean_object* v_a_136_, lean_object* v_a_137_, lean_object* v_a_138_){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = lean_box(0);
v___x_141_ = l_Lean_Meta_trySynthInstance(v_00_u03b1_134_, v___x_140_, v_a_135_, v_a_136_, v_a_137_, v_a_138_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___redArg___boxed(lean_object* v_00_u03b1_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v_00_u03b1_142_, v_a_143_, v_a_144_, v_a_145_, v_a_146_);
lean_dec(v_a_146_);
lean_dec_ref(v_a_145_);
lean_dec(v_a_144_);
lean_dec_ref(v_a_143_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ(lean_object* v_u_149_, lean_object* v_00_u03b1_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_, lean_object* v_a_154_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_Qq_Qq_trySynthInstanceQ___redArg(v_00_u03b1_150_, v_a_151_, v_a_152_, v_a_153_, v_a_154_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_trySynthInstanceQ___boxed(lean_object* v_u_157_, lean_object* v_00_u03b1_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_Qq_Qq_trySynthInstanceQ(v_u_157_, v_00_u03b1_158_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
lean_dec(v_a_162_);
lean_dec_ref(v_a_161_);
lean_dec(v_a_160_);
lean_dec_ref(v_a_159_);
lean_dec(v_u_157_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object* v_00_u03b1_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_box(0);
v___x_172_ = l_Lean_Meta_synthInstance(v_00_u03b1_165_, v___x_171_, v_a_166_, v_a_167_, v_a_168_, v_a_169_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___redArg___boxed(lean_object* v_00_u03b1_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_Qq_Qq_synthInstanceQ___redArg(v_00_u03b1_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
lean_dec(v_a_177_);
lean_dec_ref(v_a_176_);
lean_dec(v_a_175_);
lean_dec_ref(v_a_174_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ(lean_object* v_u_180_, lean_object* v_00_u03b1_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_Qq_Qq_synthInstanceQ___redArg(v_00_u03b1_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_synthInstanceQ___boxed(lean_object* v_u_188_, lean_object* v_00_u03b1_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_Qq_Qq_synthInstanceQ(v_u_188_, v_00_u03b1_189_, v_a_190_, v_a_191_, v_a_192_, v_a_193_);
lean_dec(v_a_193_);
lean_dec_ref(v_a_192_);
lean_dec(v_a_191_);
lean_dec_ref(v_a_190_);
lean_dec(v_u_188_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(lean_object* v_e_196_, lean_object* v___y_197_){
_start:
{
uint8_t v___x_199_; 
v___x_199_ = l_Lean_Expr_hasMVar(v_e_196_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; 
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v_e_196_);
return v___x_200_;
}
else
{
lean_object* v___x_201_; lean_object* v_mctx_202_; lean_object* v___x_203_; lean_object* v_fst_204_; lean_object* v_snd_205_; lean_object* v___x_206_; lean_object* v_cache_207_; lean_object* v_zetaDeltaFVarIds_208_; lean_object* v_postponed_209_; lean_object* v_diag_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_219_; 
v___x_201_ = lean_st_ref_get(v___y_197_);
v_mctx_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc_ref(v_mctx_202_);
lean_dec(v___x_201_);
v___x_203_ = l_Lean_instantiateMVarsCore(v_mctx_202_, v_e_196_);
v_fst_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_fst_204_);
v_snd_205_ = lean_ctor_get(v___x_203_, 1);
lean_inc(v_snd_205_);
lean_dec_ref(v___x_203_);
v___x_206_ = lean_st_ref_take(v___y_197_);
v_cache_207_ = lean_ctor_get(v___x_206_, 1);
v_zetaDeltaFVarIds_208_ = lean_ctor_get(v___x_206_, 2);
v_postponed_209_ = lean_ctor_get(v___x_206_, 3);
v_diag_210_ = lean_ctor_get(v___x_206_, 4);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_219_ == 0)
{
lean_object* v_unused_220_; 
v_unused_220_ = lean_ctor_get(v___x_206_, 0);
lean_dec(v_unused_220_);
v___x_212_ = v___x_206_;
v_isShared_213_ = v_isSharedCheck_219_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_diag_210_);
lean_inc(v_postponed_209_);
lean_inc(v_zetaDeltaFVarIds_208_);
lean_inc(v_cache_207_);
lean_dec(v___x_206_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_219_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v_snd_205_);
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_snd_205_);
lean_ctor_set(v_reuseFailAlloc_218_, 1, v_cache_207_);
lean_ctor_set(v_reuseFailAlloc_218_, 2, v_zetaDeltaFVarIds_208_);
lean_ctor_set(v_reuseFailAlloc_218_, 3, v_postponed_209_);
lean_ctor_set(v_reuseFailAlloc_218_, 4, v_diag_210_);
v___x_215_ = v_reuseFailAlloc_218_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_st_ref_set(v___y_197_, v___x_215_);
v___x_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_217_, 0, v_fst_204_);
return v___x_217_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg___boxed(lean_object* v_e_221_, lean_object* v___y_222_, lean_object* v___y_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(v_e_221_, v___y_222_);
lean_dec(v___y_222_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0(lean_object* v_e_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(v_e_225_, v___y_227_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___boxed(lean_object* v_e_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0(v_e_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___redArg(lean_object* v_e_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(v_e_239_, v_a_241_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___redArg___boxed(lean_object* v_e_246_, lean_object* v_a_247_, lean_object* v_a_248_, lean_object* v_a_249_, lean_object* v_a_250_, lean_object* v_a_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_Qq_Qq_instantiateMVarsQ___redArg(v_e_246_, v_a_247_, v_a_248_, v_a_249_, v_a_250_);
lean_dec(v_a_250_);
lean_dec_ref(v_a_249_);
lean_dec(v_a_248_);
lean_dec_ref(v_a_247_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ(lean_object* v_u_253_, lean_object* v_00_u03b1_254_, lean_object* v_e_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_Qq_Lean_instantiateMVars___at___00Qq_instantiateMVarsQ_spec__0___redArg(v_e_255_, v_a_257_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instantiateMVarsQ___boxed(lean_object* v_u_262_, lean_object* v_00_u03b1_263_, lean_object* v_e_264_, lean_object* v_a_265_, lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_Qq_Qq_instantiateMVarsQ(v_u_262_, v_00_u03b1_263_, v_e_264_, v_a_265_, v_a_266_, v_a_267_, v_a_268_);
lean_dec(v_a_268_);
lean_dec_ref(v_a_267_);
lean_dec(v_a_266_);
lean_dec_ref(v_a_265_);
lean_dec_ref(v_00_u03b1_263_);
lean_dec(v_u_262_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___redArg(lean_object* v_stx_271_, lean_object* v_expectedType_272_, uint8_t v_catchExPostpone_273_, uint8_t v_implicitLambda_274_, lean_object* v_errorMsgHeader_x3f_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_, lean_object* v_a_281_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_283_, 0, v_expectedType_272_);
v___x_284_ = l_Lean_Elab_Term_elabTermEnsuringType(v_stx_271_, v___x_283_, v_catchExPostpone_273_, v_implicitLambda_274_, v_errorMsgHeader_x3f_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_, v_a_280_, v_a_281_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___redArg___boxed(lean_object* v_stx_285_, lean_object* v_expectedType_286_, lean_object* v_catchExPostpone_287_, lean_object* v_implicitLambda_288_, lean_object* v_errorMsgHeader_x3f_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_){
_start:
{
uint8_t v_catchExPostpone_boxed_297_; uint8_t v_implicitLambda_boxed_298_; lean_object* v_res_299_; 
v_catchExPostpone_boxed_297_ = lean_unbox(v_catchExPostpone_287_);
v_implicitLambda_boxed_298_ = lean_unbox(v_implicitLambda_288_);
v_res_299_ = lp_Qq_Qq_elabTermEnsuringTypeQ___redArg(v_stx_285_, v_expectedType_286_, v_catchExPostpone_boxed_297_, v_implicitLambda_boxed_298_, v_errorMsgHeader_x3f_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, v_a_294_, v_a_295_);
lean_dec(v_a_295_);
lean_dec_ref(v_a_294_);
lean_dec(v_a_293_);
lean_dec_ref(v_a_292_);
lean_dec(v_a_291_);
lean_dec_ref(v_a_290_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ(lean_object* v_u_300_, lean_object* v_stx_301_, lean_object* v_expectedType_302_, uint8_t v_catchExPostpone_303_, uint8_t v_implicitLambda_304_, lean_object* v_errorMsgHeader_x3f_305_, lean_object* v_a_306_, lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_, lean_object* v_a_311_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_Qq_Qq_elabTermEnsuringTypeQ___redArg(v_stx_301_, v_expectedType_302_, v_catchExPostpone_303_, v_implicitLambda_304_, v_errorMsgHeader_x3f_305_, v_a_306_, v_a_307_, v_a_308_, v_a_309_, v_a_310_, v_a_311_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_elabTermEnsuringTypeQ___boxed(lean_object* v_u_314_, lean_object* v_stx_315_, lean_object* v_expectedType_316_, lean_object* v_catchExPostpone_317_, lean_object* v_implicitLambda_318_, lean_object* v_errorMsgHeader_x3f_319_, lean_object* v_a_320_, lean_object* v_a_321_, lean_object* v_a_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_){
_start:
{
uint8_t v_catchExPostpone_boxed_327_; uint8_t v_implicitLambda_boxed_328_; lean_object* v_res_329_; 
v_catchExPostpone_boxed_327_ = lean_unbox(v_catchExPostpone_317_);
v_implicitLambda_boxed_328_ = lean_unbox(v_implicitLambda_318_);
v_res_329_ = lp_Qq_Qq_elabTermEnsuringTypeQ(v_u_314_, v_stx_315_, v_expectedType_316_, v_catchExPostpone_boxed_327_, v_implicitLambda_boxed_328_, v_errorMsgHeader_x3f_319_, v_a_320_, v_a_321_, v_a_322_, v_a_323_, v_a_324_, v_a_325_);
lean_dec(v_a_325_);
lean_dec_ref(v_a_324_);
lean_dec(v_a_323_);
lean_dec_ref(v_a_322_);
lean_dec(v_a_321_);
lean_dec_ref(v_a_320_);
lean_dec(v_u_314_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0(lean_object* v_msgData_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_){
_start:
{
lean_object* v___x_336_; lean_object* v_env_337_; lean_object* v___x_338_; lean_object* v_mctx_339_; lean_object* v_lctx_340_; lean_object* v_options_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_336_ = lean_st_ref_get(v___y_334_);
v_env_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc_ref(v_env_337_);
lean_dec(v___x_336_);
v___x_338_ = lean_st_ref_get(v___y_332_);
v_mctx_339_ = lean_ctor_get(v___x_338_, 0);
lean_inc_ref(v_mctx_339_);
lean_dec(v___x_338_);
v_lctx_340_ = lean_ctor_get(v___y_331_, 2);
v_options_341_ = lean_ctor_get(v___y_333_, 2);
lean_inc_ref(v_options_341_);
lean_inc_ref(v_lctx_340_);
v___x_342_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_342_, 0, v_env_337_);
lean_ctor_set(v___x_342_, 1, v_mctx_339_);
lean_ctor_set(v___x_342_, 2, v_lctx_340_);
lean_ctor_set(v___x_342_, 3, v_options_341_);
v___x_343_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
lean_ctor_set(v___x_343_, 1, v_msgData_330_);
v___x_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_344_, 0, v___x_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0___boxed(lean_object* v_msgData_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0(v_msgData_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(lean_object* v_msg_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_ref_358_; lean_object* v___x_359_; lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_368_; 
v_ref_358_ = lean_ctor_get(v___y_355_, 5);
v___x_359_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_inferTypeQ_spec__0_spec__0(v_msg_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
v_a_360_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_368_ == 0)
{
v___x_362_ = v___x_359_;
v_isShared_363_ = v_isSharedCheck_368_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_359_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_368_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_364_; lean_object* v___x_366_; 
lean_inc(v_ref_358_);
v___x_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_364_, 0, v_ref_358_);
lean_ctor_set(v___x_364_, 1, v_a_360_);
if (v_isShared_363_ == 0)
{
lean_ctor_set_tag(v___x_362_, 1);
lean_ctor_set(v___x_362_, 0, v___x_364_);
v___x_366_ = v___x_362_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 1, 0);
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
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg___boxed(lean_object* v_msg_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(v_msg_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
lean_dec(v___y_373_);
lean_dec_ref(v___y_372_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
return v_res_375_;
}
}
static lean_object* _init_lp_Qq_Qq_inferTypeQ___closed__1(void){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_377_ = ((lean_object*)(lp_Qq_Qq_inferTypeQ___closed__0));
v___x_378_ = l_Lean_stringToMessageData(v___x_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_inferTypeQ(lean_object* v_e_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_){
_start:
{
lean_object* v___x_385_; 
lean_inc(v_a_383_);
lean_inc_ref(v_a_382_);
lean_inc(v_a_381_);
lean_inc_ref(v_a_380_);
lean_inc_ref(v_e_379_);
v___x_385_ = lean_infer_type(v_e_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
if (lean_obj_tag(v___x_385_) == 0)
{
lean_object* v_a_386_; lean_object* v___x_387_; 
v_a_386_ = lean_ctor_get(v___x_385_, 0);
lean_inc_n(v_a_386_, 2);
lean_dec_ref_known(v___x_385_, 1);
lean_inc(v_a_383_);
lean_inc_ref(v_a_382_);
lean_inc(v_a_381_);
lean_inc_ref(v_a_380_);
v___x_387_ = lean_infer_type(v_a_386_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; lean_object* v___x_389_; 
v_a_388_ = lean_ctor_get(v___x_387_, 0);
lean_inc(v_a_388_);
lean_dec_ref_known(v___x_387_, 1);
lean_inc(v_a_383_);
lean_inc_ref(v_a_382_);
lean_inc(v_a_381_);
lean_inc_ref(v_a_380_);
v___x_389_ = lean_whnf(v_a_388_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
if (lean_obj_tag(v___x_389_) == 0)
{
lean_object* v_a_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_404_; 
v_a_390_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_404_ == 0)
{
v___x_392_ = v___x_389_;
v_isShared_393_ = v_isSharedCheck_404_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_a_390_);
lean_dec(v___x_389_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_404_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
if (lean_obj_tag(v_a_390_) == 3)
{
lean_object* v_u_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_398_; 
v_u_394_ = lean_ctor_get(v_a_390_, 0);
lean_inc(v_u_394_);
lean_dec_ref_known(v_a_390_, 1);
v___x_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_395_, 0, v_a_386_);
lean_ctor_set(v___x_395_, 1, v_e_379_);
v___x_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_396_, 0, v_u_394_);
lean_ctor_set(v___x_396_, 1, v___x_395_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 0, v___x_396_);
v___x_398_ = v___x_392_;
goto v_reusejp_397_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v___x_396_);
v___x_398_ = v_reuseFailAlloc_399_;
goto v_reusejp_397_;
}
v_reusejp_397_:
{
return v___x_398_;
}
}
else
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
lean_del_object(v___x_392_);
lean_dec(v_a_390_);
lean_dec_ref(v_e_379_);
v___x_400_ = lean_obj_once(&lp_Qq_Qq_inferTypeQ___closed__1, &lp_Qq_Qq_inferTypeQ___closed__1_once, _init_lp_Qq_Qq_inferTypeQ___closed__1);
v___x_401_ = l_Lean_indentExpr(v_a_386_);
v___x_402_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_402_, 0, v___x_400_);
lean_ctor_set(v___x_402_, 1, v___x_401_);
v___x_403_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(v___x_402_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
return v___x_403_;
}
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v_a_386_);
lean_dec_ref(v_e_379_);
v_a_405_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_389_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_389_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
else
{
lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_420_; 
lean_dec(v_a_386_);
lean_dec_ref(v_e_379_);
v_a_413_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_387_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_387_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_418_; 
if (v_isShared_416_ == 0)
{
v___x_418_ = v___x_415_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_a_413_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
else
{
lean_object* v_a_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_428_; 
lean_dec_ref(v_e_379_);
v_a_421_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_428_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_428_ == 0)
{
v___x_423_ = v___x_385_;
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_a_421_);
lean_dec(v___x_385_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v___x_426_; 
if (v_isShared_424_ == 0)
{
v___x_426_ = v___x_423_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v_a_421_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
return v___x_426_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_inferTypeQ___boxed(lean_object* v_e_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_Qq_Qq_inferTypeQ(v_e_429_, v_a_430_, v_a_431_, v_a_432_, v_a_433_);
lean_dec(v_a_433_);
lean_dec_ref(v_a_432_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_430_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0(lean_object* v_00_u03b1_436_, lean_object* v_msg_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(v_msg_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___boxed(lean_object* v_00_u03b1_444_, lean_object* v_msg_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0(v_00_u03b1_444_, v_msg_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___redArg(lean_object* v_e_452_, lean_object* v_ty_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_){
_start:
{
lean_object* v___x_459_; 
lean_inc(v_a_457_);
lean_inc_ref(v_a_456_);
lean_inc(v_a_455_);
lean_inc_ref(v_a_454_);
lean_inc_ref(v_e_452_);
v___x_459_ = lean_infer_type(v_e_452_, v_a_454_, v_a_455_, v_a_456_, v_a_457_);
if (lean_obj_tag(v___x_459_) == 0)
{
lean_object* v_a_460_; lean_object* v___x_461_; 
v_a_460_ = lean_ctor_get(v___x_459_, 0);
lean_inc(v_a_460_);
lean_dec_ref_known(v___x_459_, 1);
v___x_461_ = l_Lean_Meta_isExprDefEq(v_a_460_, v_ty_453_, v_a_454_, v_a_455_, v_a_456_, v_a_457_);
if (lean_obj_tag(v___x_461_) == 0)
{
lean_object* v_a_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_475_; 
v_a_462_ = lean_ctor_get(v___x_461_, 0);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_461_);
if (v_isSharedCheck_475_ == 0)
{
v___x_464_ = v___x_461_;
v_isShared_465_ = v_isSharedCheck_475_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_a_462_);
lean_dec(v___x_461_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_475_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
uint8_t v___x_466_; 
v___x_466_ = lean_unbox(v_a_462_);
lean_dec(v_a_462_);
if (v___x_466_ == 0)
{
lean_object* v___x_467_; lean_object* v___x_469_; 
lean_dec_ref(v_e_452_);
v___x_467_ = lean_box(0);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 0, v___x_467_);
v___x_469_ = v___x_464_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_467_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
else
{
lean_object* v___x_471_; lean_object* v___x_473_; 
v___x_471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_471_, 0, v_e_452_);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 0, v___x_471_);
v___x_473_ = v___x_464_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v___x_471_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
}
}
else
{
lean_object* v_a_476_; lean_object* v___x_478_; uint8_t v_isShared_479_; uint8_t v_isSharedCheck_483_; 
lean_dec_ref(v_e_452_);
v_a_476_ = lean_ctor_get(v___x_461_, 0);
v_isSharedCheck_483_ = !lean_is_exclusive(v___x_461_);
if (v_isSharedCheck_483_ == 0)
{
v___x_478_ = v___x_461_;
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
else
{
lean_inc(v_a_476_);
lean_dec(v___x_461_);
v___x_478_ = lean_box(0);
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
v_resetjp_477_:
{
lean_object* v___x_481_; 
if (v_isShared_479_ == 0)
{
v___x_481_ = v___x_478_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v_a_476_);
v___x_481_ = v_reuseFailAlloc_482_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
return v___x_481_;
}
}
}
}
else
{
lean_object* v_a_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_491_; 
lean_dec_ref(v_ty_453_);
lean_dec_ref(v_e_452_);
v_a_484_ = lean_ctor_get(v___x_459_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_459_);
if (v_isSharedCheck_491_ == 0)
{
v___x_486_ = v___x_459_;
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_a_484_);
lean_dec(v___x_459_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v___x_489_; 
if (v_isShared_487_ == 0)
{
v___x_489_ = v___x_486_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_a_484_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___redArg___boxed(lean_object* v_e_492_, lean_object* v_ty_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_Qq_Qq_checkTypeQ___redArg(v_e_492_, v_ty_493_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
lean_dec(v_a_497_);
lean_dec_ref(v_a_496_);
lean_dec(v_a_495_);
lean_dec_ref(v_a_494_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ(lean_object* v_u_500_, lean_object* v_e_501_, lean_object* v_ty_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_Qq_Qq_checkTypeQ___redArg(v_e_501_, v_ty_502_, v_a_503_, v_a_504_, v_a_505_, v_a_506_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_checkTypeQ___boxed(lean_object* v_u_509_, lean_object* v_e_510_, lean_object* v_ty_511_, lean_object* v_a_512_, lean_object* v_a_513_, lean_object* v_a_514_, lean_object* v_a_515_, lean_object* v_a_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_Qq_Qq_checkTypeQ(v_u_509_, v_e_510_, v_ty_511_, v_a_512_, v_a_513_, v_a_514_, v_a_515_);
lean_dec(v_a_515_);
lean_dec_ref(v_a_514_);
lean_dec(v_a_513_);
lean_dec_ref(v_a_512_);
lean_dec(v_u_509_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg(uint8_t v_x_518_){
_start:
{
if (v_x_518_ == 0)
{
lean_object* v___x_519_; 
v___x_519_ = lean_unsigned_to_nat(0u);
return v___x_519_;
}
else
{
lean_object* v___x_520_; 
v___x_520_ = lean_unsigned_to_nat(1u);
return v___x_520_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg___boxed(lean_object* v_x_521_){
_start:
{
uint8_t v_x_boxed_522_; lean_object* v_res_523_; 
v_x_boxed_522_ = lean_unbox(v_x_521_);
v_res_523_ = lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg(v_x_boxed_522_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx(lean_object* v_u_524_, lean_object* v_00_u03b1_525_, lean_object* v_a_526_, lean_object* v_b_527_, uint8_t v_x_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_Qq_Qq_MaybeDefEq_ctorIdx___redArg(v_x_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorIdx___boxed(lean_object* v_u_530_, lean_object* v_00_u03b1_531_, lean_object* v_a_532_, lean_object* v_b_533_, lean_object* v_x_534_){
_start:
{
uint8_t v_x_boxed_535_; lean_object* v_res_536_; 
v_x_boxed_535_ = lean_unbox(v_x_534_);
v_res_536_ = lp_Qq_Qq_MaybeDefEq_ctorIdx(v_u_530_, v_00_u03b1_531_, v_a_532_, v_b_533_, v_x_boxed_535_);
lean_dec_ref(v_b_533_);
lean_dec_ref(v_a_532_);
lean_dec_ref(v_00_u03b1_531_);
lean_dec(v_u_530_);
return v_res_536_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(uint8_t v_t_537_, lean_object* v_k_538_){
_start:
{
if (v_t_537_ == 0)
{
lean_object* v___x_539_; 
v___x_539_ = lean_apply_1(v_k_538_, lean_box(0));
return v___x_539_;
}
else
{
return v_k_538_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___redArg___boxed(lean_object* v_t_540_, lean_object* v_k_541_){
_start:
{
uint8_t v_t_boxed_542_; lean_object* v_res_543_; 
v_t_boxed_542_ = lean_unbox(v_t_540_);
v_res_543_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_boxed_542_, v_k_541_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim(lean_object* v_u_544_, lean_object* v_00_u03b1_545_, lean_object* v_a_546_, lean_object* v_b_547_, lean_object* v_motive_548_, lean_object* v_ctorIdx_549_, uint8_t v_t_550_, lean_object* v_h_551_, lean_object* v_k_552_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_550_, v_k_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_ctorElim___boxed(lean_object* v_u_554_, lean_object* v_00_u03b1_555_, lean_object* v_a_556_, lean_object* v_b_557_, lean_object* v_motive_558_, lean_object* v_ctorIdx_559_, lean_object* v_t_560_, lean_object* v_h_561_, lean_object* v_k_562_){
_start:
{
uint8_t v_t_boxed_563_; lean_object* v_res_564_; 
v_t_boxed_563_ = lean_unbox(v_t_560_);
v_res_564_ = lp_Qq_Qq_MaybeDefEq_ctorElim(v_u_554_, v_00_u03b1_555_, v_a_556_, v_b_557_, v_motive_558_, v_ctorIdx_559_, v_t_boxed_563_, v_h_561_, v_k_562_);
lean_dec(v_ctorIdx_559_);
lean_dec_ref(v_b_557_);
lean_dec_ref(v_a_556_);
lean_dec_ref(v_00_u03b1_555_);
lean_dec(v_u_554_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___redArg(uint8_t v_t_565_, lean_object* v_defEq_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_565_, v_defEq_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___redArg___boxed(lean_object* v_t_568_, lean_object* v_defEq_569_){
_start:
{
uint8_t v_t_boxed_570_; lean_object* v_res_571_; 
v_t_boxed_570_ = lean_unbox(v_t_568_);
v_res_571_ = lp_Qq_Qq_MaybeDefEq_defEq_elim___redArg(v_t_boxed_570_, v_defEq_569_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim(lean_object* v_u_572_, lean_object* v_00_u03b1_573_, lean_object* v_a_574_, lean_object* v_b_575_, lean_object* v_motive_576_, uint8_t v_t_577_, lean_object* v_h_578_, lean_object* v_defEq_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_577_, v_defEq_579_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_defEq_elim___boxed(lean_object* v_u_581_, lean_object* v_00_u03b1_582_, lean_object* v_a_583_, lean_object* v_b_584_, lean_object* v_motive_585_, lean_object* v_t_586_, lean_object* v_h_587_, lean_object* v_defEq_588_){
_start:
{
uint8_t v_t_boxed_589_; lean_object* v_res_590_; 
v_t_boxed_589_ = lean_unbox(v_t_586_);
v_res_590_ = lp_Qq_Qq_MaybeDefEq_defEq_elim(v_u_581_, v_00_u03b1_582_, v_a_583_, v_b_584_, v_motive_585_, v_t_boxed_589_, v_h_587_, v_defEq_588_);
lean_dec_ref(v_b_584_);
lean_dec_ref(v_a_583_);
lean_dec_ref(v_00_u03b1_582_);
lean_dec(v_u_581_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___redArg(uint8_t v_t_591_, lean_object* v_notDefEq_592_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_591_, v_notDefEq_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___redArg___boxed(lean_object* v_t_594_, lean_object* v_notDefEq_595_){
_start:
{
uint8_t v_t_boxed_596_; lean_object* v_res_597_; 
v_t_boxed_596_ = lean_unbox(v_t_594_);
v_res_597_ = lp_Qq_Qq_MaybeDefEq_notDefEq_elim___redArg(v_t_boxed_596_, v_notDefEq_595_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim(lean_object* v_u_598_, lean_object* v_00_u03b1_599_, lean_object* v_a_600_, lean_object* v_b_601_, lean_object* v_motive_602_, uint8_t v_t_603_, lean_object* v_h_604_, lean_object* v_notDefEq_605_){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lp_Qq_Qq_MaybeDefEq_ctorElim___redArg(v_t_603_, v_notDefEq_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeDefEq_notDefEq_elim___boxed(lean_object* v_u_607_, lean_object* v_00_u03b1_608_, lean_object* v_a_609_, lean_object* v_b_610_, lean_object* v_motive_611_, lean_object* v_t_612_, lean_object* v_h_613_, lean_object* v_notDefEq_614_){
_start:
{
uint8_t v_t_boxed_615_; lean_object* v_res_616_; 
v_t_boxed_615_ = lean_unbox(v_t_612_);
v_res_616_ = lp_Qq_Qq_MaybeDefEq_notDefEq_elim(v_u_607_, v_00_u03b1_608_, v_a_609_, v_b_610_, v_motive_611_, v_t_boxed_615_, v_h_613_, v_notDefEq_614_);
lean_dec_ref(v_b_610_);
lean_dec_ref(v_a_609_);
lean_dec_ref(v_00_u03b1_608_);
lean_dec(v_u_607_);
return v_res_616_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0(uint8_t v_x_623_, lean_object* v_x_624_){
_start:
{
if (v_x_623_ == 0)
{
lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_625_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__1));
v___x_626_ = l_Repr_addAppParen(v___x_625_, v_x_624_);
return v___x_626_;
}
else
{
lean_object* v___x_627_; 
v___x_627_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__3));
return v___x_627_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___lam__0___boxed(lean_object* v_x_628_, lean_object* v_x_629_){
_start:
{
uint8_t v_x_48__boxed_630_; lean_object* v_res_631_; 
v_x_48__boxed_630_ = lean_unbox(v_x_628_);
v_res_631_ = lp_Qq_Qq_instReprMaybeDefEq___lam__0(v_x_48__boxed_630_, v_x_629_);
lean_dec(v_x_629_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq(lean_object* v_u_633_, lean_object* v_00_u03b1_634_, lean_object* v_a_635_, lean_object* v_b_636_){
_start:
{
lean_object* v___f_637_; 
v___f_637_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeDefEq___closed__0));
return v___f_637_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeDefEq___boxed(lean_object* v_u_638_, lean_object* v_00_u03b1_639_, lean_object* v_a_640_, lean_object* v_b_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_Qq_Qq_instReprMaybeDefEq(v_u_638_, v_00_u03b1_639_, v_a_640_, v_b_641_);
lean_dec_ref(v_b_641_);
lean_dec_ref(v_a_640_);
lean_dec_ref(v_00_u03b1_639_);
lean_dec(v_u_638_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___redArg(lean_object* v_a_643_, lean_object* v_b_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = l_Lean_Meta_isExprDefEq(v_a_643_, v_b_644_, v_a_645_, v_a_646_, v_a_647_, v_a_648_);
if (lean_obj_tag(v___x_650_) == 0)
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_666_; 
v_a_651_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_666_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_666_ == 0)
{
v___x_653_ = v___x_650_;
v_isShared_654_ = v_isSharedCheck_666_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_650_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_666_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
uint8_t v___x_655_; 
v___x_655_ = lean_unbox(v_a_651_);
lean_dec(v_a_651_);
if (v___x_655_ == 0)
{
uint8_t v___x_656_; lean_object* v___x_657_; lean_object* v___x_659_; 
v___x_656_ = 1;
v___x_657_ = lean_box(v___x_656_);
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_657_);
v___x_659_ = v___x_653_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v___x_657_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
else
{
uint8_t v___x_661_; lean_object* v___x_662_; lean_object* v___x_664_; 
v___x_661_ = 0;
v___x_662_ = lean_box(v___x_661_);
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_662_);
v___x_664_ = v___x_653_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v___x_662_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
}
else
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_674_; 
v_a_667_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_674_ == 0)
{
v___x_669_ = v___x_650_;
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___x_650_);
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
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___redArg___boxed(lean_object* v_a_675_, lean_object* v_b_676_, lean_object* v_a_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_, lean_object* v_a_681_){
_start:
{
lean_object* v_res_682_; 
v_res_682_ = lp_Qq_Qq_isDefEqQ___redArg(v_a_675_, v_b_676_, v_a_677_, v_a_678_, v_a_679_, v_a_680_);
lean_dec(v_a_680_);
lean_dec_ref(v_a_679_);
lean_dec(v_a_678_);
lean_dec_ref(v_a_677_);
return v_res_682_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ(lean_object* v_u_683_, lean_object* v_00_u03b1_684_, lean_object* v_a_685_, lean_object* v_b_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_Qq_Qq_isDefEqQ___redArg(v_a_685_, v_b_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_isDefEqQ___boxed(lean_object* v_u_693_, lean_object* v_00_u03b1_694_, lean_object* v_a_695_, lean_object* v_b_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_Qq_Qq_isDefEqQ(v_u_693_, v_00_u03b1_694_, v_a_695_, v_b_696_, v_a_697_, v_a_698_, v_a_699_, v_a_700_);
lean_dec(v_a_700_);
lean_dec_ref(v_a_699_);
lean_dec(v_a_698_);
lean_dec_ref(v_a_697_);
lean_dec_ref(v_00_u03b1_694_);
lean_dec(v_u_693_);
return v_res_702_;
}
}
static lean_object* _init_lp_Qq_Qq_assertDefEqQ___redArg___closed__1(void){
_start:
{
lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_704_ = ((lean_object*)(lp_Qq_Qq_assertDefEqQ___redArg___closed__0));
v___x_705_ = l_Lean_stringToMessageData(v___x_704_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___redArg(lean_object* v_a_706_, lean_object* v_b_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_){
_start:
{
lean_object* v___x_713_; 
lean_inc_ref(v_b_707_);
lean_inc_ref(v_a_706_);
v___x_713_ = lp_Qq_Qq_isDefEqQ___redArg(v_a_706_, v_b_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_);
if (lean_obj_tag(v___x_713_) == 0)
{
lean_object* v_a_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_728_; 
v_a_714_ = lean_ctor_get(v___x_713_, 0);
v_isSharedCheck_728_ = !lean_is_exclusive(v___x_713_);
if (v_isSharedCheck_728_ == 0)
{
v___x_716_ = v___x_713_;
v_isShared_717_ = v_isSharedCheck_728_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_a_714_);
lean_dec(v___x_713_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_728_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
uint8_t v___x_718_; 
v___x_718_ = lean_unbox(v_a_714_);
lean_dec(v_a_714_);
if (v___x_718_ == 0)
{
lean_object* v___x_720_; 
lean_dec_ref(v_b_707_);
lean_dec_ref(v_a_706_);
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 0, lean_box(0));
v___x_720_ = v___x_716_;
goto v_reusejp_719_;
}
else
{
lean_object* v_reuseFailAlloc_721_; 
v_reuseFailAlloc_721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_721_, 0, lean_box(0));
v___x_720_ = v_reuseFailAlloc_721_;
goto v_reusejp_719_;
}
v_reusejp_719_:
{
return v___x_720_;
}
}
else
{
lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; 
lean_del_object(v___x_716_);
v___x_722_ = l_Lean_MessageData_ofExpr(v_a_706_);
v___x_723_ = lean_obj_once(&lp_Qq_Qq_assertDefEqQ___redArg___closed__1, &lp_Qq_Qq_assertDefEqQ___redArg___closed__1_once, _init_lp_Qq_Qq_assertDefEqQ___redArg___closed__1);
v___x_724_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_722_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = l_Lean_indentExpr(v_b_707_);
v___x_726_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_726_, 0, v___x_724_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
v___x_727_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(v___x_726_, v_a_708_, v_a_709_, v_a_710_, v_a_711_);
return v___x_727_;
}
}
}
else
{
lean_object* v_a_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_736_; 
lean_dec_ref(v_b_707_);
lean_dec_ref(v_a_706_);
v_a_729_ = lean_ctor_get(v___x_713_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_713_);
if (v_isSharedCheck_736_ == 0)
{
v___x_731_ = v___x_713_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_a_729_);
lean_dec(v___x_713_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___x_734_; 
if (v_isShared_732_ == 0)
{
v___x_734_ = v___x_731_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v_a_729_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___redArg___boxed(lean_object* v_a_737_, lean_object* v_b_738_, lean_object* v_a_739_, lean_object* v_a_740_, lean_object* v_a_741_, lean_object* v_a_742_, lean_object* v_a_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_Qq_Qq_assertDefEqQ___redArg(v_a_737_, v_b_738_, v_a_739_, v_a_740_, v_a_741_, v_a_742_);
lean_dec(v_a_742_);
lean_dec_ref(v_a_741_);
lean_dec(v_a_740_);
lean_dec_ref(v_a_739_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ(lean_object* v_u_745_, lean_object* v_00_u03b1_746_, lean_object* v_a_747_, lean_object* v_b_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_){
_start:
{
lean_object* v___x_754_; 
v___x_754_ = lp_Qq_Qq_assertDefEqQ___redArg(v_a_747_, v_b_748_, v_a_749_, v_a_750_, v_a_751_, v_a_752_);
return v___x_754_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertDefEqQ___boxed(lean_object* v_u_755_, lean_object* v_00_u03b1_756_, lean_object* v_a_757_, lean_object* v_b_758_, lean_object* v_a_759_, lean_object* v_a_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v_a_763_){
_start:
{
lean_object* v_res_764_; 
v_res_764_ = lp_Qq_Qq_assertDefEqQ(v_u_755_, v_00_u03b1_756_, v_a_757_, v_b_758_, v_a_759_, v_a_760_, v_a_761_, v_a_762_);
lean_dec(v_a_762_);
lean_dec_ref(v_a_761_);
lean_dec(v_a_760_);
lean_dec_ref(v_a_759_);
lean_dec_ref(v_00_u03b1_756_);
lean_dec(v_u_755_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg(uint8_t v_x_765_){
_start:
{
if (v_x_765_ == 0)
{
lean_object* v___x_766_; 
v___x_766_ = lean_unsigned_to_nat(0u);
return v___x_766_;
}
else
{
lean_object* v___x_767_; 
v___x_767_ = lean_unsigned_to_nat(1u);
return v___x_767_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg___boxed(lean_object* v_x_768_){
_start:
{
uint8_t v_x_boxed_769_; lean_object* v_res_770_; 
v_x_boxed_769_ = lean_unbox(v_x_768_);
v_res_770_ = lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg(v_x_boxed_769_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx(lean_object* v_u_771_, lean_object* v_v_772_, uint8_t v_x_773_){
_start:
{
lean_object* v___x_774_; 
v___x_774_ = lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___redArg(v_x_773_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorIdx___boxed(lean_object* v_u_775_, lean_object* v_v_776_, lean_object* v_x_777_){
_start:
{
uint8_t v_x_boxed_778_; lean_object* v_res_779_; 
v_x_boxed_778_ = lean_unbox(v_x_777_);
v_res_779_ = lp_Qq_Qq_MaybeLevelDefEq_ctorIdx(v_u_775_, v_v_776_, v_x_boxed_778_);
lean_dec(v_v_776_);
lean_dec(v_u_775_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(uint8_t v_t_780_, lean_object* v_k_781_){
_start:
{
if (v_t_780_ == 0)
{
lean_object* v___x_782_; 
v___x_782_ = lean_apply_1(v_k_781_, lean_box(0));
return v___x_782_;
}
else
{
return v_k_781_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg___boxed(lean_object* v_t_783_, lean_object* v_k_784_){
_start:
{
uint8_t v_t_boxed_785_; lean_object* v_res_786_; 
v_t_boxed_785_ = lean_unbox(v_t_783_);
v_res_786_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_boxed_785_, v_k_784_);
return v_res_786_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim(lean_object* v_u_787_, lean_object* v_v_788_, lean_object* v_motive_789_, lean_object* v_ctorIdx_790_, uint8_t v_t_791_, lean_object* v_h_792_, lean_object* v_k_793_){
_start:
{
lean_object* v___x_794_; 
v___x_794_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_791_, v_k_793_);
return v___x_794_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_ctorElim___boxed(lean_object* v_u_795_, lean_object* v_v_796_, lean_object* v_motive_797_, lean_object* v_ctorIdx_798_, lean_object* v_t_799_, lean_object* v_h_800_, lean_object* v_k_801_){
_start:
{
uint8_t v_t_boxed_802_; lean_object* v_res_803_; 
v_t_boxed_802_ = lean_unbox(v_t_799_);
v_res_803_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim(v_u_795_, v_v_796_, v_motive_797_, v_ctorIdx_798_, v_t_boxed_802_, v_h_800_, v_k_801_);
lean_dec(v_ctorIdx_798_);
lean_dec(v_v_796_);
lean_dec(v_u_795_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___redArg(uint8_t v_t_804_, lean_object* v_defEq_805_){
_start:
{
lean_object* v___x_806_; 
v___x_806_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_804_, v_defEq_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___redArg___boxed(lean_object* v_t_807_, lean_object* v_defEq_808_){
_start:
{
uint8_t v_t_boxed_809_; lean_object* v_res_810_; 
v_t_boxed_809_ = lean_unbox(v_t_807_);
v_res_810_ = lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___redArg(v_t_boxed_809_, v_defEq_808_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim(lean_object* v_u_811_, lean_object* v_v_812_, lean_object* v_motive_813_, uint8_t v_t_814_, lean_object* v_h_815_, lean_object* v_defEq_816_){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_814_, v_defEq_816_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_defEq_elim___boxed(lean_object* v_u_818_, lean_object* v_v_819_, lean_object* v_motive_820_, lean_object* v_t_821_, lean_object* v_h_822_, lean_object* v_defEq_823_){
_start:
{
uint8_t v_t_boxed_824_; lean_object* v_res_825_; 
v_t_boxed_824_ = lean_unbox(v_t_821_);
v_res_825_ = lp_Qq_Qq_MaybeLevelDefEq_defEq_elim(v_u_818_, v_v_819_, v_motive_820_, v_t_boxed_824_, v_h_822_, v_defEq_823_);
lean_dec(v_v_819_);
lean_dec(v_u_818_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___redArg(uint8_t v_t_826_, lean_object* v_notDefEq_827_){
_start:
{
lean_object* v___x_828_; 
v___x_828_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_826_, v_notDefEq_827_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___redArg___boxed(lean_object* v_t_829_, lean_object* v_notDefEq_830_){
_start:
{
uint8_t v_t_boxed_831_; lean_object* v_res_832_; 
v_t_boxed_831_ = lean_unbox(v_t_829_);
v_res_832_ = lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___redArg(v_t_boxed_831_, v_notDefEq_830_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim(lean_object* v_u_833_, lean_object* v_v_834_, lean_object* v_motive_835_, uint8_t v_t_836_, lean_object* v_h_837_, lean_object* v_notDefEq_838_){
_start:
{
lean_object* v___x_839_; 
v___x_839_ = lp_Qq_Qq_MaybeLevelDefEq_ctorElim___redArg(v_t_836_, v_notDefEq_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim___boxed(lean_object* v_u_840_, lean_object* v_v_841_, lean_object* v_motive_842_, lean_object* v_t_843_, lean_object* v_h_844_, lean_object* v_notDefEq_845_){
_start:
{
uint8_t v_t_boxed_846_; lean_object* v_res_847_; 
v_t_boxed_846_ = lean_unbox(v_t_843_);
v_res_847_ = lp_Qq_Qq_MaybeLevelDefEq_notDefEq_elim(v_u_840_, v_v_841_, v_motive_842_, v_t_boxed_846_, v_h_844_, v_notDefEq_845_);
lean_dec(v_v_841_);
lean_dec(v_u_840_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0(uint8_t v_x_848_, lean_object* v_x_849_){
_start:
{
if (v_x_848_ == 0)
{
lean_object* v___x_850_; lean_object* v___x_851_; 
v___x_850_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__1));
v___x_851_ = l_Repr_addAppParen(v___x_850_, v_x_849_);
return v___x_851_;
}
else
{
lean_object* v___x_852_; 
v___x_852_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeDefEq___lam__0___closed__3));
return v___x_852_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0___boxed(lean_object* v_x_853_, lean_object* v_x_854_){
_start:
{
uint8_t v_x_40__boxed_855_; lean_object* v_res_856_; 
v_x_40__boxed_855_ = lean_unbox(v_x_853_);
v_res_856_ = lp_Qq_Qq_instReprMaybeLevelDefEq___lam__0(v_x_40__boxed_855_, v_x_854_);
lean_dec(v_x_854_);
return v_res_856_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq(lean_object* v_u_858_, lean_object* v_v_859_){
_start:
{
lean_object* v___f_860_; 
v___f_860_ = ((lean_object*)(lp_Qq_Qq_instReprMaybeLevelDefEq___closed__0));
return v___f_860_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprMaybeLevelDefEq___boxed(lean_object* v_u_861_, lean_object* v_v_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_Qq_Qq_instReprMaybeLevelDefEq(v_u_861_, v_v_862_);
lean_dec(v_v_862_);
lean_dec(v_u_861_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_isLevelDefEqQ(lean_object* v_u_864_, lean_object* v_v_865_, lean_object* v_a_866_, lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_a_869_){
_start:
{
lean_object* v___x_871_; 
v___x_871_ = l_Lean_Meta_isLevelDefEq(v_u_864_, v_v_865_, v_a_866_, v_a_867_, v_a_868_, v_a_869_);
if (lean_obj_tag(v___x_871_) == 0)
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_887_; 
v_a_872_ = lean_ctor_get(v___x_871_, 0);
v_isSharedCheck_887_ = !lean_is_exclusive(v___x_871_);
if (v_isSharedCheck_887_ == 0)
{
v___x_874_ = v___x_871_;
v_isShared_875_ = v_isSharedCheck_887_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_871_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_887_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
uint8_t v___x_876_; 
v___x_876_ = lean_unbox(v_a_872_);
lean_dec(v_a_872_);
if (v___x_876_ == 0)
{
uint8_t v___x_877_; lean_object* v___x_878_; lean_object* v___x_880_; 
v___x_877_ = 1;
v___x_878_ = lean_box(v___x_877_);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v___x_878_);
v___x_880_ = v___x_874_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v___x_878_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
else
{
uint8_t v___x_882_; lean_object* v___x_883_; lean_object* v___x_885_; 
v___x_882_ = 0;
v___x_883_ = lean_box(v___x_882_);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v___x_883_);
v___x_885_ = v___x_874_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v___x_883_);
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
v_a_888_ = lean_ctor_get(v___x_871_, 0);
v_isSharedCheck_895_ = !lean_is_exclusive(v___x_871_);
if (v_isSharedCheck_895_ == 0)
{
v___x_890_ = v___x_871_;
v_isShared_891_ = v_isSharedCheck_895_;
goto v_resetjp_889_;
}
else
{
lean_inc(v_a_888_);
lean_dec(v___x_871_);
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
LEAN_EXPORT lean_object* lp_Qq_Qq_isLevelDefEqQ___boxed(lean_object* v_u_896_, lean_object* v_v_897_, lean_object* v_a_898_, lean_object* v_a_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
lean_object* v_res_903_; 
v_res_903_ = lp_Qq_Qq_isLevelDefEqQ(v_u_896_, v_v_897_, v_a_898_, v_a_899_, v_a_900_, v_a_901_);
lean_dec(v_a_901_);
lean_dec_ref(v_a_900_);
lean_dec(v_a_899_);
lean_dec_ref(v_a_898_);
return v_res_903_;
}
}
static lean_object* _init_lp_Qq_Qq_assertLevelDefEqQ___closed__1(void){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_905_ = ((lean_object*)(lp_Qq_Qq_assertLevelDefEqQ___closed__0));
v___x_906_ = l_Lean_stringToMessageData(v___x_905_);
return v___x_906_;
}
}
static lean_object* _init_lp_Qq_Qq_assertLevelDefEqQ___closed__3(void){
_start:
{
lean_object* v___x_908_; lean_object* v___x_909_; 
v___x_908_ = ((lean_object*)(lp_Qq_Qq_assertLevelDefEqQ___closed__2));
v___x_909_ = l_Lean_stringToMessageData(v___x_908_);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertLevelDefEqQ(lean_object* v_u_910_, lean_object* v_v_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_){
_start:
{
lean_object* v___x_917_; 
lean_inc(v_v_911_);
lean_inc(v_u_910_);
v___x_917_ = lp_Qq_Qq_isLevelDefEqQ(v_u_910_, v_v_911_, v_a_912_, v_a_913_, v_a_914_, v_a_915_);
if (lean_obj_tag(v___x_917_) == 0)
{
lean_object* v_a_918_; lean_object* v___x_920_; uint8_t v_isShared_921_; uint8_t v_isSharedCheck_934_; 
v_a_918_ = lean_ctor_get(v___x_917_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_917_);
if (v_isSharedCheck_934_ == 0)
{
v___x_920_ = v___x_917_;
v_isShared_921_ = v_isSharedCheck_934_;
goto v_resetjp_919_;
}
else
{
lean_inc(v_a_918_);
lean_dec(v___x_917_);
v___x_920_ = lean_box(0);
v_isShared_921_ = v_isSharedCheck_934_;
goto v_resetjp_919_;
}
v_resetjp_919_:
{
uint8_t v___x_922_; 
v___x_922_ = lean_unbox(v_a_918_);
lean_dec(v_a_918_);
if (v___x_922_ == 0)
{
lean_object* v___x_924_; 
lean_dec(v_v_911_);
lean_dec(v_u_910_);
if (v_isShared_921_ == 0)
{
lean_ctor_set(v___x_920_, 0, lean_box(0));
v___x_924_ = v___x_920_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, lean_box(0));
v___x_924_ = v_reuseFailAlloc_925_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
return v___x_924_;
}
}
else
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
lean_del_object(v___x_920_);
v___x_926_ = l_Lean_MessageData_ofLevel(v_u_910_);
v___x_927_ = lean_obj_once(&lp_Qq_Qq_assertLevelDefEqQ___closed__1, &lp_Qq_Qq_assertLevelDefEqQ___closed__1_once, _init_lp_Qq_Qq_assertLevelDefEqQ___closed__1);
v___x_928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_928_, 0, v___x_926_);
lean_ctor_set(v___x_928_, 1, v___x_927_);
v___x_929_ = l_Lean_MessageData_ofLevel(v_v_911_);
v___x_930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_928_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
v___x_931_ = lean_obj_once(&lp_Qq_Qq_assertLevelDefEqQ___closed__3, &lp_Qq_Qq_assertLevelDefEqQ___closed__3_once, _init_lp_Qq_Qq_assertLevelDefEqQ___closed__3);
v___x_932_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_932_, 0, v___x_930_);
lean_ctor_set(v___x_932_, 1, v___x_931_);
v___x_933_ = lp_Qq_Lean_throwError___at___00Qq_inferTypeQ_spec__0___redArg(v___x_932_, v_a_912_, v_a_913_, v_a_914_, v_a_915_);
return v___x_933_;
}
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
lean_dec(v_v_911_);
lean_dec(v_u_910_);
v_a_935_ = lean_ctor_get(v___x_917_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_917_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_917_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_917_);
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
}
LEAN_EXPORT lean_object* lp_Qq_Qq_assertLevelDefEqQ___boxed(lean_object* v_u_943_, lean_object* v_v_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_){
_start:
{
lean_object* v_res_950_; 
v_res_950_ = lp_Qq_Qq_assertLevelDefEqQ(v_u_943_, v_v_944_, v_a_945_, v_a_946_, v_a_947_, v_a_948_);
lean_dec(v_a_948_);
lean_dec_ref(v_a_947_);
lean_dec(v_a_946_);
lean_dec_ref(v_a_945_);
return v_res_950_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Delab(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_SynthInstance(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term_TermElabM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_MetaM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Delab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_SynthInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Term_TermElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_MetaM(uint8_t builtin) {
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
lean_object* initialize_Qq_Qq_Delab(uint8_t builtin);
lean_object* initialize_Lean_Meta_SynthInstance(uint8_t builtin);
lean_object* initialize_Lean_Elab_Term_TermElabM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_MetaM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Delab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_SynthInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Term_TermElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_MetaM(builtin);
}
#ifdef __cplusplus
}
#endif
