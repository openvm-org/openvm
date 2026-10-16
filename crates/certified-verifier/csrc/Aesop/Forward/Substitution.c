// Lean compiler output
// Module: Aesop.Forward.Substitution
// Imports: public import Init public meta import Init public import Aesop.Forward.LevelIndex public import Aesop.Forward.PremiseIndex public import Aesop.Util.Basic
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_filterMapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_bracket(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Option_instBEq_beq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Array_isEqvAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkAppOptM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
uint8_t lean_expr_lt(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_compareArraySizeThenLex___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqLevelMVarId_beq(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
uint64_t l_Lean_instHashableLevelMVarId_hash(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_collectLevelMVars(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t l_Lean_Expr_hash(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Expr_eqv___boxed(lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedSubstitution_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedSubstitution_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedSubstitution_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__0_value),((lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instInhabitedSubstitution_default___closed__1 = (const lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedSubstitution_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedSubstitution = (const lean_object*)&lp_aesop_Aesop_instInhabitedSubstitution_default___closed__1_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instBEq___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instBEq___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_eqv___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instBEq___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instBEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instBEq___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instBEq___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_Substitution_instBEq___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instBEq___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Substitution_instBEq = (const lean_object*)&lp_aesop_Aesop_Substitution_instBEq___closed__1_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Substitution_instHashable___lam__0(uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__4 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__5 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__6 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__0_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__7 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__7_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__2_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__3_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__4_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__8 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__8_value),((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(7, 0, 0, 0, 0, 0, 0, 0)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed__const__1_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Substitution_instHashable___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instHashable___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instHashable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instHashable___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_Substitution_instHashable___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Substitution_instHashable = (const lean_object*)&lp_aesop_Aesop_Substitution_instHashable___closed__1_value;
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instOrd___private__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instOrd___private__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instOrd___private__1___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___private__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___private__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instOrd___lam__1___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instOrd___private__1___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_Substitution_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Substitution_instOrd = (const lean_object*)&lp_aesop_Aesop_Substitution_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_empty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insert(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insert___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_find_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_find_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insertLevel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insertLevel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_findLevel_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_findLevel_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__0___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ↦ "};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__2 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3;
static const lean_string_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " | "};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__4 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__5 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6;
static const lean_string_object lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Substitution_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instToMessageData___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instToMessageData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instToMessageData___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instToMessageData___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instToMessageData___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___closed__2 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instToMessageData___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instToMessageData___lam__3, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___closed__3 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Substitution_instToMessageData___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Substitution_instToMessageData___lam__4, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__0_value),((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__1_value),((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__2_value),((lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__3_value)} };
static const lean_object* lp_aesop_Aesop_Substitution_instToMessageData___closed__4 = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__4_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Substitution_instToMessageData = (const lean_object*)&lp_aesop_Aesop_Substitution_instToMessageData___closed__4_value;
static lean_once_cell_t lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Substitution_mergeCompatible___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Aesop.Forward.Substitution"};
static const lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__0 = (const lean_object*)&lp_aesop_Aesop_Substitution_mergeCompatible___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Substitution_mergeCompatible___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Aesop.Substitution.mergeCompatible"};
static const lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__1 = (const lean_object*)&lp_aesop_Aesop_Substitution_mergeCompatible___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Substitution_mergeCompatible___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 60, .m_data = "assertion violation: s₁.premises.size == s₂.premises.size\n  "};
static const lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__2 = (const lean_object*)&lp_aesop_Aesop_Substitution_mergeCompatible___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_mergeCompatible___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__3;
static const lean_string_object lp_aesop_Aesop_Substitution_mergeCompatible___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 56, .m_data = "assertion violation: s₁.levels.size == s₂.levels.size\n  "};
static const lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__4 = (const lean_object*)&lp_aesop_Aesop_Substitution_mergeCompatible___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_mergeCompatible___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_mergeCompatible(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_containsHyp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_containsHyp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__1;
static const lean_array_object lp_aesop_Aesop_Substitution_openRuleType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__2 = (const lean_object*)&lp_aesop_Aesop_Substitution_openRuleType___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__3;
static const lean_string_object lp_aesop_Aesop_Substitution_openRuleType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "openRuleType: substitution has incorrect size. Rule:"};
static const lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__4 = (const lean_object*)&lp_aesop_Aesop_Substitution_openRuleType___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__5;
static const lean_string_object lp_aesop_Aesop_Substitution_openRuleType___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "\nRule type:"};
static const lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__6 = (const lean_object*)&lp_aesop_Aesop_Substitution_openRuleType___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__7;
static const lean_string_object lp_aesop_Aesop_Substitution_openRuleType___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\nSubstitution:"};
static const lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__8 = (const lean_object*)&lp_aesop_Aesop_Substitution_openRuleType___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__9;
static const lean_string_object lp_aesop_Aesop_Substitution_openRuleType___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 70, .m_capacity = 70, .m_length = 69, .m_data = "openRuleType: substitution contains incorrect number of levels. Rule:"};
static const lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__10 = (const lean_object*)&lp_aesop_Aesop_Substitution_openRuleType___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Substitution_openRuleType___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Substitution_openRuleType___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_openRuleType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_openRuleType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_openRuleType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_openRuleType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instBEq___lam__0(lean_object* v___x_7_, lean_object* v_s_u2081_8_, lean_object* v_s_u2082_9_){
_start:
{
lean_object* v_premises_10_; lean_object* v_premises_11_; lean_object* v___x_12_; lean_object* v___x_13_; uint8_t v___x_14_; 
v_premises_10_ = lean_ctor_get(v_s_u2081_8_, 0);
v_premises_11_ = lean_ctor_get(v_s_u2082_9_, 0);
v___x_12_ = lean_array_get_size(v_premises_10_);
v___x_13_ = lean_array_get_size(v_premises_11_);
v___x_14_ = lean_nat_dec_eq(v___x_12_, v___x_13_);
if (v___x_14_ == 0)
{
lean_dec_ref(v___x_7_);
return v___x_14_;
}
else
{
lean_object* v___x_15_; uint8_t v___x_16_; 
v___x_15_ = lean_alloc_closure((void*)(l_Option_instBEq_beq___boxed), 4, 2);
lean_closure_set(v___x_15_, 0, lean_box(0));
lean_closure_set(v___x_15_, 1, v___x_7_);
v___x_16_ = l_Array_isEqvAux___redArg(v_premises_10_, v_premises_11_, v___x_15_, v___x_12_);
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instBEq___lam__0___boxed(lean_object* v___x_17_, lean_object* v_s_u2081_18_, lean_object* v_s_u2082_19_){
_start:
{
uint8_t v_res_20_; lean_object* v_r_21_; 
v_res_20_ = lp_aesop_Aesop_Substitution_instBEq___lam__0(v___x_17_, v_s_u2081_18_, v_s_u2082_19_);
lean_dec_ref(v_s_u2082_19_);
lean_dec_ref(v_s_u2081_18_);
v_r_21_ = lean_box(v_res_20_);
return v_r_21_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Substitution_instHashable___lam__0(uint64_t v_x1_26_, lean_object* v_x2_27_){
_start:
{
if (lean_obj_tag(v_x2_27_) == 0)
{
uint64_t v___x_28_; uint64_t v___x_29_; 
v___x_28_ = 11ULL;
v___x_29_ = lean_uint64_mix_hash(v_x1_26_, v___x_28_);
return v___x_29_;
}
else
{
lean_object* v_val_30_; uint64_t v___x_31_; uint64_t v___x_32_; uint64_t v___x_33_; uint64_t v___x_34_; 
v_val_30_ = lean_ctor_get(v_x2_27_, 0);
v___x_31_ = l_Lean_Expr_hash(v_val_30_);
v___x_32_ = 13ULL;
v___x_33_ = lean_uint64_mix_hash(v___x_31_, v___x_32_);
v___x_34_ = lean_uint64_mix_hash(v_x1_26_, v___x_33_);
return v___x_34_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__0___boxed(lean_object* v_x1_35_, lean_object* v_x2_36_){
_start:
{
uint64_t v_x1_115__boxed_37_; uint64_t v_res_38_; lean_object* v_r_39_; 
v_x1_115__boxed_37_ = lean_unbox_uint64(v_x1_35_);
lean_dec_ref(v_x1_35_);
v_res_38_ = lp_aesop_Aesop_Substitution_instHashable___lam__0(v_x1_115__boxed_37_, v_x2_36_);
lean_dec(v_x2_36_);
v_r_39_ = lean_box_uint64(v_res_38_);
return v_r_39_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Substitution_instHashable___lam__1(lean_object* v___f_61_, lean_object* v_s_62_){
_start:
{
lean_object* v_premises_63_; uint64_t v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v_premises_63_ = lean_ctor_get(v_s_62_, 0);
lean_inc_ref(v_premises_63_);
lean_dec_ref(v_s_62_);
v___x_64_ = 7ULL;
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = lean_array_get_size(v_premises_63_);
v___x_67_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9));
v___x_68_ = lean_nat_dec_lt(v___x_65_, v___x_66_);
if (v___x_68_ == 0)
{
lean_dec_ref(v_premises_63_);
lean_dec_ref(v___f_61_);
return v___x_64_;
}
else
{
uint8_t v___x_69_; 
v___x_69_ = lean_nat_dec_le(v___x_66_, v___x_66_);
if (v___x_69_ == 0)
{
if (v___x_68_ == 0)
{
lean_dec_ref(v_premises_63_);
lean_dec_ref(v___f_61_);
return v___x_64_;
}
else
{
size_t v___x_70_; size_t v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; uint64_t v___x_74_; 
v___x_70_ = ((size_t)0ULL);
v___x_71_ = lean_usize_of_nat(v___x_66_);
v___x_72_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed__const__1));
v___x_73_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_67_, v___f_61_, v_premises_63_, v___x_70_, v___x_71_, v___x_72_);
v___x_74_ = lean_unbox_uint64(v___x_73_);
lean_dec(v___x_73_);
return v___x_74_;
}
}
else
{
size_t v___x_75_; size_t v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; uint64_t v___x_79_; 
v___x_75_ = ((size_t)0ULL);
v___x_76_ = lean_usize_of_nat(v___x_66_);
v___x_77_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed__const__1));
v___x_78_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_67_, v___f_61_, v_premises_63_, v___x_75_, v___x_76_, v___x_77_);
v___x_79_ = lean_unbox_uint64(v___x_78_);
lean_dec(v___x_78_);
return v___x_79_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instHashable___lam__1___boxed(lean_object* v___f_80_, lean_object* v_s_81_){
_start:
{
uint64_t v_res_82_; lean_object* v_r_83_; 
v_res_82_ = lp_aesop_Aesop_Substitution_instHashable___lam__1(v___f_80_, v_s_81_);
v_r_83_ = lean_box_uint64(v_res_82_);
return v_r_83_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(lean_object* v_e_u2081_88_, lean_object* v_e_u2082_89_){
_start:
{
uint8_t v___x_90_; 
v___x_90_ = lean_expr_eqv(v_e_u2081_88_, v_e_u2082_89_);
if (v___x_90_ == 0)
{
uint8_t v___x_91_; 
v___x_91_ = lean_expr_lt(v_e_u2081_88_, v_e_u2082_89_);
if (v___x_91_ == 0)
{
uint8_t v___x_92_; 
v___x_92_ = 2;
return v___x_92_;
}
else
{
uint8_t v___x_93_; 
v___x_93_ = 0;
return v___x_93_;
}
}
else
{
uint8_t v___x_94_; 
v___x_94_ = 1;
return v___x_94_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs___boxed(lean_object* v_e_u2081_95_, lean_object* v_e_u2082_96_){
_start:
{
uint8_t v_res_97_; lean_object* v_r_98_; 
v_res_97_ = lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(v_e_u2081_95_, v_e_u2082_96_);
lean_dec_ref(v_e_u2082_96_);
lean_dec_ref(v_e_u2081_95_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0(lean_object* v_x_99_, lean_object* v_x_100_){
_start:
{
if (lean_obj_tag(v_x_99_) == 0)
{
if (lean_obj_tag(v_x_100_) == 0)
{
uint8_t v___x_101_; 
v___x_101_ = 1;
return v___x_101_;
}
else
{
uint8_t v___x_102_; 
v___x_102_ = 0;
return v___x_102_;
}
}
else
{
if (lean_obj_tag(v_x_100_) == 0)
{
uint8_t v___x_103_; 
v___x_103_ = 2;
return v___x_103_;
}
else
{
lean_object* v_val_104_; lean_object* v_val_105_; uint8_t v___x_106_; 
v_val_104_ = lean_ctor_get(v_x_99_, 0);
v_val_105_ = lean_ctor_get(v_x_100_, 0);
v___x_106_ = lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(v_val_104_, v_val_105_);
return v___x_106_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0___boxed(lean_object* v_x_107_, lean_object* v_x_108_){
_start:
{
uint8_t v_res_109_; lean_object* v_r_110_; 
v_res_109_ = lp_aesop_Aesop_Substitution_instOrd___private__1___lam__0(v_x_107_, v_x_108_);
lean_dec(v_x_108_);
lean_dec(v_x_107_);
v_r_110_ = lean_box(v_res_109_);
return v_r_110_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___private__1(lean_object* v_s_u2081_112_, lean_object* v_s_u2082_113_){
_start:
{
lean_object* v_premises_114_; lean_object* v_premises_115_; lean_object* v___f_116_; uint8_t v___x_117_; 
v_premises_114_ = lean_ctor_get(v_s_u2081_112_, 0);
v_premises_115_ = lean_ctor_get(v_s_u2082_113_, 0);
v___f_116_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instOrd___private__1___closed__0));
v___x_117_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_116_, v_premises_114_, v_premises_115_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___private__1___boxed(lean_object* v_s_u2081_118_, lean_object* v_s_u2082_119_){
_start:
{
uint8_t v_res_120_; lean_object* v_r_121_; 
v_res_120_ = lp_aesop_Aesop_Substitution_instOrd___private__1(v_s_u2081_118_, v_s_u2082_119_);
lean_dec_ref(v_s_u2082_119_);
lean_dec_ref(v_s_u2081_118_);
v_r_121_ = lean_box(v_res_120_);
return v_r_121_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_instOrd___lam__1(lean_object* v___f_122_, lean_object* v_s_u2081_123_, lean_object* v_s_u2082_124_){
_start:
{
lean_object* v_premises_125_; lean_object* v_premises_126_; uint8_t v___x_127_; 
v_premises_125_ = lean_ctor_get(v_s_u2081_123_, 0);
v_premises_126_ = lean_ctor_get(v_s_u2082_124_, 0);
v___x_127_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_122_, v_premises_125_, v_premises_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instOrd___lam__1___boxed(lean_object* v___f_128_, lean_object* v_s_u2081_129_, lean_object* v_s_u2082_130_){
_start:
{
uint8_t v_res_131_; lean_object* v_r_132_; 
v_res_131_ = lp_aesop_Aesop_Substitution_instOrd___lam__1(v___f_128_, v_s_u2081_129_, v_s_u2082_130_);
lean_dec_ref(v_s_u2082_130_);
lean_dec_ref(v_s_u2081_129_);
v_r_132_ = lean_box(v_res_131_);
return v_r_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_empty(lean_object* v_numPremises_136_, lean_object* v_numLevels_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_138_ = lean_box(0);
v___x_139_ = lean_mk_array(v_numPremises_136_, v___x_138_);
v___x_140_ = lean_mk_array(v_numLevels_137_, v___x_138_);
v___x_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insert(lean_object* v_pi_142_, lean_object* v_inst_143_, lean_object* v_s_144_){
_start:
{
lean_object* v_premises_145_; lean_object* v_levels_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_155_; 
v_premises_145_ = lean_ctor_get(v_s_144_, 0);
v_levels_146_ = lean_ctor_get(v_s_144_, 1);
v_isSharedCheck_155_ = !lean_is_exclusive(v_s_144_);
if (v_isSharedCheck_155_ == 0)
{
v___x_148_ = v_s_144_;
v_isShared_149_ = v_isSharedCheck_155_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_levels_146_);
lean_inc(v_premises_145_);
lean_dec(v_s_144_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_155_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_153_; 
v___x_150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_150_, 0, v_inst_143_);
v___x_151_ = lean_array_set(v_premises_145_, v_pi_142_, v___x_150_);
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 0, v___x_151_);
v___x_153_ = v___x_148_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v___x_151_);
lean_ctor_set(v_reuseFailAlloc_154_, 1, v_levels_146_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insert___boxed(lean_object* v_pi_156_, lean_object* v_inst_157_, lean_object* v_s_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop_Aesop_Substitution_insert(v_pi_156_, v_inst_157_, v_s_158_);
lean_dec(v_pi_156_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_find_x3f(lean_object* v_pi_160_, lean_object* v_s_161_){
_start:
{
lean_object* v_premises_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v_premises_162_ = lean_ctor_get(v_s_161_, 0);
v___x_163_ = lean_box(0);
v___x_164_ = lean_array_get_borrowed(v___x_163_, v_premises_162_, v_pi_160_);
lean_inc(v___x_164_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_find_x3f___boxed(lean_object* v_pi_165_, lean_object* v_s_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_aesop_Aesop_Substitution_find_x3f(v_pi_165_, v_s_166_);
lean_dec_ref(v_s_166_);
lean_dec(v_pi_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insertLevel(lean_object* v_li_168_, lean_object* v_inst_169_, lean_object* v_s_170_){
_start:
{
lean_object* v_premises_171_; lean_object* v_levels_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_181_; 
v_premises_171_ = lean_ctor_get(v_s_170_, 0);
v_levels_172_ = lean_ctor_get(v_s_170_, 1);
v_isSharedCheck_181_ = !lean_is_exclusive(v_s_170_);
if (v_isSharedCheck_181_ == 0)
{
v___x_174_ = v_s_170_;
v_isShared_175_ = v_isSharedCheck_181_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_levels_172_);
lean_inc(v_premises_171_);
lean_dec(v_s_170_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_181_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_179_; 
v___x_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_176_, 0, v_inst_169_);
v___x_177_ = lean_array_set(v_levels_172_, v_li_168_, v___x_176_);
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 1, v___x_177_);
v___x_179_ = v___x_174_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_premises_171_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v___x_177_);
v___x_179_ = v_reuseFailAlloc_180_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
return v___x_179_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_insertLevel___boxed(lean_object* v_li_182_, lean_object* v_inst_183_, lean_object* v_s_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_aesop_Aesop_Substitution_insertLevel(v_li_182_, v_inst_183_, v_s_184_);
lean_dec(v_li_182_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_findLevel_x3f(lean_object* v_li_186_, lean_object* v_s_187_){
_start:
{
lean_object* v_levels_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v_levels_188_ = lean_ctor_get(v_s_187_, 1);
v___x_189_ = lean_box(0);
v___x_190_ = lean_array_get_borrowed(v___x_189_, v_levels_188_, v_li_186_);
lean_inc(v___x_190_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_findLevel_x3f___boxed(lean_object* v_li_191_, lean_object* v_s_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_aesop_Aesop_Substitution_findLevel_x3f(v_li_191_, v_s_192_);
lean_dec_ref(v_s_192_);
lean_dec(v_li_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__0(lean_object* v_x_194_){
_start:
{
lean_inc(v_x_194_);
return v_x_194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__0___boxed(lean_object* v_x_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_aesop_Aesop_Substitution_instToMessageData___lam__0(v_x_195_);
lean_dec(v_x_195_);
return v_res_196_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__0));
v___x_199_ = l_Lean_stringToMessageData(v___x_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__1(lean_object* v_i_200_, lean_object* v_a_201_, lean_object* v_x_202_){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_203_ = l_Nat_reprFast(v_i_200_);
v___x_204_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
v___x_205_ = l_Lean_MessageData_ofFormat(v___x_204_);
v___x_206_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1);
v___x_207_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_205_);
lean_ctor_set(v___x_207_, 1, v___x_206_);
v___x_208_ = l_Lean_MessageData_ofExpr(v_a_201_);
v___x_209_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_207_);
lean_ctor_set(v___x_209_, 1, v___x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__2(lean_object* v_x_210_){
_start:
{
lean_inc(v_x_210_);
return v_x_210_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__2___boxed(lean_object* v_x_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_aesop_Aesop_Substitution_instToMessageData___lam__2(v_x_211_);
lean_dec(v_x_211_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__3(lean_object* v_i_213_, lean_object* v_a_214_, lean_object* v_x_215_){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_216_ = l_Nat_reprFast(v_i_213_);
v___x_217_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_217_, 0, v___x_216_);
v___x_218_ = l_Lean_MessageData_ofFormat(v___x_217_);
v___x_219_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1);
v___x_220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_220_, 0, v___x_218_);
lean_ctor_set(v___x_220_, 1, v___x_219_);
v___x_221_ = l_Lean_MessageData_ofLevel(v_a_214_);
v___x_222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_220_);
lean_ctor_set(v___x_222_, 1, v___x_221_);
return v___x_222_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3(void){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__2));
v___x_228_ = l_Lean_MessageData_ofFormat(v___x_227_);
return v___x_228_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__5));
v___x_233_ = l_Lean_MessageData_ofFormat(v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_instToMessageData___lam__4(lean_object* v___f_235_, lean_object* v___f_236_, lean_object* v___f_237_, lean_object* v___f_238_, lean_object* v_s_239_){
_start:
{
lean_object* v_premises_240_; lean_object* v_levels_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_269_; 
v_premises_240_ = lean_ctor_get(v_s_239_, 0);
v_levels_241_ = lean_ctor_get(v_s_239_, 1);
v_isSharedCheck_269_ = !lean_is_exclusive(v_s_239_);
if (v_isSharedCheck_269_ == 0)
{
v___x_243_ = v_s_239_;
v_isShared_244_ = v_isSharedCheck_269_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_levels_241_);
lean_inc(v_premises_240_);
lean_dec(v_s_239_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_269_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; size_t v_sz_249_; size_t v___x_250_; lean_object* v___x_251_; lean_object* v_ps_252_; lean_object* v___x_253_; lean_object* v___x_254_; size_t v_sz_255_; lean_object* v___x_256_; lean_object* v_ls_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_263_; 
v___x_245_ = lean_unsigned_to_nat(0u);
v___x_246_ = lean_array_get_size(v_premises_240_);
v___x_247_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9));
v___x_248_ = l_Array_filterMapM___redArg(v___x_247_, v___f_235_, v_premises_240_, v___x_245_, v___x_246_);
v_sz_249_ = lean_array_size(v___x_248_);
v___x_250_ = ((size_t)0ULL);
lean_inc(v___x_248_);
v___x_251_ = l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_247_, v___x_248_, v___f_236_, v_sz_249_, v___x_250_, v___x_248_);
lean_dec(v___x_248_);
v_ps_252_ = lean_array_to_list(v___x_251_);
v___x_253_ = lean_array_get_size(v_levels_241_);
v___x_254_ = l_Array_filterMapM___redArg(v___x_247_, v___f_237_, v_levels_241_, v___x_245_, v___x_253_);
v_sz_255_ = lean_array_size(v___x_254_);
lean_inc(v___x_254_);
v___x_256_ = l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_247_, v___x_254_, v___f_238_, v_sz_255_, v___x_250_, v___x_254_);
lean_dec(v___x_254_);
v_ls_257_ = lean_array_to_list(v___x_256_);
v___x_258_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0));
v___x_259_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3);
v___x_260_ = l_Lean_MessageData_joinSep(v_ps_252_, v___x_259_);
v___x_261_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6);
if (v_isShared_244_ == 0)
{
lean_ctor_set_tag(v___x_243_, 7);
lean_ctor_set(v___x_243_, 1, v___x_261_);
lean_ctor_set(v___x_243_, 0, v___x_260_);
v___x_263_ = v___x_243_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v___x_261_);
v___x_263_ = v_reuseFailAlloc_268_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_264_ = l_Lean_MessageData_joinSep(v_ls_257_, v___x_259_);
v___x_265_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_263_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
v___x_266_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7));
v___x_267_ = l_Lean_MessageData_bracket(v___x_258_, v___x_265_, v___x_266_);
return v___x_267_;
}
}
}
}
static lean_object* _init_lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0(void){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_280_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedSubstitution_default));
v___x_281_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instHashable___lam__1___closed__9));
v___x_282_ = l_instInhabitedOfMonad___redArg(v___x_281_, v___x_280_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0(lean_object* v_msg_283_){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_284_ = lean_obj_once(&lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0, &lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0_once, _init_lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0___closed__0);
v___x_285_ = lean_panic_fn_borrowed(v___x_284_, v_msg_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg(lean_object* v___x_286_, lean_object* v___x_287_, lean_object* v___x_288_, lean_object* v___x_289_, lean_object* v_range_290_, lean_object* v_b_291_, lean_object* v_i_292_){
_start:
{
lean_object* v_stop_293_; lean_object* v_step_294_; lean_object* v_a_296_; uint8_t v___x_299_; 
v_stop_293_ = lean_ctor_get(v_range_290_, 1);
v_step_294_ = lean_ctor_get(v_range_290_, 2);
v___x_299_ = lean_nat_dec_lt(v_i_292_, v_stop_293_);
if (v___x_299_ == 0)
{
lean_dec(v_i_292_);
return v_b_291_;
}
else
{
lean_object* v___x_300_; 
v___x_300_ = lean_array_fget_borrowed(v___x_286_, v_i_292_);
if (lean_obj_tag(v___x_300_) == 1)
{
lean_object* v_val_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v_val_301_ = lean_ctor_get(v___x_300_, 0);
v___x_302_ = lean_box(0);
v___x_303_ = lean_array_get_borrowed(v___x_302_, v___x_287_, v_i_292_);
if (lean_obj_tag(v___x_303_) == 0)
{
uint8_t v___x_304_; 
v___x_304_ = lean_nat_dec_eq(v___x_288_, v___x_289_);
if (v___x_304_ == 0)
{
v_a_296_ = v_b_291_;
goto v___jp_295_;
}
else
{
lean_object* v___x_305_; 
lean_inc(v_val_301_);
v___x_305_ = lp_aesop_Aesop_Substitution_insertLevel(v_i_292_, v_val_301_, v_b_291_);
v_a_296_ = v___x_305_;
goto v___jp_295_;
}
}
else
{
v_a_296_ = v_b_291_;
goto v___jp_295_;
}
}
else
{
v_a_296_ = v_b_291_;
goto v___jp_295_;
}
}
v___jp_295_:
{
lean_object* v___x_297_; 
v___x_297_ = lean_nat_add(v_i_292_, v_step_294_);
lean_dec(v_i_292_);
v_b_291_ = v_a_296_;
v_i_292_ = v___x_297_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg___boxed(lean_object* v___x_306_, lean_object* v___x_307_, lean_object* v___x_308_, lean_object* v___x_309_, lean_object* v_range_310_, lean_object* v_b_311_, lean_object* v_i_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg(v___x_306_, v___x_307_, v___x_308_, v___x_309_, v_range_310_, v_b_311_, v_i_312_);
lean_dec_ref(v_range_310_);
lean_dec(v___x_309_);
lean_dec(v___x_308_);
lean_dec_ref(v___x_307_);
lean_dec_ref(v___x_306_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg(lean_object* v___x_314_, lean_object* v___x_315_, lean_object* v___x_316_, lean_object* v___x_317_, lean_object* v_range_318_, lean_object* v_b_319_, lean_object* v_i_320_){
_start:
{
lean_object* v_stop_321_; lean_object* v_step_322_; lean_object* v_a_324_; uint8_t v___x_327_; 
v_stop_321_ = lean_ctor_get(v_range_318_, 1);
v_step_322_ = lean_ctor_get(v_range_318_, 2);
v___x_327_ = lean_nat_dec_lt(v_i_320_, v_stop_321_);
if (v___x_327_ == 0)
{
lean_dec(v_i_320_);
return v_b_319_;
}
else
{
lean_object* v___x_328_; 
v___x_328_ = lean_array_fget_borrowed(v___x_314_, v_i_320_);
if (lean_obj_tag(v___x_328_) == 1)
{
lean_object* v_val_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v_val_329_ = lean_ctor_get(v___x_328_, 0);
v___x_330_ = lean_box(0);
v___x_331_ = lean_array_get_borrowed(v___x_330_, v___x_315_, v_i_320_);
if (lean_obj_tag(v___x_331_) == 0)
{
uint8_t v___x_332_; 
v___x_332_ = lean_nat_dec_eq(v___x_316_, v___x_317_);
if (v___x_332_ == 0)
{
v_a_324_ = v_b_319_;
goto v___jp_323_;
}
else
{
lean_object* v_result_333_; 
lean_inc(v_val_329_);
v_result_333_ = lp_aesop_Aesop_Substitution_insert(v_i_320_, v_val_329_, v_b_319_);
v_a_324_ = v_result_333_;
goto v___jp_323_;
}
}
else
{
v_a_324_ = v_b_319_;
goto v___jp_323_;
}
}
else
{
v_a_324_ = v_b_319_;
goto v___jp_323_;
}
}
v___jp_323_:
{
lean_object* v___x_325_; 
v___x_325_ = lean_nat_add(v_i_320_, v_step_322_);
lean_dec(v_i_320_);
v_b_319_ = v_a_324_;
v_i_320_ = v___x_325_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg___boxed(lean_object* v___x_334_, lean_object* v___x_335_, lean_object* v___x_336_, lean_object* v___x_337_, lean_object* v_range_338_, lean_object* v_b_339_, lean_object* v_i_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg(v___x_334_, v___x_335_, v___x_336_, v___x_337_, v_range_338_, v_b_339_, v_i_340_);
lean_dec_ref(v_range_338_);
lean_dec(v___x_337_);
lean_dec(v___x_336_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
return v_res_341_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_mergeCompatible___closed__3(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_345_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__2));
v___x_346_ = lean_unsigned_to_nat(2u);
v___x_347_ = lean_unsigned_to_nat(95u);
v___x_348_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__1));
v___x_349_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__0));
v___x_350_ = l_mkPanicMessageWithDecl(v___x_349_, v___x_348_, v___x_347_, v___x_346_, v___x_345_);
return v___x_350_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_mergeCompatible___closed__5(void){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_352_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__4));
v___x_353_ = lean_unsigned_to_nat(2u);
v___x_354_ = lean_unsigned_to_nat(96u);
v___x_355_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__1));
v___x_356_ = ((lean_object*)(lp_aesop_Aesop_Substitution_mergeCompatible___closed__0));
v___x_357_ = l_mkPanicMessageWithDecl(v___x_356_, v___x_355_, v___x_354_, v___x_353_, v___x_352_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_mergeCompatible(lean_object* v_s_u2081_358_, lean_object* v_s_u2082_359_){
_start:
{
lean_object* v_premises_360_; lean_object* v_levels_361_; lean_object* v_premises_362_; lean_object* v_levels_363_; lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v_premises_360_ = lean_ctor_get(v_s_u2081_358_, 0);
lean_inc_ref(v_premises_360_);
v_levels_361_ = lean_ctor_get(v_s_u2081_358_, 1);
lean_inc_ref(v_levels_361_);
v_premises_362_ = lean_ctor_get(v_s_u2082_359_, 0);
v_levels_363_ = lean_ctor_get(v_s_u2082_359_, 1);
v___x_364_ = lean_array_get_size(v_premises_360_);
v___x_365_ = lean_array_get_size(v_premises_362_);
v___x_366_ = lean_nat_dec_eq(v___x_364_, v___x_365_);
if (v___x_366_ == 0)
{
lean_object* v___x_367_; lean_object* v___x_368_; 
lean_dec_ref(v_levels_361_);
lean_dec_ref(v_premises_360_);
lean_dec_ref(v_s_u2081_358_);
v___x_367_ = lean_obj_once(&lp_aesop_Aesop_Substitution_mergeCompatible___closed__3, &lp_aesop_Aesop_Substitution_mergeCompatible___closed__3_once, _init_lp_aesop_Aesop_Substitution_mergeCompatible___closed__3);
v___x_368_ = lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0(v___x_367_);
return v___x_368_;
}
else
{
lean_object* v___x_369_; lean_object* v___x_370_; uint8_t v___x_371_; 
v___x_369_ = lean_array_get_size(v_levels_361_);
v___x_370_ = lean_array_get_size(v_levels_363_);
v___x_371_ = lean_nat_dec_eq(v___x_369_, v___x_370_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; 
lean_dec_ref(v_levels_361_);
lean_dec_ref(v_premises_360_);
lean_dec_ref(v_s_u2081_358_);
v___x_372_ = lean_obj_once(&lp_aesop_Aesop_Substitution_mergeCompatible___closed__5, &lp_aesop_Aesop_Substitution_mergeCompatible___closed__5_once, _init_lp_aesop_Aesop_Substitution_mergeCompatible___closed__5);
v___x_373_ = lp_aesop_panic___at___00Aesop_Substitution_mergeCompatible_spec__0(v___x_372_);
return v___x_373_;
}
else
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_374_ = lean_unsigned_to_nat(0u);
v___x_375_ = lean_unsigned_to_nat(1u);
v___x_376_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_376_, 0, v___x_374_);
lean_ctor_set(v___x_376_, 1, v___x_365_);
lean_ctor_set(v___x_376_, 2, v___x_375_);
v___x_377_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg(v_premises_362_, v_premises_360_, v___x_369_, v___x_370_, v___x_376_, v_s_u2081_358_, v___x_374_);
lean_dec_ref_known(v___x_376_, 3);
lean_dec_ref(v_premises_360_);
v___x_378_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_378_, 0, v___x_374_);
lean_ctor_set(v___x_378_, 1, v___x_370_);
lean_ctor_set(v___x_378_, 2, v___x_375_);
v___x_379_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg(v_levels_363_, v_levels_361_, v___x_369_, v___x_370_, v___x_378_, v___x_377_, v___x_374_);
lean_dec_ref_known(v___x_378_, 3);
lean_dec_ref(v_levels_361_);
return v___x_379_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_mergeCompatible___boxed(lean_object* v_s_u2081_380_, lean_object* v_s_u2082_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_aesop_Aesop_Substitution_mergeCompatible(v_s_u2081_380_, v_s_u2082_381_);
lean_dec_ref(v_s_u2082_381_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1(lean_object* v___x_383_, lean_object* v___x_384_, lean_object* v___x_385_, lean_object* v___x_386_, lean_object* v_range_387_, lean_object* v_b_388_, lean_object* v_i_389_, lean_object* v_hs_390_, lean_object* v_hl_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___redArg(v___x_383_, v___x_384_, v___x_385_, v___x_386_, v_range_387_, v_b_388_, v_i_389_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1___boxed(lean_object* v___x_393_, lean_object* v___x_394_, lean_object* v___x_395_, lean_object* v___x_396_, lean_object* v_range_397_, lean_object* v_b_398_, lean_object* v_i_399_, lean_object* v_hs_400_, lean_object* v_hl_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__1(v___x_393_, v___x_394_, v___x_395_, v___x_396_, v_range_397_, v_b_398_, v_i_399_, v_hs_400_, v_hl_401_);
lean_dec_ref(v_range_397_);
lean_dec(v___x_396_);
lean_dec(v___x_395_);
lean_dec_ref(v___x_394_);
lean_dec_ref(v___x_393_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2(lean_object* v___x_403_, lean_object* v___x_404_, lean_object* v___x_405_, lean_object* v___x_406_, lean_object* v_range_407_, lean_object* v_b_408_, lean_object* v_i_409_, lean_object* v_hs_410_, lean_object* v_hl_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___redArg(v___x_403_, v___x_404_, v___x_405_, v___x_406_, v_range_407_, v_b_408_, v_i_409_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2___boxed(lean_object* v___x_413_, lean_object* v___x_414_, lean_object* v___x_415_, lean_object* v___x_416_, lean_object* v_range_417_, lean_object* v_b_418_, lean_object* v_i_419_, lean_object* v_hs_420_, lean_object* v_hl_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_mergeCompatible_spec__2(v___x_413_, v___x_414_, v___x_415_, v___x_416_, v_range_417_, v_b_418_, v_i_419_, v_hs_420_, v_hl_421_);
lean_dec_ref(v_range_417_);
lean_dec(v___x_416_);
lean_dec(v___x_415_);
lean_dec_ref(v___x_414_);
lean_dec_ref(v___x_413_);
return v_res_422_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0(lean_object* v_hyp_423_, lean_object* v_as_424_, size_t v_i_425_, size_t v_stop_426_){
_start:
{
uint8_t v___x_427_; 
v___x_427_ = lean_usize_dec_eq(v_i_425_, v_stop_426_);
if (v___x_427_ == 0)
{
uint8_t v___x_428_; uint8_t v___y_430_; lean_object* v___x_434_; 
v___x_428_ = 1;
v___x_434_ = lean_array_uget_borrowed(v_as_424_, v_i_425_);
if (lean_obj_tag(v___x_434_) == 0)
{
v___y_430_ = v___x_427_;
goto v___jp_429_;
}
else
{
lean_object* v_val_435_; uint8_t v___x_436_; 
v_val_435_ = lean_ctor_get(v___x_434_, 0);
v___x_436_ = l_Lean_Expr_containsFVar(v_val_435_, v_hyp_423_);
v___y_430_ = v___x_436_;
goto v___jp_429_;
}
v___jp_429_:
{
if (v___y_430_ == 0)
{
size_t v___x_431_; size_t v___x_432_; 
v___x_431_ = ((size_t)1ULL);
v___x_432_ = lean_usize_add(v_i_425_, v___x_431_);
v_i_425_ = v___x_432_;
goto _start;
}
else
{
return v___x_428_;
}
}
}
else
{
uint8_t v___x_437_; 
v___x_437_ = 0;
return v___x_437_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0___boxed(lean_object* v_hyp_438_, lean_object* v_as_439_, lean_object* v_i_440_, lean_object* v_stop_441_){
_start:
{
size_t v_i_boxed_442_; size_t v_stop_boxed_443_; uint8_t v_res_444_; lean_object* v_r_445_; 
v_i_boxed_442_ = lean_unbox_usize(v_i_440_);
lean_dec(v_i_440_);
v_stop_boxed_443_ = lean_unbox_usize(v_stop_441_);
lean_dec(v_stop_441_);
v_res_444_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0(v_hyp_438_, v_as_439_, v_i_boxed_442_, v_stop_boxed_443_);
lean_dec_ref(v_as_439_);
lean_dec(v_hyp_438_);
v_r_445_ = lean_box(v_res_444_);
return v_r_445_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Substitution_containsHyp(lean_object* v_hyp_446_, lean_object* v_s_447_){
_start:
{
lean_object* v_premises_448_; lean_object* v___x_449_; lean_object* v___x_450_; uint8_t v___x_451_; 
v_premises_448_ = lean_ctor_get(v_s_447_, 0);
v___x_449_ = lean_unsigned_to_nat(0u);
v___x_450_ = lean_array_get_size(v_premises_448_);
v___x_451_ = lean_nat_dec_lt(v___x_449_, v___x_450_);
if (v___x_451_ == 0)
{
return v___x_451_;
}
else
{
if (v___x_451_ == 0)
{
return v___x_451_;
}
else
{
size_t v___x_452_; size_t v___x_453_; uint8_t v___x_454_; 
v___x_452_ = ((size_t)0ULL);
v___x_453_ = lean_usize_of_nat(v___x_450_);
v___x_454_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Substitution_containsHyp_spec__0(v_hyp_446_, v_premises_448_, v___x_452_, v___x_453_);
return v___x_454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_containsHyp___boxed(lean_object* v_hyp_455_, lean_object* v_s_456_){
_start:
{
uint8_t v_res_457_; lean_object* v_r_458_; 
v_res_457_ = lp_aesop_Aesop_Substitution_containsHyp(v_hyp_455_, v_s_456_);
lean_dec_ref(v_s_456_);
lean_dec(v_hyp_455_);
v_r_458_ = lean_box(v_res_457_);
return v_r_458_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(lean_object* v_e_459_, lean_object* v___y_460_){
_start:
{
uint8_t v___x_462_; 
v___x_462_ = l_Lean_Expr_hasMVar(v_e_459_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; 
v___x_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_463_, 0, v_e_459_);
return v___x_463_;
}
else
{
lean_object* v___x_464_; lean_object* v_mctx_465_; lean_object* v___x_466_; lean_object* v_fst_467_; lean_object* v_snd_468_; lean_object* v___x_469_; lean_object* v_cache_470_; lean_object* v_zetaDeltaFVarIds_471_; lean_object* v_postponed_472_; lean_object* v_diag_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_482_; 
v___x_464_ = lean_st_ref_get(v___y_460_);
v_mctx_465_ = lean_ctor_get(v___x_464_, 0);
lean_inc_ref(v_mctx_465_);
lean_dec(v___x_464_);
v___x_466_ = l_Lean_instantiateMVarsCore(v_mctx_465_, v_e_459_);
v_fst_467_ = lean_ctor_get(v___x_466_, 0);
lean_inc(v_fst_467_);
v_snd_468_ = lean_ctor_get(v___x_466_, 1);
lean_inc(v_snd_468_);
lean_dec_ref(v___x_466_);
v___x_469_ = lean_st_ref_take(v___y_460_);
v_cache_470_ = lean_ctor_get(v___x_469_, 1);
v_zetaDeltaFVarIds_471_ = lean_ctor_get(v___x_469_, 2);
v_postponed_472_ = lean_ctor_get(v___x_469_, 3);
v_diag_473_ = lean_ctor_get(v___x_469_, 4);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_469_);
if (v_isSharedCheck_482_ == 0)
{
lean_object* v_unused_483_; 
v_unused_483_ = lean_ctor_get(v___x_469_, 0);
lean_dec(v_unused_483_);
v___x_475_ = v___x_469_;
v_isShared_476_ = v_isSharedCheck_482_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_diag_473_);
lean_inc(v_postponed_472_);
lean_inc(v_zetaDeltaFVarIds_471_);
lean_inc(v_cache_470_);
lean_dec(v___x_469_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_482_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_478_; 
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 0, v_snd_468_);
v___x_478_ = v___x_475_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_snd_468_);
lean_ctor_set(v_reuseFailAlloc_481_, 1, v_cache_470_);
lean_ctor_set(v_reuseFailAlloc_481_, 2, v_zetaDeltaFVarIds_471_);
lean_ctor_set(v_reuseFailAlloc_481_, 3, v_postponed_472_);
lean_ctor_set(v_reuseFailAlloc_481_, 4, v_diag_473_);
v___x_478_ = v_reuseFailAlloc_481_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = lean_st_ref_set(v___y_460_, v___x_478_);
v___x_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_480_, 0, v_fst_467_);
return v___x_480_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg___boxed(lean_object* v_e_484_, lean_object* v___y_485_, lean_object* v___y_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(v_e_484_, v___y_485_);
lean_dec(v___y_485_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0(lean_object* v_e_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(v_e_488_, v___y_490_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___boxed(lean_object* v_e_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0(v_e_495_, v___y_496_, v___y_497_, v___y_498_, v___y_499_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21___redArg(lean_object* v_x_502_, lean_object* v_x_503_, lean_object* v_x_504_, lean_object* v_x_505_){
_start:
{
lean_object* v_ks_506_; lean_object* v_vs_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_531_; 
v_ks_506_ = lean_ctor_get(v_x_502_, 0);
v_vs_507_ = lean_ctor_get(v_x_502_, 1);
v_isSharedCheck_531_ = !lean_is_exclusive(v_x_502_);
if (v_isSharedCheck_531_ == 0)
{
v___x_509_ = v_x_502_;
v_isShared_510_ = v_isSharedCheck_531_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_vs_507_);
lean_inc(v_ks_506_);
lean_dec(v_x_502_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_531_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_511_; uint8_t v___x_512_; 
v___x_511_ = lean_array_get_size(v_ks_506_);
v___x_512_ = lean_nat_dec_lt(v_x_503_, v___x_511_);
if (v___x_512_ == 0)
{
lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
lean_dec(v_x_503_);
v___x_513_ = lean_array_push(v_ks_506_, v_x_504_);
v___x_514_ = lean_array_push(v_vs_507_, v_x_505_);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v___x_514_);
lean_ctor_set(v___x_509_, 0, v___x_513_);
v___x_516_ = v___x_509_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_513_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v___x_514_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
else
{
lean_object* v_k_x27_518_; uint8_t v___x_519_; 
v_k_x27_518_ = lean_array_fget_borrowed(v_ks_506_, v_x_503_);
v___x_519_ = l_Lean_instBEqLevelMVarId_beq(v_x_504_, v_k_x27_518_);
if (v___x_519_ == 0)
{
lean_object* v___x_521_; 
if (v_isShared_510_ == 0)
{
v___x_521_ = v___x_509_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_525_; 
v_reuseFailAlloc_525_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_525_, 0, v_ks_506_);
lean_ctor_set(v_reuseFailAlloc_525_, 1, v_vs_507_);
v___x_521_ = v_reuseFailAlloc_525_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
lean_object* v___x_522_; lean_object* v___x_523_; 
v___x_522_ = lean_unsigned_to_nat(1u);
v___x_523_ = lean_nat_add(v_x_503_, v___x_522_);
lean_dec(v_x_503_);
v_x_502_ = v___x_521_;
v_x_503_ = v___x_523_;
goto _start;
}
}
else
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_529_; 
v___x_526_ = lean_array_fset(v_ks_506_, v_x_503_, v_x_504_);
v___x_527_ = lean_array_fset(v_vs_507_, v_x_503_, v_x_505_);
lean_dec(v_x_503_);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v___x_527_);
lean_ctor_set(v___x_509_, 0, v___x_526_);
v___x_529_ = v___x_509_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_526_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v___x_527_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17___redArg(lean_object* v_n_532_, lean_object* v_k_533_, lean_object* v_v_534_){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = lean_unsigned_to_nat(0u);
v___x_536_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21___redArg(v_n_532_, v___x_535_, v_k_533_, v_v_534_);
return v___x_536_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(lean_object* v_x_538_, size_t v_x_539_, size_t v_x_540_, lean_object* v_x_541_, lean_object* v_x_542_){
_start:
{
if (lean_obj_tag(v_x_538_) == 0)
{
lean_object* v_es_543_; size_t v___x_544_; size_t v___x_545_; lean_object* v_j_546_; lean_object* v___x_547_; uint8_t v___x_548_; 
v_es_543_ = lean_ctor_get(v_x_538_, 0);
v___x_544_ = ((size_t)31ULL);
v___x_545_ = lean_usize_land(v_x_539_, v___x_544_);
v_j_546_ = lean_usize_to_nat(v___x_545_);
v___x_547_ = lean_array_get_size(v_es_543_);
v___x_548_ = lean_nat_dec_lt(v_j_546_, v___x_547_);
if (v___x_548_ == 0)
{
lean_dec(v_j_546_);
lean_dec(v_x_542_);
lean_dec(v_x_541_);
return v_x_538_;
}
else
{
lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_587_; 
lean_inc_ref(v_es_543_);
v_isSharedCheck_587_ = !lean_is_exclusive(v_x_538_);
if (v_isSharedCheck_587_ == 0)
{
lean_object* v_unused_588_; 
v_unused_588_ = lean_ctor_get(v_x_538_, 0);
lean_dec(v_unused_588_);
v___x_550_ = v_x_538_;
v_isShared_551_ = v_isSharedCheck_587_;
goto v_resetjp_549_;
}
else
{
lean_dec(v_x_538_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_587_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v_v_552_; lean_object* v___x_553_; lean_object* v_xs_x27_554_; lean_object* v___y_556_; 
v_v_552_ = lean_array_fget(v_es_543_, v_j_546_);
v___x_553_ = lean_box(0);
v_xs_x27_554_ = lean_array_fset(v_es_543_, v_j_546_, v___x_553_);
switch(lean_obj_tag(v_v_552_))
{
case 0:
{
lean_object* v_key_561_; lean_object* v_val_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_572_; 
v_key_561_ = lean_ctor_get(v_v_552_, 0);
v_val_562_ = lean_ctor_get(v_v_552_, 1);
v_isSharedCheck_572_ = !lean_is_exclusive(v_v_552_);
if (v_isSharedCheck_572_ == 0)
{
v___x_564_ = v_v_552_;
v_isShared_565_ = v_isSharedCheck_572_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_val_562_);
lean_inc(v_key_561_);
lean_dec(v_v_552_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_572_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
uint8_t v___x_566_; 
v___x_566_ = l_Lean_instBEqLevelMVarId_beq(v_x_541_, v_key_561_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; lean_object* v___x_568_; 
lean_del_object(v___x_564_);
v___x_567_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_561_, v_val_562_, v_x_541_, v_x_542_);
v___x_568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_568_, 0, v___x_567_);
v___y_556_ = v___x_568_;
goto v___jp_555_;
}
else
{
lean_object* v___x_570_; 
lean_dec(v_val_562_);
lean_dec(v_key_561_);
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 1, v_x_542_);
lean_ctor_set(v___x_564_, 0, v_x_541_);
v___x_570_ = v___x_564_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_x_541_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v_x_542_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
v___y_556_ = v___x_570_;
goto v___jp_555_;
}
}
}
}
case 1:
{
lean_object* v_node_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_585_; 
v_node_573_ = lean_ctor_get(v_v_552_, 0);
v_isSharedCheck_585_ = !lean_is_exclusive(v_v_552_);
if (v_isSharedCheck_585_ == 0)
{
v___x_575_ = v_v_552_;
v_isShared_576_ = v_isSharedCheck_585_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_node_573_);
lean_dec(v_v_552_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_585_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
size_t v___x_577_; size_t v___x_578_; size_t v___x_579_; size_t v___x_580_; lean_object* v___x_581_; lean_object* v___x_583_; 
v___x_577_ = ((size_t)5ULL);
v___x_578_ = lean_usize_shift_right(v_x_539_, v___x_577_);
v___x_579_ = ((size_t)1ULL);
v___x_580_ = lean_usize_add(v_x_540_, v___x_579_);
v___x_581_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(v_node_573_, v___x_578_, v___x_580_, v_x_541_, v_x_542_);
if (v_isShared_576_ == 0)
{
lean_ctor_set(v___x_575_, 0, v___x_581_);
v___x_583_ = v___x_575_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_584_; 
v_reuseFailAlloc_584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_584_, 0, v___x_581_);
v___x_583_ = v_reuseFailAlloc_584_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
v___y_556_ = v___x_583_;
goto v___jp_555_;
}
}
}
default: 
{
lean_object* v___x_586_; 
v___x_586_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_586_, 0, v_x_541_);
lean_ctor_set(v___x_586_, 1, v_x_542_);
v___y_556_ = v___x_586_;
goto v___jp_555_;
}
}
v___jp_555_:
{
lean_object* v___x_557_; lean_object* v___x_559_; 
v___x_557_ = lean_array_fset(v_xs_x27_554_, v_j_546_, v___y_556_);
lean_dec(v_j_546_);
if (v_isShared_551_ == 0)
{
lean_ctor_set(v___x_550_, 0, v___x_557_);
v___x_559_ = v___x_550_;
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
}
}
}
else
{
lean_object* v_ks_589_; lean_object* v_vs_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_610_; 
v_ks_589_ = lean_ctor_get(v_x_538_, 0);
v_vs_590_ = lean_ctor_get(v_x_538_, 1);
v_isSharedCheck_610_ = !lean_is_exclusive(v_x_538_);
if (v_isSharedCheck_610_ == 0)
{
v___x_592_ = v_x_538_;
v_isShared_593_ = v_isSharedCheck_610_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_vs_590_);
lean_inc(v_ks_589_);
lean_dec(v_x_538_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_610_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_ks_589_);
lean_ctor_set(v_reuseFailAlloc_609_, 1, v_vs_590_);
v___x_595_ = v_reuseFailAlloc_609_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
lean_object* v_newNode_596_; uint8_t v___y_598_; size_t v___x_604_; uint8_t v___x_605_; 
v_newNode_596_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17___redArg(v___x_595_, v_x_541_, v_x_542_);
v___x_604_ = ((size_t)7ULL);
v___x_605_ = lean_usize_dec_le(v___x_604_, v_x_540_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; lean_object* v___x_607_; uint8_t v___x_608_; 
v___x_606_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_596_);
v___x_607_ = lean_unsigned_to_nat(4u);
v___x_608_ = lean_nat_dec_lt(v___x_606_, v___x_607_);
lean_dec(v___x_606_);
v___y_598_ = v___x_608_;
goto v___jp_597_;
}
else
{
v___y_598_ = v___x_605_;
goto v___jp_597_;
}
v___jp_597_:
{
if (v___y_598_ == 0)
{
lean_object* v_ks_599_; lean_object* v_vs_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v_ks_599_ = lean_ctor_get(v_newNode_596_, 0);
lean_inc_ref(v_ks_599_);
v_vs_600_ = lean_ctor_get(v_newNode_596_, 1);
lean_inc_ref(v_vs_600_);
lean_dec_ref(v_newNode_596_);
v___x_601_ = lean_unsigned_to_nat(0u);
v___x_602_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0);
v___x_603_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg(v_x_540_, v_ks_599_, v_vs_600_, v___x_601_, v___x_602_);
lean_dec_ref(v_vs_600_);
lean_dec_ref(v_ks_599_);
return v___x_603_;
}
else
{
return v_newNode_596_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg(size_t v_depth_611_, lean_object* v_keys_612_, lean_object* v_vals_613_, lean_object* v_i_614_, lean_object* v_entries_615_){
_start:
{
lean_object* v___x_616_; uint8_t v___x_617_; 
v___x_616_ = lean_array_get_size(v_keys_612_);
v___x_617_ = lean_nat_dec_lt(v_i_614_, v___x_616_);
if (v___x_617_ == 0)
{
lean_dec(v_i_614_);
return v_entries_615_;
}
else
{
lean_object* v_k_618_; lean_object* v_v_619_; uint64_t v___x_620_; size_t v_h_621_; size_t v___x_622_; lean_object* v___x_623_; size_t v___x_624_; size_t v___x_625_; size_t v___x_626_; size_t v_h_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v_k_618_ = lean_array_fget_borrowed(v_keys_612_, v_i_614_);
v_v_619_ = lean_array_fget_borrowed(v_vals_613_, v_i_614_);
v___x_620_ = l_Lean_instHashableLevelMVarId_hash(v_k_618_);
v_h_621_ = lean_uint64_to_usize(v___x_620_);
v___x_622_ = ((size_t)5ULL);
v___x_623_ = lean_unsigned_to_nat(1u);
v___x_624_ = ((size_t)1ULL);
v___x_625_ = lean_usize_sub(v_depth_611_, v___x_624_);
v___x_626_ = lean_usize_mul(v___x_622_, v___x_625_);
v_h_627_ = lean_usize_shift_right(v_h_621_, v___x_626_);
v___x_628_ = lean_nat_add(v_i_614_, v___x_623_);
lean_dec(v_i_614_);
lean_inc(v_v_619_);
lean_inc(v_k_618_);
v___x_629_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(v_entries_615_, v_h_627_, v_depth_611_, v_k_618_, v_v_619_);
v_i_614_ = v___x_628_;
v_entries_615_ = v___x_629_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg___boxed(lean_object* v_depth_631_, lean_object* v_keys_632_, lean_object* v_vals_633_, lean_object* v_i_634_, lean_object* v_entries_635_){
_start:
{
size_t v_depth_boxed_636_; lean_object* v_res_637_; 
v_depth_boxed_636_ = lean_unbox_usize(v_depth_631_);
lean_dec(v_depth_631_);
v_res_637_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg(v_depth_boxed_636_, v_keys_632_, v_vals_633_, v_i_634_, v_entries_635_);
lean_dec_ref(v_vals_633_);
lean_dec_ref(v_keys_632_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___boxed(lean_object* v_x_638_, lean_object* v_x_639_, lean_object* v_x_640_, lean_object* v_x_641_, lean_object* v_x_642_){
_start:
{
size_t v_x_7810__boxed_643_; size_t v_x_7811__boxed_644_; lean_object* v_res_645_; 
v_x_7810__boxed_643_ = lean_unbox_usize(v_x_639_);
lean_dec(v_x_639_);
v_x_7811__boxed_644_ = lean_unbox_usize(v_x_640_);
lean_dec(v_x_640_);
v_res_645_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(v_x_638_, v_x_7810__boxed_643_, v_x_7811__boxed_644_, v_x_641_, v_x_642_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4___redArg(lean_object* v_x_646_, lean_object* v_x_647_, lean_object* v_x_648_){
_start:
{
uint64_t v___x_649_; size_t v___x_650_; size_t v___x_651_; lean_object* v___x_652_; 
v___x_649_ = l_Lean_instHashableLevelMVarId_hash(v_x_647_);
v___x_650_ = lean_uint64_to_usize(v___x_649_);
v___x_651_ = ((size_t)1ULL);
v___x_652_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(v_x_646_, v___x_650_, v___x_651_, v_x_647_, v_x_648_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg(lean_object* v_mvarId_653_, lean_object* v_val_654_, lean_object* v___y_655_){
_start:
{
lean_object* v___x_657_; lean_object* v_mctx_658_; lean_object* v_cache_659_; lean_object* v_zetaDeltaFVarIds_660_; lean_object* v_postponed_661_; lean_object* v_diag_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_690_; 
v___x_657_ = lean_st_ref_take(v___y_655_);
v_mctx_658_ = lean_ctor_get(v___x_657_, 0);
v_cache_659_ = lean_ctor_get(v___x_657_, 1);
v_zetaDeltaFVarIds_660_ = lean_ctor_get(v___x_657_, 2);
v_postponed_661_ = lean_ctor_get(v___x_657_, 3);
v_diag_662_ = lean_ctor_get(v___x_657_, 4);
v_isSharedCheck_690_ = !lean_is_exclusive(v___x_657_);
if (v_isSharedCheck_690_ == 0)
{
v___x_664_ = v___x_657_;
v_isShared_665_ = v_isSharedCheck_690_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_diag_662_);
lean_inc(v_postponed_661_);
lean_inc(v_zetaDeltaFVarIds_660_);
lean_inc(v_cache_659_);
lean_inc(v_mctx_658_);
lean_dec(v___x_657_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_690_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
lean_object* v_depth_666_; lean_object* v_levelAssignDepth_667_; lean_object* v_lmvarCounter_668_; lean_object* v_mvarCounter_669_; lean_object* v_lDecls_670_; lean_object* v_decls_671_; lean_object* v_userNames_672_; lean_object* v_lAssignment_673_; lean_object* v_eAssignment_674_; lean_object* v_dAssignment_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_689_; 
v_depth_666_ = lean_ctor_get(v_mctx_658_, 0);
v_levelAssignDepth_667_ = lean_ctor_get(v_mctx_658_, 1);
v_lmvarCounter_668_ = lean_ctor_get(v_mctx_658_, 2);
v_mvarCounter_669_ = lean_ctor_get(v_mctx_658_, 3);
v_lDecls_670_ = lean_ctor_get(v_mctx_658_, 4);
v_decls_671_ = lean_ctor_get(v_mctx_658_, 5);
v_userNames_672_ = lean_ctor_get(v_mctx_658_, 6);
v_lAssignment_673_ = lean_ctor_get(v_mctx_658_, 7);
v_eAssignment_674_ = lean_ctor_get(v_mctx_658_, 8);
v_dAssignment_675_ = lean_ctor_get(v_mctx_658_, 9);
v_isSharedCheck_689_ = !lean_is_exclusive(v_mctx_658_);
if (v_isSharedCheck_689_ == 0)
{
v___x_677_ = v_mctx_658_;
v_isShared_678_ = v_isSharedCheck_689_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_dAssignment_675_);
lean_inc(v_eAssignment_674_);
lean_inc(v_lAssignment_673_);
lean_inc(v_userNames_672_);
lean_inc(v_decls_671_);
lean_inc(v_lDecls_670_);
lean_inc(v_mvarCounter_669_);
lean_inc(v_lmvarCounter_668_);
lean_inc(v_levelAssignDepth_667_);
lean_inc(v_depth_666_);
lean_dec(v_mctx_658_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_689_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v___x_679_; lean_object* v___x_681_; 
v___x_679_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4___redArg(v_lAssignment_673_, v_mvarId_653_, v_val_654_);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 7, v___x_679_);
v___x_681_ = v___x_677_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v_depth_666_);
lean_ctor_set(v_reuseFailAlloc_688_, 1, v_levelAssignDepth_667_);
lean_ctor_set(v_reuseFailAlloc_688_, 2, v_lmvarCounter_668_);
lean_ctor_set(v_reuseFailAlloc_688_, 3, v_mvarCounter_669_);
lean_ctor_set(v_reuseFailAlloc_688_, 4, v_lDecls_670_);
lean_ctor_set(v_reuseFailAlloc_688_, 5, v_decls_671_);
lean_ctor_set(v_reuseFailAlloc_688_, 6, v_userNames_672_);
lean_ctor_set(v_reuseFailAlloc_688_, 7, v___x_679_);
lean_ctor_set(v_reuseFailAlloc_688_, 8, v_eAssignment_674_);
lean_ctor_set(v_reuseFailAlloc_688_, 9, v_dAssignment_675_);
v___x_681_ = v_reuseFailAlloc_688_;
goto v_reusejp_680_;
}
v_reusejp_680_:
{
lean_object* v___x_683_; 
if (v_isShared_665_ == 0)
{
lean_ctor_set(v___x_664_, 0, v___x_681_);
v___x_683_ = v___x_664_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_681_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v_cache_659_);
lean_ctor_set(v_reuseFailAlloc_687_, 2, v_zetaDeltaFVarIds_660_);
lean_ctor_set(v_reuseFailAlloc_687_, 3, v_postponed_661_);
lean_ctor_set(v_reuseFailAlloc_687_, 4, v_diag_662_);
v___x_683_ = v_reuseFailAlloc_687_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_684_ = lean_st_ref_set(v___y_655_, v___x_683_);
v___x_685_ = lean_box(0);
v___x_686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_686_, 0, v___x_685_);
return v___x_686_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg___boxed(lean_object* v_mvarId_691_, lean_object* v_val_692_, lean_object* v___y_693_, lean_object* v___y_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg(v_mvarId_691_, v_val_692_, v___y_693_);
lean_dec(v___y_693_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(lean_object* v_subst_696_, lean_object* v___x_697_, lean_object* v_range_698_, lean_object* v_b_699_, lean_object* v_i_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_){
_start:
{
lean_object* v_stop_706_; lean_object* v_step_707_; uint8_t v___x_708_; 
v_stop_706_ = lean_ctor_get(v_range_698_, 1);
v_step_707_ = lean_ctor_get(v_range_698_, 2);
v___x_708_ = lean_nat_dec_lt(v_i_700_, v_stop_706_);
if (v___x_708_ == 0)
{
lean_object* v___x_709_; 
lean_dec(v_i_700_);
v___x_709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_709_, 0, v_b_699_);
return v___x_709_;
}
else
{
lean_object* v___x_710_; lean_object* v___x_714_; 
v___x_710_ = lean_box(0);
v___x_714_ = lp_aesop_Aesop_Substitution_findLevel_x3f(v_i_700_, v_subst_696_);
if (lean_obj_tag(v___x_714_) == 1)
{
lean_object* v_val_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v_val_715_ = lean_ctor_get(v___x_714_, 0);
lean_inc(v_val_715_);
lean_dec_ref_known(v___x_714_, 1);
v___x_716_ = lean_array_fget_borrowed(v___x_697_, v_i_700_);
lean_inc(v___x_716_);
v___x_717_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg(v___x_716_, v_val_715_, v___y_702_);
lean_dec_ref(v___x_717_);
goto v___jp_711_;
}
else
{
lean_dec(v___x_714_);
goto v___jp_711_;
}
v___jp_711_:
{
lean_object* v___x_712_; 
v___x_712_ = lean_nat_add(v_i_700_, v_step_707_);
lean_dec(v_i_700_);
v_b_699_ = v___x_710_;
v_i_700_ = v___x_712_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg___boxed(lean_object* v_subst_718_, lean_object* v___x_719_, lean_object* v_range_720_, lean_object* v_b_721_, lean_object* v_i_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(v_subst_718_, v___x_719_, v_range_720_, v_b_721_, v_i_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
lean_dec_ref(v___y_723_);
lean_dec_ref(v_range_720_);
lean_dec_ref(v___x_719_);
lean_dec_ref(v_subst_718_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11(lean_object* v_as_729_, size_t v_i_730_, size_t v_stop_731_, lean_object* v_b_732_){
_start:
{
lean_object* v___y_734_; uint8_t v___x_738_; 
v___x_738_ = lean_usize_dec_eq(v_i_730_, v_stop_731_);
if (v___x_738_ == 0)
{
lean_object* v___x_739_; 
v___x_739_ = lean_array_uget_borrowed(v_as_729_, v_i_730_);
if (lean_obj_tag(v___x_739_) == 0)
{
v___y_734_ = v_b_732_;
goto v___jp_733_;
}
else
{
lean_object* v_val_740_; lean_object* v___x_741_; 
v_val_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_val_740_);
v___x_741_ = lean_array_push(v_b_732_, v_val_740_);
v___y_734_ = v___x_741_;
goto v___jp_733_;
}
}
else
{
return v_b_732_;
}
v___jp_733_:
{
size_t v___x_735_; size_t v___x_736_; 
v___x_735_ = ((size_t)1ULL);
v___x_736_ = lean_usize_add(v_i_730_, v___x_735_);
v_i_730_ = v___x_736_;
v_b_732_ = v___y_734_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11___boxed(lean_object* v_as_742_, lean_object* v_i_743_, lean_object* v_stop_744_, lean_object* v_b_745_){
_start:
{
size_t v_i_boxed_746_; size_t v_stop_boxed_747_; lean_object* v_res_748_; 
v_i_boxed_746_ = lean_unbox_usize(v_i_743_);
lean_dec(v_i_743_);
v_stop_boxed_747_ = lean_unbox_usize(v_stop_744_);
lean_dec(v_stop_744_);
v_res_748_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11(v_as_742_, v_i_boxed_746_, v_stop_boxed_747_, v_b_745_);
lean_dec_ref(v_as_742_);
return v_res_748_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8(lean_object* v_as_751_, lean_object* v_start_752_, lean_object* v_stop_753_){
_start:
{
lean_object* v___x_754_; uint8_t v___x_755_; 
v___x_754_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___closed__0));
v___x_755_ = lean_nat_dec_lt(v_start_752_, v_stop_753_);
if (v___x_755_ == 0)
{
return v___x_754_;
}
else
{
lean_object* v___x_756_; uint8_t v___x_757_; 
v___x_756_ = lean_array_get_size(v_as_751_);
v___x_757_ = lean_nat_dec_le(v_stop_753_, v___x_756_);
if (v___x_757_ == 0)
{
uint8_t v___x_758_; 
v___x_758_ = lean_nat_dec_lt(v_start_752_, v___x_756_);
if (v___x_758_ == 0)
{
return v___x_754_;
}
else
{
size_t v___x_759_; size_t v___x_760_; lean_object* v___x_761_; 
v___x_759_ = lean_usize_of_nat(v_start_752_);
v___x_760_ = lean_usize_of_nat(v___x_756_);
v___x_761_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11(v_as_751_, v___x_759_, v___x_760_, v___x_754_);
return v___x_761_;
}
}
else
{
size_t v___x_762_; size_t v___x_763_; lean_object* v___x_764_; 
v___x_762_ = lean_usize_of_nat(v_start_752_);
v___x_763_ = lean_usize_of_nat(v_stop_753_);
v___x_764_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8_spec__11(v_as_751_, v___x_762_, v___x_763_, v___x_754_);
return v___x_764_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8___boxed(lean_object* v_as_765_, lean_object* v_start_766_, lean_object* v_stop_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8(v_as_765_, v_start_766_, v_stop_767_);
lean_dec(v_stop_767_);
lean_dec(v_start_766_);
lean_dec_ref(v_as_765_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(size_t v_sz_769_, size_t v_i_770_, lean_object* v_bs_771_){
_start:
{
uint8_t v___x_772_; 
v___x_772_ = lean_usize_dec_lt(v_i_770_, v_sz_769_);
if (v___x_772_ == 0)
{
return v_bs_771_;
}
else
{
lean_object* v_v_773_; lean_object* v___x_774_; lean_object* v_bs_x27_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; size_t v___x_784_; size_t v___x_785_; lean_object* v___x_786_; 
v_v_773_ = lean_array_uget(v_bs_771_, v_i_770_);
v___x_774_ = lean_unsigned_to_nat(0u);
v_bs_x27_775_ = lean_array_uset(v_bs_771_, v_i_770_, v___x_774_);
v___x_776_ = lean_usize_to_nat(v_i_770_);
v___x_777_ = l_Nat_reprFast(v___x_776_);
v___x_778_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_778_, 0, v___x_777_);
v___x_779_ = l_Lean_MessageData_ofFormat(v___x_778_);
v___x_780_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1);
v___x_781_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_781_, 0, v___x_779_);
lean_ctor_set(v___x_781_, 1, v___x_780_);
v___x_782_ = l_Lean_MessageData_ofExpr(v_v_773_);
v___x_783_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_783_, 0, v___x_781_);
lean_ctor_set(v___x_783_, 1, v___x_782_);
v___x_784_ = ((size_t)1ULL);
v___x_785_ = lean_usize_add(v_i_770_, v___x_784_);
v___x_786_ = lean_array_uset(v_bs_x27_775_, v_i_770_, v___x_783_);
v_i_770_ = v___x_785_;
v_bs_771_ = v___x_786_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg___boxed(lean_object* v_sz_788_, lean_object* v_i_789_, lean_object* v_bs_790_){
_start:
{
size_t v_sz_boxed_791_; size_t v_i_boxed_792_; lean_object* v_res_793_; 
v_sz_boxed_791_ = lean_unbox_usize(v_sz_788_);
lean_dec(v_sz_788_);
v_i_boxed_792_ = lean_unbox_usize(v_i_789_);
lean_dec(v_i_789_);
v_res_793_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(v_sz_boxed_791_, v_i_boxed_792_, v_bs_790_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14(lean_object* v_msgData_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v___x_800_; lean_object* v_env_801_; lean_object* v___x_802_; lean_object* v_mctx_803_; lean_object* v_lctx_804_; lean_object* v_options_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_800_ = lean_st_ref_get(v___y_798_);
v_env_801_ = lean_ctor_get(v___x_800_, 0);
lean_inc_ref(v_env_801_);
lean_dec(v___x_800_);
v___x_802_ = lean_st_ref_get(v___y_796_);
v_mctx_803_ = lean_ctor_get(v___x_802_, 0);
lean_inc_ref(v_mctx_803_);
lean_dec(v___x_802_);
v_lctx_804_ = lean_ctor_get(v___y_795_, 2);
v_options_805_ = lean_ctor_get(v___y_797_, 2);
lean_inc_ref(v_options_805_);
lean_inc_ref(v_lctx_804_);
v___x_806_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_806_, 0, v_env_801_);
lean_ctor_set(v___x_806_, 1, v_mctx_803_);
lean_ctor_set(v___x_806_, 2, v_lctx_804_);
lean_ctor_set(v___x_806_, 3, v_options_805_);
v___x_807_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_807_, 0, v___x_806_);
lean_ctor_set(v___x_807_, 1, v_msgData_794_);
v___x_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_808_, 0, v___x_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14___boxed(lean_object* v_msgData_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14(v_msgData_809_, v___y_810_, v___y_811_, v___y_812_, v___y_813_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
lean_dec(v___y_811_);
lean_dec_ref(v___y_810_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(lean_object* v_msg_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v_ref_822_; lean_object* v___x_823_; lean_object* v_a_824_; lean_object* v___x_826_; uint8_t v_isShared_827_; uint8_t v_isSharedCheck_832_; 
v_ref_822_ = lean_ctor_get(v___y_819_, 5);
v___x_823_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10_spec__14(v_msg_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
v_a_824_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_832_ == 0)
{
v___x_826_ = v___x_823_;
v_isShared_827_ = v_isSharedCheck_832_;
goto v_resetjp_825_;
}
else
{
lean_inc(v_a_824_);
lean_dec(v___x_823_);
v___x_826_ = lean_box(0);
v_isShared_827_ = v_isSharedCheck_832_;
goto v_resetjp_825_;
}
v_resetjp_825_:
{
lean_object* v___x_828_; lean_object* v___x_830_; 
lean_inc(v_ref_822_);
v___x_828_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_828_, 0, v_ref_822_);
lean_ctor_set(v___x_828_, 1, v_a_824_);
if (v_isShared_827_ == 0)
{
lean_ctor_set_tag(v___x_826_, 1);
lean_ctor_set(v___x_826_, 0, v___x_828_);
v___x_830_ = v___x_826_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v___x_828_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg___boxed(lean_object* v_msg_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(v_msg_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_);
lean_dec(v___y_837_);
lean_dec_ref(v___y_836_);
lean_dec(v___y_835_);
lean_dec_ref(v___y_834_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(size_t v_sz_840_, size_t v_i_841_, lean_object* v_bs_842_){
_start:
{
uint8_t v___x_843_; 
v___x_843_ = lean_usize_dec_lt(v_i_841_, v_sz_840_);
if (v___x_843_ == 0)
{
return v_bs_842_;
}
else
{
lean_object* v_v_844_; lean_object* v___x_845_; lean_object* v_bs_x27_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; size_t v___x_855_; size_t v___x_856_; lean_object* v___x_857_; 
v_v_844_ = lean_array_uget(v_bs_842_, v_i_841_);
v___x_845_ = lean_unsigned_to_nat(0u);
v_bs_x27_846_ = lean_array_uset(v_bs_842_, v_i_841_, v___x_845_);
v___x_847_ = lean_usize_to_nat(v_i_841_);
v___x_848_ = l_Nat_reprFast(v___x_847_);
v___x_849_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
v___x_850_ = l_Lean_MessageData_ofFormat(v___x_849_);
v___x_851_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__1___closed__1);
v___x_852_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_852_, 0, v___x_850_);
lean_ctor_set(v___x_852_, 1, v___x_851_);
v___x_853_ = l_Lean_MessageData_ofLevel(v_v_844_);
v___x_854_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_854_, 0, v___x_852_);
lean_ctor_set(v___x_854_, 1, v___x_853_);
v___x_855_ = ((size_t)1ULL);
v___x_856_ = lean_usize_add(v_i_841_, v___x_855_);
v___x_857_ = lean_array_uset(v_bs_x27_846_, v_i_841_, v___x_854_);
v_i_841_ = v___x_856_;
v_bs_842_ = v___x_857_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg___boxed(lean_object* v_sz_859_, lean_object* v_i_860_, lean_object* v_bs_861_){
_start:
{
size_t v_sz_boxed_862_; size_t v_i_boxed_863_; lean_object* v_res_864_; 
v_sz_boxed_862_ = lean_unbox_usize(v_sz_859_);
lean_dec(v_sz_859_);
v_i_boxed_863_ = lean_unbox_usize(v_i_860_);
lean_dec(v_i_860_);
v_res_864_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(v_sz_boxed_862_, v_i_boxed_863_, v_bs_861_);
return v_res_864_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1(size_t v_sz_865_, size_t v_i_866_, lean_object* v_bs_867_){
_start:
{
uint8_t v___x_868_; 
v___x_868_ = lean_usize_dec_lt(v_i_866_, v_sz_865_);
if (v___x_868_ == 0)
{
return v_bs_867_;
}
else
{
lean_object* v_v_869_; lean_object* v___x_870_; lean_object* v_bs_x27_871_; lean_object* v___x_872_; size_t v___x_873_; size_t v___x_874_; lean_object* v___x_875_; 
v_v_869_ = lean_array_uget(v_bs_867_, v_i_866_);
v___x_870_ = lean_unsigned_to_nat(0u);
v_bs_x27_871_ = lean_array_uset(v_bs_867_, v_i_866_, v___x_870_);
v___x_872_ = l_Lean_Expr_mvarId_x21(v_v_869_);
lean_dec(v_v_869_);
v___x_873_ = ((size_t)1ULL);
v___x_874_ = lean_usize_add(v_i_866_, v___x_873_);
v___x_875_ = lean_array_uset(v_bs_x27_871_, v_i_866_, v___x_872_);
v_i_866_ = v___x_874_;
v_bs_867_ = v___x_875_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1___boxed(lean_object* v_sz_877_, lean_object* v_i_878_, lean_object* v_bs_879_){
_start:
{
size_t v_sz_boxed_880_; size_t v_i_boxed_881_; lean_object* v_res_882_; 
v_sz_boxed_880_ = lean_unbox_usize(v_sz_877_);
lean_dec(v_sz_877_);
v_i_boxed_881_ = lean_unbox_usize(v_i_878_);
lean_dec(v_i_878_);
v_res_882_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1(v_sz_boxed_880_, v_i_boxed_881_, v_bs_879_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17___redArg(lean_object* v_x_883_, lean_object* v_x_884_, lean_object* v_x_885_, lean_object* v_x_886_){
_start:
{
lean_object* v_ks_887_; lean_object* v_vs_888_; lean_object* v___x_890_; uint8_t v_isShared_891_; uint8_t v_isSharedCheck_912_; 
v_ks_887_ = lean_ctor_get(v_x_883_, 0);
v_vs_888_ = lean_ctor_get(v_x_883_, 1);
v_isSharedCheck_912_ = !lean_is_exclusive(v_x_883_);
if (v_isSharedCheck_912_ == 0)
{
v___x_890_ = v_x_883_;
v_isShared_891_ = v_isSharedCheck_912_;
goto v_resetjp_889_;
}
else
{
lean_inc(v_vs_888_);
lean_inc(v_ks_887_);
lean_dec(v_x_883_);
v___x_890_ = lean_box(0);
v_isShared_891_ = v_isSharedCheck_912_;
goto v_resetjp_889_;
}
v_resetjp_889_:
{
lean_object* v___x_892_; uint8_t v___x_893_; 
v___x_892_ = lean_array_get_size(v_ks_887_);
v___x_893_ = lean_nat_dec_lt(v_x_884_, v___x_892_);
if (v___x_893_ == 0)
{
lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_897_; 
lean_dec(v_x_884_);
v___x_894_ = lean_array_push(v_ks_887_, v_x_885_);
v___x_895_ = lean_array_push(v_vs_888_, v_x_886_);
if (v_isShared_891_ == 0)
{
lean_ctor_set(v___x_890_, 1, v___x_895_);
lean_ctor_set(v___x_890_, 0, v___x_894_);
v___x_897_ = v___x_890_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v___x_894_);
lean_ctor_set(v_reuseFailAlloc_898_, 1, v___x_895_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
else
{
lean_object* v_k_x27_899_; uint8_t v___x_900_; 
v_k_x27_899_ = lean_array_fget_borrowed(v_ks_887_, v_x_884_);
v___x_900_ = l_Lean_instBEqMVarId_beq(v_x_885_, v_k_x27_899_);
if (v___x_900_ == 0)
{
lean_object* v___x_902_; 
if (v_isShared_891_ == 0)
{
v___x_902_ = v___x_890_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_ks_887_);
lean_ctor_set(v_reuseFailAlloc_906_, 1, v_vs_888_);
v___x_902_ = v_reuseFailAlloc_906_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
lean_object* v___x_903_; lean_object* v___x_904_; 
v___x_903_ = lean_unsigned_to_nat(1u);
v___x_904_ = lean_nat_add(v_x_884_, v___x_903_);
lean_dec(v_x_884_);
v_x_883_ = v___x_902_;
v_x_884_ = v___x_904_;
goto _start;
}
}
else
{
lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_910_; 
v___x_907_ = lean_array_fset(v_ks_887_, v_x_884_, v_x_885_);
v___x_908_ = lean_array_fset(v_vs_888_, v_x_884_, v_x_886_);
lean_dec(v_x_884_);
if (v_isShared_891_ == 0)
{
lean_ctor_set(v___x_890_, 1, v___x_908_);
lean_ctor_set(v___x_890_, 0, v___x_907_);
v___x_910_ = v___x_890_;
goto v_reusejp_909_;
}
else
{
lean_object* v_reuseFailAlloc_911_; 
v_reuseFailAlloc_911_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_911_, 0, v___x_907_);
lean_ctor_set(v_reuseFailAlloc_911_, 1, v___x_908_);
v___x_910_ = v_reuseFailAlloc_911_;
goto v_reusejp_909_;
}
v_reusejp_909_:
{
return v___x_910_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13___redArg(lean_object* v_n_913_, lean_object* v_k_914_, lean_object* v_v_915_){
_start:
{
lean_object* v___x_916_; lean_object* v___x_917_; 
v___x_916_ = lean_unsigned_to_nat(0u);
v___x_917_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17___redArg(v_n_913_, v___x_916_, v_k_914_, v_v_915_);
return v___x_917_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(lean_object* v_x_918_, size_t v_x_919_, size_t v_x_920_, lean_object* v_x_921_, lean_object* v_x_922_){
_start:
{
if (lean_obj_tag(v_x_918_) == 0)
{
lean_object* v_es_923_; size_t v___x_924_; size_t v___x_925_; lean_object* v_j_926_; lean_object* v___x_927_; uint8_t v___x_928_; 
v_es_923_ = lean_ctor_get(v_x_918_, 0);
v___x_924_ = ((size_t)31ULL);
v___x_925_ = lean_usize_land(v_x_919_, v___x_924_);
v_j_926_ = lean_usize_to_nat(v___x_925_);
v___x_927_ = lean_array_get_size(v_es_923_);
v___x_928_ = lean_nat_dec_lt(v_j_926_, v___x_927_);
if (v___x_928_ == 0)
{
lean_dec(v_j_926_);
lean_dec(v_x_922_);
lean_dec(v_x_921_);
return v_x_918_;
}
else
{
lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_967_; 
lean_inc_ref(v_es_923_);
v_isSharedCheck_967_ = !lean_is_exclusive(v_x_918_);
if (v_isSharedCheck_967_ == 0)
{
lean_object* v_unused_968_; 
v_unused_968_ = lean_ctor_get(v_x_918_, 0);
lean_dec(v_unused_968_);
v___x_930_ = v_x_918_;
v_isShared_931_ = v_isSharedCheck_967_;
goto v_resetjp_929_;
}
else
{
lean_dec(v_x_918_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_967_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v_v_932_; lean_object* v___x_933_; lean_object* v_xs_x27_934_; lean_object* v___y_936_; 
v_v_932_ = lean_array_fget(v_es_923_, v_j_926_);
v___x_933_ = lean_box(0);
v_xs_x27_934_ = lean_array_fset(v_es_923_, v_j_926_, v___x_933_);
switch(lean_obj_tag(v_v_932_))
{
case 0:
{
lean_object* v_key_941_; lean_object* v_val_942_; lean_object* v___x_944_; uint8_t v_isShared_945_; uint8_t v_isSharedCheck_952_; 
v_key_941_ = lean_ctor_get(v_v_932_, 0);
v_val_942_ = lean_ctor_get(v_v_932_, 1);
v_isSharedCheck_952_ = !lean_is_exclusive(v_v_932_);
if (v_isSharedCheck_952_ == 0)
{
v___x_944_ = v_v_932_;
v_isShared_945_ = v_isSharedCheck_952_;
goto v_resetjp_943_;
}
else
{
lean_inc(v_val_942_);
lean_inc(v_key_941_);
lean_dec(v_v_932_);
v___x_944_ = lean_box(0);
v_isShared_945_ = v_isSharedCheck_952_;
goto v_resetjp_943_;
}
v_resetjp_943_:
{
uint8_t v___x_946_; 
v___x_946_ = l_Lean_instBEqMVarId_beq(v_x_921_, v_key_941_);
if (v___x_946_ == 0)
{
lean_object* v___x_947_; lean_object* v___x_948_; 
lean_del_object(v___x_944_);
v___x_947_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_941_, v_val_942_, v_x_921_, v_x_922_);
v___x_948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_948_, 0, v___x_947_);
v___y_936_ = v___x_948_;
goto v___jp_935_;
}
else
{
lean_object* v___x_950_; 
lean_dec(v_val_942_);
lean_dec(v_key_941_);
if (v_isShared_945_ == 0)
{
lean_ctor_set(v___x_944_, 1, v_x_922_);
lean_ctor_set(v___x_944_, 0, v_x_921_);
v___x_950_ = v___x_944_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v_x_921_);
lean_ctor_set(v_reuseFailAlloc_951_, 1, v_x_922_);
v___x_950_ = v_reuseFailAlloc_951_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
v___y_936_ = v___x_950_;
goto v___jp_935_;
}
}
}
}
case 1:
{
lean_object* v_node_953_; lean_object* v___x_955_; uint8_t v_isShared_956_; uint8_t v_isSharedCheck_965_; 
v_node_953_ = lean_ctor_get(v_v_932_, 0);
v_isSharedCheck_965_ = !lean_is_exclusive(v_v_932_);
if (v_isSharedCheck_965_ == 0)
{
v___x_955_ = v_v_932_;
v_isShared_956_ = v_isSharedCheck_965_;
goto v_resetjp_954_;
}
else
{
lean_inc(v_node_953_);
lean_dec(v_v_932_);
v___x_955_ = lean_box(0);
v_isShared_956_ = v_isSharedCheck_965_;
goto v_resetjp_954_;
}
v_resetjp_954_:
{
size_t v___x_957_; size_t v___x_958_; size_t v___x_959_; size_t v___x_960_; lean_object* v___x_961_; lean_object* v___x_963_; 
v___x_957_ = ((size_t)5ULL);
v___x_958_ = lean_usize_shift_right(v_x_919_, v___x_957_);
v___x_959_ = ((size_t)1ULL);
v___x_960_ = lean_usize_add(v_x_920_, v___x_959_);
v___x_961_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(v_node_953_, v___x_958_, v___x_960_, v_x_921_, v_x_922_);
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 0, v___x_961_);
v___x_963_ = v___x_955_;
goto v_reusejp_962_;
}
else
{
lean_object* v_reuseFailAlloc_964_; 
v_reuseFailAlloc_964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_964_, 0, v___x_961_);
v___x_963_ = v_reuseFailAlloc_964_;
goto v_reusejp_962_;
}
v_reusejp_962_:
{
v___y_936_ = v___x_963_;
goto v___jp_935_;
}
}
}
default: 
{
lean_object* v___x_966_; 
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v_x_921_);
lean_ctor_set(v___x_966_, 1, v_x_922_);
v___y_936_ = v___x_966_;
goto v___jp_935_;
}
}
v___jp_935_:
{
lean_object* v___x_937_; lean_object* v___x_939_; 
v___x_937_ = lean_array_fset(v_xs_x27_934_, v_j_926_, v___y_936_);
lean_dec(v_j_926_);
if (v_isShared_931_ == 0)
{
lean_ctor_set(v___x_930_, 0, v___x_937_);
v___x_939_ = v___x_930_;
goto v_reusejp_938_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v___x_937_);
v___x_939_ = v_reuseFailAlloc_940_;
goto v_reusejp_938_;
}
v_reusejp_938_:
{
return v___x_939_;
}
}
}
}
}
else
{
lean_object* v_ks_969_; lean_object* v_vs_970_; lean_object* v___x_972_; uint8_t v_isShared_973_; uint8_t v_isSharedCheck_990_; 
v_ks_969_ = lean_ctor_get(v_x_918_, 0);
v_vs_970_ = lean_ctor_get(v_x_918_, 1);
v_isSharedCheck_990_ = !lean_is_exclusive(v_x_918_);
if (v_isSharedCheck_990_ == 0)
{
v___x_972_ = v_x_918_;
v_isShared_973_ = v_isSharedCheck_990_;
goto v_resetjp_971_;
}
else
{
lean_inc(v_vs_970_);
lean_inc(v_ks_969_);
lean_dec(v_x_918_);
v___x_972_ = lean_box(0);
v_isShared_973_ = v_isSharedCheck_990_;
goto v_resetjp_971_;
}
v_resetjp_971_:
{
lean_object* v___x_975_; 
if (v_isShared_973_ == 0)
{
v___x_975_ = v___x_972_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v_ks_969_);
lean_ctor_set(v_reuseFailAlloc_989_, 1, v_vs_970_);
v___x_975_ = v_reuseFailAlloc_989_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
lean_object* v_newNode_976_; uint8_t v___y_978_; size_t v___x_984_; uint8_t v___x_985_; 
v_newNode_976_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13___redArg(v___x_975_, v_x_921_, v_x_922_);
v___x_984_ = ((size_t)7ULL);
v___x_985_ = lean_usize_dec_le(v___x_984_, v_x_920_);
if (v___x_985_ == 0)
{
lean_object* v___x_986_; lean_object* v___x_987_; uint8_t v___x_988_; 
v___x_986_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_976_);
v___x_987_ = lean_unsigned_to_nat(4u);
v___x_988_ = lean_nat_dec_lt(v___x_986_, v___x_987_);
lean_dec(v___x_986_);
v___y_978_ = v___x_988_;
goto v___jp_977_;
}
else
{
v___y_978_ = v___x_985_;
goto v___jp_977_;
}
v___jp_977_:
{
if (v___y_978_ == 0)
{
lean_object* v_ks_979_; lean_object* v_vs_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v_ks_979_ = lean_ctor_get(v_newNode_976_, 0);
lean_inc_ref(v_ks_979_);
v_vs_980_ = lean_ctor_get(v_newNode_976_, 1);
lean_inc_ref(v_vs_980_);
lean_dec_ref(v_newNode_976_);
v___x_981_ = lean_unsigned_to_nat(0u);
v___x_982_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg___closed__0);
v___x_983_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg(v_x_920_, v_ks_979_, v_vs_980_, v___x_981_, v___x_982_);
lean_dec_ref(v_vs_980_);
lean_dec_ref(v_ks_979_);
return v___x_983_;
}
else
{
return v_newNode_976_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg(size_t v_depth_991_, lean_object* v_keys_992_, lean_object* v_vals_993_, lean_object* v_i_994_, lean_object* v_entries_995_){
_start:
{
lean_object* v___x_996_; uint8_t v___x_997_; 
v___x_996_ = lean_array_get_size(v_keys_992_);
v___x_997_ = lean_nat_dec_lt(v_i_994_, v___x_996_);
if (v___x_997_ == 0)
{
lean_dec(v_i_994_);
return v_entries_995_;
}
else
{
lean_object* v_k_998_; lean_object* v_v_999_; uint64_t v___x_1000_; size_t v_h_1001_; size_t v___x_1002_; lean_object* v___x_1003_; size_t v___x_1004_; size_t v___x_1005_; size_t v___x_1006_; size_t v_h_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; 
v_k_998_ = lean_array_fget_borrowed(v_keys_992_, v_i_994_);
v_v_999_ = lean_array_fget_borrowed(v_vals_993_, v_i_994_);
v___x_1000_ = l_Lean_instHashableMVarId_hash(v_k_998_);
v_h_1001_ = lean_uint64_to_usize(v___x_1000_);
v___x_1002_ = ((size_t)5ULL);
v___x_1003_ = lean_unsigned_to_nat(1u);
v___x_1004_ = ((size_t)1ULL);
v___x_1005_ = lean_usize_sub(v_depth_991_, v___x_1004_);
v___x_1006_ = lean_usize_mul(v___x_1002_, v___x_1005_);
v_h_1007_ = lean_usize_shift_right(v_h_1001_, v___x_1006_);
v___x_1008_ = lean_nat_add(v_i_994_, v___x_1003_);
lean_dec(v_i_994_);
lean_inc(v_v_999_);
lean_inc(v_k_998_);
v___x_1009_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(v_entries_995_, v_h_1007_, v_depth_991_, v_k_998_, v_v_999_);
v_i_994_ = v___x_1008_;
v_entries_995_ = v___x_1009_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg___boxed(lean_object* v_depth_1011_, lean_object* v_keys_1012_, lean_object* v_vals_1013_, lean_object* v_i_1014_, lean_object* v_entries_1015_){
_start:
{
size_t v_depth_boxed_1016_; lean_object* v_res_1017_; 
v_depth_boxed_1016_ = lean_unbox_usize(v_depth_1011_);
lean_dec(v_depth_1011_);
v_res_1017_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg(v_depth_boxed_1016_, v_keys_1012_, v_vals_1013_, v_i_1014_, v_entries_1015_);
lean_dec_ref(v_vals_1013_);
lean_dec_ref(v_keys_1012_);
return v_res_1017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_x_1018_, lean_object* v_x_1019_, lean_object* v_x_1020_, lean_object* v_x_1021_, lean_object* v_x_1022_){
_start:
{
size_t v_x_8324__boxed_1023_; size_t v_x_8325__boxed_1024_; lean_object* v_res_1025_; 
v_x_8324__boxed_1023_ = lean_unbox_usize(v_x_1019_);
lean_dec(v_x_1019_);
v_x_8325__boxed_1024_ = lean_unbox_usize(v_x_1020_);
lean_dec(v_x_1020_);
v_res_1025_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(v_x_1018_, v_x_8324__boxed_1023_, v_x_8325__boxed_1024_, v_x_1021_, v_x_1022_);
return v_res_1025_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2___redArg(lean_object* v_x_1026_, lean_object* v_x_1027_, lean_object* v_x_1028_){
_start:
{
uint64_t v___x_1029_; size_t v___x_1030_; size_t v___x_1031_; lean_object* v___x_1032_; 
v___x_1029_ = l_Lean_instHashableMVarId_hash(v_x_1027_);
v___x_1030_ = lean_uint64_to_usize(v___x_1029_);
v___x_1031_ = ((size_t)1ULL);
v___x_1032_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(v_x_1026_, v___x_1030_, v___x_1031_, v_x_1027_, v_x_1028_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg(lean_object* v_mvarId_1033_, lean_object* v_val_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v___x_1037_; lean_object* v_mctx_1038_; lean_object* v_cache_1039_; lean_object* v_zetaDeltaFVarIds_1040_; lean_object* v_postponed_1041_; lean_object* v_diag_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1070_; 
v___x_1037_ = lean_st_ref_take(v___y_1035_);
v_mctx_1038_ = lean_ctor_get(v___x_1037_, 0);
v_cache_1039_ = lean_ctor_get(v___x_1037_, 1);
v_zetaDeltaFVarIds_1040_ = lean_ctor_get(v___x_1037_, 2);
v_postponed_1041_ = lean_ctor_get(v___x_1037_, 3);
v_diag_1042_ = lean_ctor_get(v___x_1037_, 4);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1037_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1044_ = v___x_1037_;
v_isShared_1045_ = v_isSharedCheck_1070_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_diag_1042_);
lean_inc(v_postponed_1041_);
lean_inc(v_zetaDeltaFVarIds_1040_);
lean_inc(v_cache_1039_);
lean_inc(v_mctx_1038_);
lean_dec(v___x_1037_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1070_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v_depth_1046_; lean_object* v_levelAssignDepth_1047_; lean_object* v_lmvarCounter_1048_; lean_object* v_mvarCounter_1049_; lean_object* v_lDecls_1050_; lean_object* v_decls_1051_; lean_object* v_userNames_1052_; lean_object* v_lAssignment_1053_; lean_object* v_eAssignment_1054_; lean_object* v_dAssignment_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1069_; 
v_depth_1046_ = lean_ctor_get(v_mctx_1038_, 0);
v_levelAssignDepth_1047_ = lean_ctor_get(v_mctx_1038_, 1);
v_lmvarCounter_1048_ = lean_ctor_get(v_mctx_1038_, 2);
v_mvarCounter_1049_ = lean_ctor_get(v_mctx_1038_, 3);
v_lDecls_1050_ = lean_ctor_get(v_mctx_1038_, 4);
v_decls_1051_ = lean_ctor_get(v_mctx_1038_, 5);
v_userNames_1052_ = lean_ctor_get(v_mctx_1038_, 6);
v_lAssignment_1053_ = lean_ctor_get(v_mctx_1038_, 7);
v_eAssignment_1054_ = lean_ctor_get(v_mctx_1038_, 8);
v_dAssignment_1055_ = lean_ctor_get(v_mctx_1038_, 9);
v_isSharedCheck_1069_ = !lean_is_exclusive(v_mctx_1038_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1057_ = v_mctx_1038_;
v_isShared_1058_ = v_isSharedCheck_1069_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_dAssignment_1055_);
lean_inc(v_eAssignment_1054_);
lean_inc(v_lAssignment_1053_);
lean_inc(v_userNames_1052_);
lean_inc(v_decls_1051_);
lean_inc(v_lDecls_1050_);
lean_inc(v_mvarCounter_1049_);
lean_inc(v_lmvarCounter_1048_);
lean_inc(v_levelAssignDepth_1047_);
lean_inc(v_depth_1046_);
lean_dec(v_mctx_1038_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1069_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
lean_object* v___x_1059_; lean_object* v___x_1061_; 
v___x_1059_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2___redArg(v_eAssignment_1054_, v_mvarId_1033_, v_val_1034_);
if (v_isShared_1058_ == 0)
{
lean_ctor_set(v___x_1057_, 8, v___x_1059_);
v___x_1061_ = v___x_1057_;
goto v_reusejp_1060_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v_depth_1046_);
lean_ctor_set(v_reuseFailAlloc_1068_, 1, v_levelAssignDepth_1047_);
lean_ctor_set(v_reuseFailAlloc_1068_, 2, v_lmvarCounter_1048_);
lean_ctor_set(v_reuseFailAlloc_1068_, 3, v_mvarCounter_1049_);
lean_ctor_set(v_reuseFailAlloc_1068_, 4, v_lDecls_1050_);
lean_ctor_set(v_reuseFailAlloc_1068_, 5, v_decls_1051_);
lean_ctor_set(v_reuseFailAlloc_1068_, 6, v_userNames_1052_);
lean_ctor_set(v_reuseFailAlloc_1068_, 7, v_lAssignment_1053_);
lean_ctor_set(v_reuseFailAlloc_1068_, 8, v___x_1059_);
lean_ctor_set(v_reuseFailAlloc_1068_, 9, v_dAssignment_1055_);
v___x_1061_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1060_;
}
v_reusejp_1060_:
{
lean_object* v___x_1063_; 
if (v_isShared_1045_ == 0)
{
lean_ctor_set(v___x_1044_, 0, v___x_1061_);
v___x_1063_ = v___x_1044_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1067_; 
v_reuseFailAlloc_1067_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1067_, 0, v___x_1061_);
lean_ctor_set(v_reuseFailAlloc_1067_, 1, v_cache_1039_);
lean_ctor_set(v_reuseFailAlloc_1067_, 2, v_zetaDeltaFVarIds_1040_);
lean_ctor_set(v_reuseFailAlloc_1067_, 3, v_postponed_1041_);
lean_ctor_set(v_reuseFailAlloc_1067_, 4, v_diag_1042_);
v___x_1063_ = v_reuseFailAlloc_1067_;
goto v_reusejp_1062_;
}
v_reusejp_1062_:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v___x_1064_ = lean_st_ref_set(v___y_1035_, v___x_1063_);
v___x_1065_ = lean_box(0);
v___x_1066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1065_);
return v___x_1066_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg___boxed(lean_object* v_mvarId_1071_, lean_object* v_val_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_){
_start:
{
lean_object* v_res_1075_; 
v_res_1075_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg(v_mvarId_1071_, v_val_1072_, v___y_1073_);
lean_dec(v___y_1073_);
return v_res_1075_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg(lean_object* v_subst_1076_, lean_object* v___x_1077_, lean_object* v_range_1078_, lean_object* v_b_1079_, lean_object* v_i_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_){
_start:
{
lean_object* v_stop_1086_; lean_object* v_step_1087_; uint8_t v___x_1088_; 
v_stop_1086_ = lean_ctor_get(v_range_1078_, 1);
v_step_1087_ = lean_ctor_get(v_range_1078_, 2);
v___x_1088_ = lean_nat_dec_lt(v_i_1080_, v_stop_1086_);
if (v___x_1088_ == 0)
{
lean_object* v___x_1089_; 
lean_dec(v_i_1080_);
v___x_1089_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1089_, 0, v_b_1079_);
return v___x_1089_;
}
else
{
lean_object* v___x_1090_; lean_object* v___x_1094_; 
v___x_1090_ = lean_box(0);
v___x_1094_ = lp_aesop_Aesop_Substitution_find_x3f(v_i_1080_, v_subst_1076_);
if (lean_obj_tag(v___x_1094_) == 1)
{
lean_object* v_val_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; 
v_val_1095_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_val_1095_);
lean_dec_ref_known(v___x_1094_, 1);
v___x_1096_ = lean_array_fget_borrowed(v___x_1077_, v_i_1080_);
lean_inc(v___x_1096_);
v___x_1097_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg(v___x_1096_, v_val_1095_, v___y_1082_);
lean_dec_ref(v___x_1097_);
goto v___jp_1091_;
}
else
{
lean_dec(v___x_1094_);
goto v___jp_1091_;
}
v___jp_1091_:
{
lean_object* v___x_1092_; 
v___x_1092_ = lean_nat_add(v_i_1080_, v_step_1087_);
lean_dec(v_i_1080_);
v_b_1079_ = v___x_1090_;
v_i_1080_ = v___x_1092_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg___boxed(lean_object* v_subst_1098_, lean_object* v___x_1099_, lean_object* v_range_1100_, lean_object* v_b_1101_, lean_object* v_i_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_){
_start:
{
lean_object* v_res_1108_; 
v_res_1108_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg(v_subst_1098_, v___x_1099_, v_range_1100_, v_b_1101_, v_i_1102_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
lean_dec(v___y_1106_);
lean_dec_ref(v___y_1105_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec_ref(v_range_1100_);
lean_dec_ref(v___x_1099_);
lean_dec_ref(v_subst_1098_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8(lean_object* v_as_1109_, size_t v_i_1110_, size_t v_stop_1111_, lean_object* v_b_1112_){
_start:
{
lean_object* v___y_1114_; uint8_t v___x_1118_; 
v___x_1118_ = lean_usize_dec_eq(v_i_1110_, v_stop_1111_);
if (v___x_1118_ == 0)
{
lean_object* v___x_1119_; 
v___x_1119_ = lean_array_uget_borrowed(v_as_1109_, v_i_1110_);
if (lean_obj_tag(v___x_1119_) == 0)
{
v___y_1114_ = v_b_1112_;
goto v___jp_1113_;
}
else
{
lean_object* v_val_1120_; lean_object* v___x_1121_; 
v_val_1120_ = lean_ctor_get(v___x_1119_, 0);
lean_inc(v_val_1120_);
v___x_1121_ = lean_array_push(v_b_1112_, v_val_1120_);
v___y_1114_ = v___x_1121_;
goto v___jp_1113_;
}
}
else
{
return v_b_1112_;
}
v___jp_1113_:
{
size_t v___x_1115_; size_t v___x_1116_; 
v___x_1115_ = ((size_t)1ULL);
v___x_1116_ = lean_usize_add(v_i_1110_, v___x_1115_);
v_i_1110_ = v___x_1116_;
v_b_1112_ = v___y_1114_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8___boxed(lean_object* v_as_1122_, lean_object* v_i_1123_, lean_object* v_stop_1124_, lean_object* v_b_1125_){
_start:
{
size_t v_i_boxed_1126_; size_t v_stop_boxed_1127_; lean_object* v_res_1128_; 
v_i_boxed_1126_ = lean_unbox_usize(v_i_1123_);
lean_dec(v_i_1123_);
v_stop_boxed_1127_ = lean_unbox_usize(v_stop_1124_);
lean_dec(v_stop_1124_);
v_res_1128_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8(v_as_1122_, v_i_boxed_1126_, v_stop_boxed_1127_, v_b_1125_);
lean_dec_ref(v_as_1122_);
return v_res_1128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6(lean_object* v_as_1131_, lean_object* v_start_1132_, lean_object* v_stop_1133_){
_start:
{
lean_object* v___x_1134_; uint8_t v___x_1135_; 
v___x_1134_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___closed__0));
v___x_1135_ = lean_nat_dec_lt(v_start_1132_, v_stop_1133_);
if (v___x_1135_ == 0)
{
return v___x_1134_;
}
else
{
lean_object* v___x_1136_; uint8_t v___x_1137_; 
v___x_1136_ = lean_array_get_size(v_as_1131_);
v___x_1137_ = lean_nat_dec_le(v_stop_1133_, v___x_1136_);
if (v___x_1137_ == 0)
{
uint8_t v___x_1138_; 
v___x_1138_ = lean_nat_dec_lt(v_start_1132_, v___x_1136_);
if (v___x_1138_ == 0)
{
return v___x_1134_;
}
else
{
size_t v___x_1139_; size_t v___x_1140_; lean_object* v___x_1141_; 
v___x_1139_ = lean_usize_of_nat(v_start_1132_);
v___x_1140_ = lean_usize_of_nat(v___x_1136_);
v___x_1141_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8(v_as_1131_, v___x_1139_, v___x_1140_, v___x_1134_);
return v___x_1141_;
}
}
else
{
size_t v___x_1142_; size_t v___x_1143_; lean_object* v___x_1144_; 
v___x_1142_ = lean_usize_of_nat(v_start_1132_);
v___x_1143_ = lean_usize_of_nat(v_stop_1133_);
v___x_1144_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6_spec__8(v_as_1131_, v___x_1142_, v___x_1143_, v___x_1134_);
return v___x_1144_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6___boxed(lean_object* v_as_1145_, lean_object* v_start_1146_, lean_object* v_stop_1147_){
_start:
{
lean_object* v_res_1148_; 
v_res_1148_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6(v_as_1145_, v_start_1146_, v_stop_1147_);
lean_dec(v_stop_1147_);
lean_dec(v_start_1146_);
lean_dec_ref(v_as_1145_);
return v_res_1148_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__0(void){
_start:
{
lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1149_ = lean_box(0);
v___x_1150_ = lean_unsigned_to_nat(16u);
v___x_1151_ = lean_mk_array(v___x_1150_, v___x_1149_);
return v___x_1151_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__1(void){
_start:
{
lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; 
v___x_1152_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__0, &lp_aesop_Aesop_Substitution_openRuleType___closed__0_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__0);
v___x_1153_ = lean_unsigned_to_nat(0u);
v___x_1154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1154_, 0, v___x_1153_);
lean_ctor_set(v___x_1154_, 1, v___x_1152_);
return v___x_1154_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__3(void){
_start:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; 
v___x_1157_ = ((lean_object*)(lp_aesop_Aesop_Substitution_openRuleType___closed__2));
v___x_1158_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__1, &lp_aesop_Aesop_Substitution_openRuleType___closed__1_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__1);
v___x_1159_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1158_);
lean_ctor_set(v___x_1159_, 1, v___x_1158_);
lean_ctor_set(v___x_1159_, 2, v___x_1157_);
return v___x_1159_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__5(void){
_start:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1161_ = ((lean_object*)(lp_aesop_Aesop_Substitution_openRuleType___closed__4));
v___x_1162_ = l_Lean_stringToMessageData(v___x_1161_);
return v___x_1162_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__7(void){
_start:
{
lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1164_ = ((lean_object*)(lp_aesop_Aesop_Substitution_openRuleType___closed__6));
v___x_1165_ = l_Lean_stringToMessageData(v___x_1164_);
return v___x_1165_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__9(void){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1167_ = ((lean_object*)(lp_aesop_Aesop_Substitution_openRuleType___closed__8));
v___x_1168_ = l_Lean_stringToMessageData(v___x_1167_);
return v___x_1168_;
}
}
static lean_object* _init_lp_aesop_Aesop_Substitution_openRuleType___closed__11(void){
_start:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1170_ = ((lean_object*)(lp_aesop_Aesop_Substitution_openRuleType___closed__10));
v___x_1171_ = l_Lean_stringToMessageData(v___x_1170_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_openRuleType(lean_object* v_e_1172_, lean_object* v_subst_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_){
_start:
{
lean_object* v___x_1179_; lean_object* v_a_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v_result_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1349_; 
lean_inc_ref(v_e_1172_);
v___x_1179_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(v_e_1172_, v_a_1175_);
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
lean_inc(v_a_1180_);
lean_dec_ref(v___x_1179_);
v___x_1181_ = lean_unsigned_to_nat(0u);
v___x_1182_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__3, &lp_aesop_Aesop_Substitution_openRuleType___closed__3_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__3);
v___x_1183_ = l_Lean_collectLevelMVars(v___x_1182_, v_a_1180_);
v_result_1184_ = lean_ctor_get(v___x_1183_, 2);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1183_);
if (v_isSharedCheck_1349_ == 0)
{
lean_object* v_unused_1350_; lean_object* v_unused_1351_; 
v_unused_1350_ = lean_ctor_get(v___x_1183_, 1);
lean_dec(v_unused_1350_);
v_unused_1351_ = lean_ctor_get(v___x_1183_, 0);
lean_dec(v_unused_1351_);
v___x_1186_ = v___x_1183_;
v_isShared_1187_ = v_isSharedCheck_1349_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_result_1184_);
lean_dec(v___x_1183_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1349_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___y_1189_; lean_object* v___y_1190_; lean_object* v___y_1191_; lean_object* v___y_1192_; lean_object* v___y_1193_; lean_object* v___y_1194_; lean_object* v_premises_1214_; lean_object* v_levels_1215_; lean_object* v___y_1217_; lean_object* v___y_1218_; lean_object* v___y_1219_; lean_object* v___y_1220_; lean_object* v___x_1303_; lean_object* v___x_1304_; uint8_t v___x_1305_; 
v_premises_1214_ = lean_ctor_get(v_subst_1173_, 0);
v_levels_1215_ = lean_ctor_get(v_subst_1173_, 1);
v___x_1303_ = lean_array_get_size(v_levels_1215_);
v___x_1304_ = lean_array_get_size(v_result_1184_);
v___x_1305_ = lean_nat_dec_eq(v___x_1303_, v___x_1304_);
if (v___x_1305_ == 0)
{
lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1346_; 
lean_inc_ref(v_levels_1215_);
lean_inc_ref(v_premises_1214_);
lean_del_object(v___x_1186_);
lean_dec_ref(v_result_1184_);
v_isSharedCheck_1346_ = !lean_is_exclusive(v_subst_1173_);
if (v_isSharedCheck_1346_ == 0)
{
lean_object* v_unused_1347_; lean_object* v_unused_1348_; 
v_unused_1347_ = lean_ctor_get(v_subst_1173_, 1);
lean_dec(v_unused_1347_);
v_unused_1348_ = lean_ctor_get(v_subst_1173_, 0);
lean_dec(v_unused_1348_);
v___x_1307_ = v_subst_1173_;
v_isShared_1308_ = v_isSharedCheck_1346_;
goto v_resetjp_1306_;
}
else
{
lean_dec(v_subst_1173_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1346_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1312_; 
v___x_1309_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__11, &lp_aesop_Aesop_Substitution_openRuleType___closed__11_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__11);
v___x_1310_ = l_Lean_indentExpr(v_e_1172_);
if (v_isShared_1308_ == 0)
{
lean_ctor_set_tag(v___x_1307_, 7);
lean_ctor_set(v___x_1307_, 1, v___x_1310_);
lean_ctor_set(v___x_1307_, 0, v___x_1309_);
v___x_1312_ = v___x_1307_;
goto v_reusejp_1311_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v___x_1309_);
lean_ctor_set(v_reuseFailAlloc_1345_, 1, v___x_1310_);
v___x_1312_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1311_;
}
v_reusejp_1311_:
{
lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; size_t v_sz_1317_; size_t v___x_1318_; lean_object* v___x_1319_; lean_object* v_ps_1320_; lean_object* v___x_1321_; size_t v_sz_1322_; lean_object* v___x_1323_; lean_object* v_ls_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v_a_1337_; lean_object* v___x_1339_; uint8_t v_isShared_1340_; uint8_t v_isSharedCheck_1344_; 
v___x_1313_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__9, &lp_aesop_Aesop_Substitution_openRuleType___closed__9_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__9);
v___x_1314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1314_, 0, v___x_1312_);
lean_ctor_set(v___x_1314_, 1, v___x_1313_);
v___x_1315_ = lean_array_get_size(v_premises_1214_);
v___x_1316_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6(v_premises_1214_, v___x_1181_, v___x_1315_);
lean_dec_ref(v_premises_1214_);
v_sz_1317_ = lean_array_size(v___x_1316_);
v___x_1318_ = ((size_t)0ULL);
v___x_1319_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(v_sz_1317_, v___x_1318_, v___x_1316_);
v_ps_1320_ = lean_array_to_list(v___x_1319_);
v___x_1321_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8(v_levels_1215_, v___x_1181_, v___x_1303_);
lean_dec_ref(v_levels_1215_);
v_sz_1322_ = lean_array_size(v___x_1321_);
v___x_1323_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(v_sz_1322_, v___x_1318_, v___x_1321_);
v_ls_1324_ = lean_array_to_list(v___x_1323_);
v___x_1325_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0));
v___x_1326_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3);
v___x_1327_ = l_Lean_MessageData_joinSep(v_ps_1320_, v___x_1326_);
v___x_1328_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6);
v___x_1329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1329_, 0, v___x_1327_);
lean_ctor_set(v___x_1329_, 1, v___x_1328_);
v___x_1330_ = l_Lean_MessageData_joinSep(v_ls_1324_, v___x_1326_);
v___x_1331_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1331_, 0, v___x_1329_);
lean_ctor_set(v___x_1331_, 1, v___x_1330_);
v___x_1332_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7));
v___x_1333_ = l_Lean_MessageData_bracket(v___x_1325_, v___x_1331_, v___x_1332_);
v___x_1334_ = l_Lean_indentD(v___x_1333_);
v___x_1335_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1314_);
lean_ctor_set(v___x_1335_, 1, v___x_1334_);
v___x_1336_ = lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(v___x_1335_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_);
v_a_1337_ = lean_ctor_get(v___x_1336_, 0);
v_isSharedCheck_1344_ = !lean_is_exclusive(v___x_1336_);
if (v_isSharedCheck_1344_ == 0)
{
v___x_1339_ = v___x_1336_;
v_isShared_1340_ = v_isSharedCheck_1344_;
goto v_resetjp_1338_;
}
else
{
lean_inc(v_a_1337_);
lean_dec(v___x_1336_);
v___x_1339_ = lean_box(0);
v_isShared_1340_ = v_isSharedCheck_1344_;
goto v_resetjp_1338_;
}
v_resetjp_1338_:
{
lean_object* v___x_1342_; 
if (v_isShared_1340_ == 0)
{
v___x_1342_ = v___x_1339_;
goto v_reusejp_1341_;
}
else
{
lean_object* v_reuseFailAlloc_1343_; 
v_reuseFailAlloc_1343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1343_, 0, v_a_1337_);
v___x_1342_ = v_reuseFailAlloc_1343_;
goto v_reusejp_1341_;
}
v_reusejp_1341_:
{
return v___x_1342_;
}
}
}
}
}
else
{
v___y_1217_ = v_a_1174_;
v___y_1218_ = v_a_1175_;
v___y_1219_ = v_a_1176_;
v___y_1220_ = v_a_1177_;
goto v___jp_1216_;
}
v___jp_1188_:
{
lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1198_; 
v___x_1195_ = lean_array_get_size(v_result_1184_);
v___x_1196_ = lean_unsigned_to_nat(1u);
if (v_isShared_1187_ == 0)
{
lean_ctor_set(v___x_1186_, 2, v___x_1196_);
lean_ctor_set(v___x_1186_, 1, v___x_1195_);
lean_ctor_set(v___x_1186_, 0, v___x_1181_);
v___x_1198_ = v___x_1186_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v___x_1181_);
lean_ctor_set(v_reuseFailAlloc_1213_, 1, v___x_1195_);
lean_ctor_set(v_reuseFailAlloc_1213_, 2, v___x_1196_);
v___x_1198_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1211_; 
v___x_1199_ = lean_box(0);
v___x_1200_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(v_subst_1173_, v_result_1184_, v___x_1198_, v___x_1199_, v___x_1181_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
lean_dec_ref(v___x_1198_);
lean_dec_ref(v_result_1184_);
lean_dec_ref(v___x_1200_);
v___x_1201_ = lean_array_get_size(v___y_1189_);
v___x_1202_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1181_);
lean_ctor_set(v___x_1202_, 1, v___x_1201_);
lean_ctor_set(v___x_1202_, 2, v___x_1196_);
v___x_1203_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg(v_subst_1173_, v___y_1189_, v___x_1202_, v___x_1199_, v___x_1181_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
lean_dec_ref_known(v___x_1202_, 3);
lean_dec_ref(v_subst_1173_);
v_isSharedCheck_1211_ = !lean_is_exclusive(v___x_1203_);
if (v_isSharedCheck_1211_ == 0)
{
lean_object* v_unused_1212_; 
v_unused_1212_ = lean_ctor_get(v___x_1203_, 0);
lean_dec(v_unused_1212_);
v___x_1205_ = v___x_1203_;
v_isShared_1206_ = v_isSharedCheck_1211_;
goto v_resetjp_1204_;
}
else
{
lean_dec(v___x_1203_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1211_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
lean_object* v___x_1207_; lean_object* v___x_1209_; 
v___x_1207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1207_, 0, v___y_1189_);
lean_ctor_set(v___x_1207_, 1, v___y_1190_);
if (v_isShared_1206_ == 0)
{
lean_ctor_set(v___x_1205_, 0, v___x_1207_);
v___x_1209_ = v___x_1205_;
goto v_reusejp_1208_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v___x_1207_);
v___x_1209_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1208_;
}
v_reusejp_1208_:
{
return v___x_1209_;
}
}
}
}
v___jp_1216_:
{
lean_object* v___x_1221_; 
lean_inc(v___y_1220_);
lean_inc_ref(v___y_1219_);
lean_inc(v___y_1218_);
lean_inc_ref(v___y_1217_);
lean_inc_ref(v_e_1172_);
v___x_1221_ = lean_infer_type(v_e_1172_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
if (lean_obj_tag(v___x_1221_) == 0)
{
lean_object* v_a_1222_; lean_object* v___x_1223_; uint8_t v___x_1224_; lean_object* v___x_1225_; 
v_a_1222_ = lean_ctor_get(v___x_1221_, 0);
lean_inc_n(v_a_1222_, 2);
lean_dec_ref_known(v___x_1221_, 1);
v___x_1223_ = lean_box(0);
v___x_1224_ = 0;
v___x_1225_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_1222_, v___x_1223_, v___x_1224_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
if (lean_obj_tag(v___x_1225_) == 0)
{
lean_object* v_a_1226_; lean_object* v_fst_1227_; lean_object* v_snd_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1286_; 
v_a_1226_ = lean_ctor_get(v___x_1225_, 0);
lean_inc(v_a_1226_);
lean_dec_ref_known(v___x_1225_, 1);
v_fst_1227_ = lean_ctor_get(v_a_1226_, 0);
v_snd_1228_ = lean_ctor_get(v_a_1226_, 1);
v_isSharedCheck_1286_ = !lean_is_exclusive(v_a_1226_);
if (v_isSharedCheck_1286_ == 0)
{
v___x_1230_ = v_a_1226_;
v_isShared_1231_ = v_isSharedCheck_1286_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_snd_1228_);
lean_inc(v_fst_1227_);
lean_dec(v_a_1226_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1286_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
size_t v_sz_1232_; size_t v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; uint8_t v___x_1237_; 
v_sz_1232_ = lean_array_size(v_fst_1227_);
v___x_1233_ = ((size_t)0ULL);
lean_inc(v_fst_1227_);
v___x_1234_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1(v_sz_1232_, v___x_1233_, v_fst_1227_);
v___x_1235_ = lean_array_get_size(v_premises_1214_);
v___x_1236_ = lean_array_get_size(v_fst_1227_);
lean_dec(v_fst_1227_);
v___x_1237_ = lean_nat_dec_eq(v___x_1235_, v___x_1236_);
if (v___x_1237_ == 0)
{
lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1283_; 
lean_inc_ref(v_levels_1215_);
lean_inc_ref(v_premises_1214_);
lean_dec_ref(v___x_1234_);
lean_dec(v_snd_1228_);
lean_del_object(v___x_1186_);
lean_dec_ref(v_result_1184_);
v_isSharedCheck_1283_ = !lean_is_exclusive(v_subst_1173_);
if (v_isSharedCheck_1283_ == 0)
{
lean_object* v_unused_1284_; lean_object* v_unused_1285_; 
v_unused_1284_ = lean_ctor_get(v_subst_1173_, 1);
lean_dec(v_unused_1284_);
v_unused_1285_ = lean_ctor_get(v_subst_1173_, 0);
lean_dec(v_unused_1285_);
v___x_1239_ = v_subst_1173_;
v_isShared_1240_ = v_isSharedCheck_1283_;
goto v_resetjp_1238_;
}
else
{
lean_dec(v_subst_1173_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1283_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1244_; 
v___x_1241_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__5, &lp_aesop_Aesop_Substitution_openRuleType___closed__5_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__5);
v___x_1242_ = l_Lean_indentExpr(v_e_1172_);
if (v_isShared_1231_ == 0)
{
lean_ctor_set_tag(v___x_1230_, 7);
lean_ctor_set(v___x_1230_, 1, v___x_1242_);
lean_ctor_set(v___x_1230_, 0, v___x_1241_);
v___x_1244_ = v___x_1230_;
goto v_reusejp_1243_;
}
else
{
lean_object* v_reuseFailAlloc_1282_; 
v_reuseFailAlloc_1282_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1282_, 0, v___x_1241_);
lean_ctor_set(v_reuseFailAlloc_1282_, 1, v___x_1242_);
v___x_1244_ = v_reuseFailAlloc_1282_;
goto v_reusejp_1243_;
}
v_reusejp_1243_:
{
lean_object* v___x_1245_; lean_object* v___x_1247_; 
v___x_1245_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__7, &lp_aesop_Aesop_Substitution_openRuleType___closed__7_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__7);
if (v_isShared_1240_ == 0)
{
lean_ctor_set_tag(v___x_1239_, 7);
lean_ctor_set(v___x_1239_, 1, v___x_1245_);
lean_ctor_set(v___x_1239_, 0, v___x_1244_);
v___x_1247_ = v___x_1239_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1281_; 
v_reuseFailAlloc_1281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1281_, 0, v___x_1244_);
lean_ctor_set(v_reuseFailAlloc_1281_, 1, v___x_1245_);
v___x_1247_ = v_reuseFailAlloc_1281_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; size_t v_sz_1253_; lean_object* v___x_1254_; lean_object* v_ps_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; size_t v_sz_1258_; lean_object* v___x_1259_; lean_object* v_ls_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v_a_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1280_; 
v___x_1248_ = l_Lean_indentExpr(v_a_1222_);
v___x_1249_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1247_);
lean_ctor_set(v___x_1249_, 1, v___x_1248_);
v___x_1250_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__9, &lp_aesop_Aesop_Substitution_openRuleType___closed__9_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__9);
v___x_1251_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1251_, 0, v___x_1249_);
lean_ctor_set(v___x_1251_, 1, v___x_1250_);
v___x_1252_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__6(v_premises_1214_, v___x_1181_, v___x_1235_);
lean_dec_ref(v_premises_1214_);
v_sz_1253_ = lean_array_size(v___x_1252_);
v___x_1254_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(v_sz_1253_, v___x_1233_, v___x_1252_);
v_ps_1255_ = lean_array_to_list(v___x_1254_);
v___x_1256_ = lean_array_get_size(v_levels_1215_);
v___x_1257_ = lp_aesop_Array_filterMapM___at___00Aesop_Substitution_openRuleType_spec__8(v_levels_1215_, v___x_1181_, v___x_1256_);
lean_dec_ref(v_levels_1215_);
v_sz_1258_ = lean_array_size(v___x_1257_);
v___x_1259_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(v_sz_1258_, v___x_1233_, v___x_1257_);
v_ls_1260_ = lean_array_to_list(v___x_1259_);
v___x_1261_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__0));
v___x_1262_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__3);
v___x_1263_ = l_Lean_MessageData_joinSep(v_ps_1255_, v___x_1262_);
v___x_1264_ = lean_obj_once(&lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6, &lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6_once, _init_lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__6);
v___x_1265_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1265_, 0, v___x_1263_);
lean_ctor_set(v___x_1265_, 1, v___x_1264_);
v___x_1266_ = l_Lean_MessageData_joinSep(v_ls_1260_, v___x_1262_);
v___x_1267_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1265_);
lean_ctor_set(v___x_1267_, 1, v___x_1266_);
v___x_1268_ = ((lean_object*)(lp_aesop_Aesop_Substitution_instToMessageData___lam__4___closed__7));
v___x_1269_ = l_Lean_MessageData_bracket(v___x_1261_, v___x_1267_, v___x_1268_);
v___x_1270_ = l_Lean_indentD(v___x_1269_);
v___x_1271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1251_);
lean_ctor_set(v___x_1271_, 1, v___x_1270_);
v___x_1272_ = lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(v___x_1271_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
v_a_1273_ = lean_ctor_get(v___x_1272_, 0);
v_isSharedCheck_1280_ = !lean_is_exclusive(v___x_1272_);
if (v_isSharedCheck_1280_ == 0)
{
v___x_1275_ = v___x_1272_;
v_isShared_1276_ = v_isSharedCheck_1280_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_a_1273_);
lean_dec(v___x_1272_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1280_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
lean_object* v___x_1278_; 
if (v_isShared_1276_ == 0)
{
v___x_1278_ = v___x_1275_;
goto v_reusejp_1277_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v_a_1273_);
v___x_1278_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1277_;
}
v_reusejp_1277_:
{
return v___x_1278_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_1230_);
lean_dec(v_a_1222_);
lean_dec_ref(v_e_1172_);
v___y_1189_ = v___x_1234_;
v___y_1190_ = v_snd_1228_;
v___y_1191_ = v___y_1217_;
v___y_1192_ = v___y_1218_;
v___y_1193_ = v___y_1219_;
v___y_1194_ = v___y_1220_;
goto v___jp_1188_;
}
}
}
else
{
lean_object* v_a_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1294_; 
lean_dec(v_a_1222_);
lean_del_object(v___x_1186_);
lean_dec_ref(v_result_1184_);
lean_dec_ref(v_subst_1173_);
lean_dec_ref(v_e_1172_);
v_a_1287_ = lean_ctor_get(v___x_1225_, 0);
v_isSharedCheck_1294_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1294_ == 0)
{
v___x_1289_ = v___x_1225_;
v_isShared_1290_ = v_isSharedCheck_1294_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_a_1287_);
lean_dec(v___x_1225_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1294_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
lean_object* v___x_1292_; 
if (v_isShared_1290_ == 0)
{
v___x_1292_ = v___x_1289_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v_a_1287_);
v___x_1292_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
return v___x_1292_;
}
}
}
}
else
{
lean_object* v_a_1295_; lean_object* v___x_1297_; uint8_t v_isShared_1298_; uint8_t v_isSharedCheck_1302_; 
lean_del_object(v___x_1186_);
lean_dec_ref(v_result_1184_);
lean_dec_ref(v_subst_1173_);
lean_dec_ref(v_e_1172_);
v_a_1295_ = lean_ctor_get(v___x_1221_, 0);
v_isSharedCheck_1302_ = !lean_is_exclusive(v___x_1221_);
if (v_isSharedCheck_1302_ == 0)
{
v___x_1297_ = v___x_1221_;
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
else
{
lean_inc(v_a_1295_);
lean_dec(v___x_1221_);
v___x_1297_ = lean_box(0);
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
v_resetjp_1296_:
{
lean_object* v___x_1300_; 
if (v_isShared_1298_ == 0)
{
v___x_1300_ = v___x_1297_;
goto v_reusejp_1299_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v_a_1295_);
v___x_1300_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1299_;
}
v_reusejp_1299_:
{
return v___x_1300_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_openRuleType___boxed(lean_object* v_e_1352_, lean_object* v_subst_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_, lean_object* v_a_1358_){
_start:
{
lean_object* v_res_1359_; 
v_res_1359_ = lp_aesop_Aesop_Substitution_openRuleType(v_e_1352_, v_subst_1353_, v_a_1354_, v_a_1355_, v_a_1356_, v_a_1357_);
lean_dec(v_a_1357_);
lean_dec_ref(v_a_1356_);
lean_dec(v_a_1355_);
lean_dec_ref(v_a_1354_);
return v_res_1359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2(lean_object* v_mvarId_1360_, lean_object* v_val_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_){
_start:
{
lean_object* v___x_1367_; 
v___x_1367_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___redArg(v_mvarId_1360_, v_val_1361_, v___y_1363_);
return v___x_1367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2___boxed(lean_object* v_mvarId_1368_, lean_object* v_val_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2(v_mvarId_1368_, v_val_1369_, v___y_1370_, v___y_1371_, v___y_1372_, v___y_1373_);
lean_dec(v___y_1373_);
lean_dec_ref(v___y_1372_);
lean_dec(v___y_1371_);
lean_dec_ref(v___y_1370_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3(lean_object* v_mvarId_1376_, lean_object* v_val_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_){
_start:
{
lean_object* v___x_1383_; 
v___x_1383_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___redArg(v_mvarId_1376_, v_val_1377_, v___y_1379_);
return v___x_1383_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3___boxed(lean_object* v_mvarId_1384_, lean_object* v_val_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_){
_start:
{
lean_object* v_res_1391_; 
v_res_1391_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3(v_mvarId_1384_, v_val_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
lean_dec(v___y_1389_);
lean_dec_ref(v___y_1388_);
lean_dec(v___y_1387_);
lean_dec_ref(v___y_1386_);
return v_res_1391_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4(lean_object* v_subst_1392_, lean_object* v___x_1393_, lean_object* v_range_1394_, lean_object* v_b_1395_, lean_object* v_i_1396_, lean_object* v_hs_1397_, lean_object* v_hl_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
lean_object* v___x_1404_; 
v___x_1404_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(v_subst_1392_, v___x_1393_, v_range_1394_, v_b_1395_, v_i_1396_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___boxed(lean_object* v_subst_1405_, lean_object* v___x_1406_, lean_object* v_range_1407_, lean_object* v_b_1408_, lean_object* v_i_1409_, lean_object* v_hs_1410_, lean_object* v_hl_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_){
_start:
{
lean_object* v_res_1417_; 
v_res_1417_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4(v_subst_1405_, v___x_1406_, v_range_1407_, v_b_1408_, v_i_1409_, v_hs_1410_, v_hl_1411_, v___y_1412_, v___y_1413_, v___y_1414_, v___y_1415_);
lean_dec(v___y_1415_);
lean_dec_ref(v___y_1414_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1412_);
lean_dec_ref(v_range_1407_);
lean_dec_ref(v___x_1406_);
lean_dec_ref(v_subst_1405_);
return v_res_1417_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5(lean_object* v_subst_1418_, lean_object* v___x_1419_, lean_object* v_range_1420_, lean_object* v_b_1421_, lean_object* v_i_1422_, lean_object* v_hs_1423_, lean_object* v_hl_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_){
_start:
{
lean_object* v___x_1430_; 
v___x_1430_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___redArg(v_subst_1418_, v___x_1419_, v_range_1420_, v_b_1421_, v_i_1422_, v___y_1425_, v___y_1426_, v___y_1427_, v___y_1428_);
return v___x_1430_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5___boxed(lean_object* v_subst_1431_, lean_object* v___x_1432_, lean_object* v_range_1433_, lean_object* v_b_1434_, lean_object* v_i_1435_, lean_object* v_hs_1436_, lean_object* v_hl_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_){
_start:
{
lean_object* v_res_1443_; 
v_res_1443_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__5(v_subst_1431_, v___x_1432_, v_range_1433_, v_b_1434_, v_i_1435_, v_hs_1436_, v_hl_1437_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_);
lean_dec(v___y_1441_);
lean_dec_ref(v___y_1440_);
lean_dec(v___y_1439_);
lean_dec_ref(v___y_1438_);
lean_dec_ref(v_range_1433_);
lean_dec_ref(v___x_1432_);
lean_dec_ref(v_subst_1431_);
return v_res_1443_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7(lean_object* v_as_1444_, size_t v_sz_1445_, size_t v_i_1446_, lean_object* v_bs_1447_){
_start:
{
lean_object* v___x_1448_; 
v___x_1448_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___redArg(v_sz_1445_, v_i_1446_, v_bs_1447_);
return v___x_1448_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7___boxed(lean_object* v_as_1449_, lean_object* v_sz_1450_, lean_object* v_i_1451_, lean_object* v_bs_1452_){
_start:
{
size_t v_sz_boxed_1453_; size_t v_i_boxed_1454_; lean_object* v_res_1455_; 
v_sz_boxed_1453_ = lean_unbox_usize(v_sz_1450_);
lean_dec(v_sz_1450_);
v_i_boxed_1454_ = lean_unbox_usize(v_i_1451_);
lean_dec(v_i_1451_);
v_res_1455_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__7(v_as_1449_, v_sz_boxed_1453_, v_i_boxed_1454_, v_bs_1452_);
lean_dec_ref(v_as_1449_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9(lean_object* v_as_1456_, size_t v_sz_1457_, size_t v_i_1458_, lean_object* v_bs_1459_){
_start:
{
lean_object* v___x_1460_; 
v___x_1460_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___redArg(v_sz_1457_, v_i_1458_, v_bs_1459_);
return v___x_1460_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9___boxed(lean_object* v_as_1461_, lean_object* v_sz_1462_, lean_object* v_i_1463_, lean_object* v_bs_1464_){
_start:
{
size_t v_sz_boxed_1465_; size_t v_i_boxed_1466_; lean_object* v_res_1467_; 
v_sz_boxed_1465_ = lean_unbox_usize(v_sz_1462_);
lean_dec(v_sz_1462_);
v_i_boxed_1466_ = lean_unbox_usize(v_i_1463_);
lean_dec(v_i_1463_);
v_res_1467_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__9(v_as_1461_, v_sz_boxed_1465_, v_i_boxed_1466_, v_bs_1464_);
lean_dec_ref(v_as_1461_);
return v_res_1467_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10(lean_object* v_00_u03b1_1468_, lean_object* v_msg_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
lean_object* v___x_1475_; 
v___x_1475_ = lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___redArg(v_msg_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
return v___x_1475_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10___boxed(lean_object* v_00_u03b1_1476_, lean_object* v_msg_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
lean_object* v_res_1483_; 
v_res_1483_ = lp_aesop_Lean_throwError___at___00Aesop_Substitution_openRuleType_spec__10(v_00_u03b1_1476_, v_msg_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
return v_res_1483_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2(lean_object* v_00_u03b2_1484_, lean_object* v_x_1485_, lean_object* v_x_1486_, lean_object* v_x_1487_){
_start:
{
lean_object* v___x_1488_; 
v___x_1488_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2___redArg(v_x_1485_, v_x_1486_, v_x_1487_);
return v___x_1488_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4(lean_object* v_00_u03b2_1489_, lean_object* v_x_1490_, lean_object* v_x_1491_, lean_object* v_x_1492_){
_start:
{
lean_object* v___x_1493_; 
v___x_1493_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4___redArg(v_x_1490_, v_x_1491_, v_x_1492_);
return v___x_1493_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3(lean_object* v_00_u03b2_1494_, lean_object* v_x_1495_, size_t v_x_1496_, size_t v_x_1497_, lean_object* v_x_1498_, lean_object* v_x_1499_){
_start:
{
lean_object* v___x_1500_; 
v___x_1500_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___redArg(v_x_1495_, v_x_1496_, v_x_1497_, v_x_1498_, v_x_1499_);
return v___x_1500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3___boxed(lean_object* v_00_u03b2_1501_, lean_object* v_x_1502_, lean_object* v_x_1503_, lean_object* v_x_1504_, lean_object* v_x_1505_, lean_object* v_x_1506_){
_start:
{
size_t v_x_9164__boxed_1507_; size_t v_x_9165__boxed_1508_; lean_object* v_res_1509_; 
v_x_9164__boxed_1507_ = lean_unbox_usize(v_x_1503_);
lean_dec(v_x_1503_);
v_x_9165__boxed_1508_ = lean_unbox_usize(v_x_1504_);
lean_dec(v_x_1504_);
v_res_1509_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3(v_00_u03b2_1501_, v_x_1502_, v_x_9164__boxed_1507_, v_x_9165__boxed_1508_, v_x_1505_, v_x_1506_);
return v_res_1509_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6(lean_object* v_00_u03b2_1510_, lean_object* v_x_1511_, size_t v_x_1512_, size_t v_x_1513_, lean_object* v_x_1514_, lean_object* v_x_1515_){
_start:
{
lean_object* v___x_1516_; 
v___x_1516_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___redArg(v_x_1511_, v_x_1512_, v_x_1513_, v_x_1514_, v_x_1515_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6___boxed(lean_object* v_00_u03b2_1517_, lean_object* v_x_1518_, lean_object* v_x_1519_, lean_object* v_x_1520_, lean_object* v_x_1521_, lean_object* v_x_1522_){
_start:
{
size_t v_x_9181__boxed_1523_; size_t v_x_9182__boxed_1524_; lean_object* v_res_1525_; 
v_x_9181__boxed_1523_ = lean_unbox_usize(v_x_1519_);
lean_dec(v_x_1519_);
v_x_9182__boxed_1524_ = lean_unbox_usize(v_x_1520_);
lean_dec(v_x_1520_);
v_res_1525_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6(v_00_u03b2_1517_, v_x_1518_, v_x_9181__boxed_1523_, v_x_9182__boxed_1524_, v_x_1521_, v_x_1522_);
return v_res_1525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13(lean_object* v_00_u03b2_1526_, lean_object* v_n_1527_, lean_object* v_k_1528_, lean_object* v_v_1529_){
_start:
{
lean_object* v___x_1530_; 
v___x_1530_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13___redArg(v_n_1527_, v_k_1528_, v_v_1529_);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14(lean_object* v_00_u03b2_1531_, size_t v_depth_1532_, lean_object* v_keys_1533_, lean_object* v_vals_1534_, lean_object* v_heq_1535_, lean_object* v_i_1536_, lean_object* v_entries_1537_){
_start:
{
lean_object* v___x_1538_; 
v___x_1538_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___redArg(v_depth_1532_, v_keys_1533_, v_vals_1534_, v_i_1536_, v_entries_1537_);
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14___boxed(lean_object* v_00_u03b2_1539_, lean_object* v_depth_1540_, lean_object* v_keys_1541_, lean_object* v_vals_1542_, lean_object* v_heq_1543_, lean_object* v_i_1544_, lean_object* v_entries_1545_){
_start:
{
size_t v_depth_boxed_1546_; lean_object* v_res_1547_; 
v_depth_boxed_1546_ = lean_unbox_usize(v_depth_1540_);
lean_dec(v_depth_1540_);
v_res_1547_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__14(v_00_u03b2_1539_, v_depth_boxed_1546_, v_keys_1541_, v_vals_1542_, v_heq_1543_, v_i_1544_, v_entries_1545_);
lean_dec_ref(v_vals_1542_);
lean_dec_ref(v_keys_1541_);
return v_res_1547_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17(lean_object* v_00_u03b2_1548_, lean_object* v_n_1549_, lean_object* v_k_1550_, lean_object* v_v_1551_){
_start:
{
lean_object* v___x_1552_; 
v___x_1552_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17___redArg(v_n_1549_, v_k_1550_, v_v_1551_);
return v___x_1552_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18(lean_object* v_00_u03b2_1553_, size_t v_depth_1554_, lean_object* v_keys_1555_, lean_object* v_vals_1556_, lean_object* v_heq_1557_, lean_object* v_i_1558_, lean_object* v_entries_1559_){
_start:
{
lean_object* v___x_1560_; 
v___x_1560_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___redArg(v_depth_1554_, v_keys_1555_, v_vals_1556_, v_i_1558_, v_entries_1559_);
return v___x_1560_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18___boxed(lean_object* v_00_u03b2_1561_, lean_object* v_depth_1562_, lean_object* v_keys_1563_, lean_object* v_vals_1564_, lean_object* v_heq_1565_, lean_object* v_i_1566_, lean_object* v_entries_1567_){
_start:
{
size_t v_depth_boxed_1568_; lean_object* v_res_1569_; 
v_depth_boxed_1568_ = lean_unbox_usize(v_depth_1562_);
lean_dec(v_depth_1562_);
v_res_1569_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__18(v_00_u03b2_1561_, v_depth_boxed_1568_, v_keys_1563_, v_vals_1564_, v_heq_1565_, v_i_1566_, v_entries_1567_);
lean_dec_ref(v_vals_1564_);
lean_dec_ref(v_keys_1563_);
return v_res_1569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17(lean_object* v_00_u03b2_1570_, lean_object* v_x_1571_, lean_object* v_x_1572_, lean_object* v_x_1573_, lean_object* v_x_1574_){
_start:
{
lean_object* v___x_1575_; 
v___x_1575_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_Substitution_openRuleType_spec__2_spec__2_spec__3_spec__13_spec__17___redArg(v_x_1571_, v_x_1572_, v_x_1573_, v_x_1574_);
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21(lean_object* v_00_u03b2_1576_, lean_object* v_x_1577_, lean_object* v_x_1578_, lean_object* v_x_1579_, lean_object* v_x_1580_){
_start:
{
lean_object* v___x_1581_; 
v___x_1581_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_Substitution_openRuleType_spec__3_spec__4_spec__6_spec__17_spec__21___redArg(v_x_1577_, v_x_1578_, v_x_1579_, v_x_1580_);
return v___x_1581_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0(lean_object* v_k_1582_, lean_object* v_b_1583_, lean_object* v_c_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_){
_start:
{
lean_object* v___x_1590_; 
lean_inc(v___y_1588_);
lean_inc_ref(v___y_1587_);
lean_inc(v___y_1586_);
lean_inc_ref(v___y_1585_);
v___x_1590_ = lean_apply_7(v_k_1582_, v_b_1583_, v_c_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_, lean_box(0));
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0___boxed(lean_object* v_k_1591_, lean_object* v_b_1592_, lean_object* v_c_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_){
_start:
{
lean_object* v_res_1599_; 
v_res_1599_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0(v_k_1591_, v_b_1592_, v_c_1593_, v___y_1594_, v___y_1595_, v___y_1596_, v___y_1597_);
lean_dec(v___y_1597_);
lean_dec_ref(v___y_1596_);
lean_dec(v___y_1595_);
lean_dec_ref(v___y_1594_);
return v_res_1599_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg(lean_object* v_type_1600_, lean_object* v_k_1601_, uint8_t v_cleanupAnnotations_1602_, uint8_t v_whnfType_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_){
_start:
{
lean_object* v___f_1609_; lean_object* v___x_1610_; 
v___f_1609_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1609_, 0, v_k_1601_);
v___x_1610_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_1600_, v___f_1609_, v_cleanupAnnotations_1602_, v_whnfType_1603_, v___y_1604_, v___y_1605_, v___y_1606_, v___y_1607_);
if (lean_obj_tag(v___x_1610_) == 0)
{
lean_object* v_a_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1618_; 
v_a_1611_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1613_ = v___x_1610_;
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_a_1611_);
lean_dec(v___x_1610_);
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
v_reuseFailAlloc_1617_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
v_a_1619_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1610_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1610_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg___boxed(lean_object* v_type_1627_, lean_object* v_k_1628_, lean_object* v_cleanupAnnotations_1629_, lean_object* v_whnfType_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1636_; uint8_t v_whnfType_boxed_1637_; lean_object* v_res_1638_; 
v_cleanupAnnotations_boxed_1636_ = lean_unbox(v_cleanupAnnotations_1629_);
v_whnfType_boxed_1637_ = lean_unbox(v_whnfType_1630_);
v_res_1638_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg(v_type_1627_, v_k_1628_, v_cleanupAnnotations_boxed_1636_, v_whnfType_boxed_1637_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_);
lean_dec(v___y_1634_);
lean_dec_ref(v___y_1633_);
lean_dec(v___y_1632_);
lean_dec_ref(v___y_1631_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1(lean_object* v_00_u03b1_1639_, lean_object* v_type_1640_, lean_object* v_k_1641_, uint8_t v_cleanupAnnotations_1642_, uint8_t v_whnfType_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_){
_start:
{
lean_object* v___x_1649_; 
v___x_1649_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg(v_type_1640_, v_k_1641_, v_cleanupAnnotations_1642_, v_whnfType_1643_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
return v___x_1649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___boxed(lean_object* v_00_u03b1_1650_, lean_object* v_type_1651_, lean_object* v_k_1652_, lean_object* v_cleanupAnnotations_1653_, lean_object* v_whnfType_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1660_; uint8_t v_whnfType_boxed_1661_; lean_object* v_res_1662_; 
v_cleanupAnnotations_boxed_1660_ = lean_unbox(v_cleanupAnnotations_1653_);
v_whnfType_boxed_1661_ = lean_unbox(v_whnfType_1654_);
v_res_1662_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1(v_00_u03b1_1650_, v_type_1651_, v_k_1652_, v_cleanupAnnotations_boxed_1660_, v_whnfType_boxed_1661_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec(v___y_1656_);
lean_dec_ref(v___y_1655_);
return v_res_1662_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg(lean_object* v_k_1663_, uint8_t v_allowLevelAssignments_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_){
_start:
{
lean_object* v___x_1670_; 
v___x_1670_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1664_, v_k_1663_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1678_; 
v_a_1671_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___x_1670_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1670_);
v___x_1673_ = lean_box(0);
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
v_resetjp_1672_:
{
lean_object* v___x_1676_; 
if (v_isShared_1674_ == 0)
{
v___x_1676_ = v___x_1673_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v_a_1671_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
else
{
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1686_; 
v_a_1679_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1686_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1681_ = v___x_1670_;
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1670_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
lean_object* v___x_1684_; 
if (v_isShared_1682_ == 0)
{
v___x_1684_ = v___x_1681_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_a_1679_);
v___x_1684_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
return v___x_1684_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg___boxed(lean_object* v_k_1687_, lean_object* v_allowLevelAssignments_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1694_; lean_object* v_res_1695_; 
v_allowLevelAssignments_boxed_1694_ = lean_unbox(v_allowLevelAssignments_1688_);
v_res_1695_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg(v_k_1687_, v_allowLevelAssignments_boxed_1694_, v___y_1689_, v___y_1690_, v___y_1691_, v___y_1692_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
lean_dec(v___y_1690_);
lean_dec_ref(v___y_1689_);
return v_res_1695_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2(lean_object* v_00_u03b1_1696_, lean_object* v_k_1697_, uint8_t v_allowLevelAssignments_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_){
_start:
{
lean_object* v___x_1704_; 
v___x_1704_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg(v_k_1697_, v_allowLevelAssignments_1698_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___boxed(lean_object* v_00_u03b1_1705_, lean_object* v_k_1706_, lean_object* v_allowLevelAssignments_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1713_; lean_object* v_res_1714_; 
v_allowLevelAssignments_boxed_1713_ = lean_unbox(v_allowLevelAssignments_1707_);
v_res_1714_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2(v_00_u03b1_1705_, v_k_1706_, v_allowLevelAssignments_boxed_1713_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_);
lean_dec(v___y_1711_);
lean_dec_ref(v___y_1710_);
lean_dec(v___y_1709_);
lean_dec_ref(v___y_1708_);
return v_res_1714_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg(lean_object* v_subst_1715_, lean_object* v_fvarIds_1716_, lean_object* v_range_1717_, lean_object* v_b_1718_, lean_object* v_i_1719_){
_start:
{
lean_object* v_stop_1721_; lean_object* v_step_1722_; lean_object* v_a_1724_; uint8_t v___x_1727_; 
v_stop_1721_ = lean_ctor_get(v_range_1717_, 1);
v_step_1722_ = lean_ctor_get(v_range_1717_, 2);
v___x_1727_ = lean_nat_dec_lt(v_i_1719_, v_stop_1721_);
if (v___x_1727_ == 0)
{
lean_object* v___x_1728_; 
lean_dec(v_i_1719_);
v___x_1728_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1728_, 0, v_b_1718_);
return v___x_1728_;
}
else
{
lean_object* v_fst_1729_; lean_object* v_snd_1730_; lean_object* v___x_1732_; uint8_t v_isShared_1733_; uint8_t v_isSharedCheck_1746_; 
v_fst_1729_ = lean_ctor_get(v_b_1718_, 0);
v_snd_1730_ = lean_ctor_get(v_b_1718_, 1);
v_isSharedCheck_1746_ = !lean_is_exclusive(v_b_1718_);
if (v_isSharedCheck_1746_ == 0)
{
v___x_1732_ = v_b_1718_;
v_isShared_1733_ = v_isSharedCheck_1746_;
goto v_resetjp_1731_;
}
else
{
lean_inc(v_snd_1730_);
lean_inc(v_fst_1729_);
lean_dec(v_b_1718_);
v___x_1732_ = lean_box(0);
v_isShared_1733_ = v_isSharedCheck_1746_;
goto v_resetjp_1731_;
}
v_resetjp_1731_:
{
lean_object* v___x_1734_; 
v___x_1734_ = lp_aesop_Aesop_Substitution_find_x3f(v_i_1719_, v_subst_1715_);
if (lean_obj_tag(v___x_1734_) == 1)
{
lean_object* v___x_1735_; lean_object* v___x_1737_; 
v___x_1735_ = lean_array_push(v_fst_1729_, v___x_1734_);
if (v_isShared_1733_ == 0)
{
lean_ctor_set(v___x_1732_, 0, v___x_1735_);
v___x_1737_ = v___x_1732_;
goto v_reusejp_1736_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v___x_1735_);
lean_ctor_set(v_reuseFailAlloc_1738_, 1, v_snd_1730_);
v___x_1737_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1736_;
}
v_reusejp_1736_:
{
v_a_1724_ = v___x_1737_;
goto v___jp_1723_;
}
}
else
{
lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1744_; 
lean_dec(v___x_1734_);
v___x_1739_ = lean_array_fget_borrowed(v_fvarIds_1716_, v_i_1719_);
lean_inc_n(v___x_1739_, 2);
v___x_1740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1740_, 0, v___x_1739_);
v___x_1741_ = lean_array_push(v_fst_1729_, v___x_1740_);
v___x_1742_ = lean_array_push(v_snd_1730_, v___x_1739_);
if (v_isShared_1733_ == 0)
{
lean_ctor_set(v___x_1732_, 1, v___x_1742_);
lean_ctor_set(v___x_1732_, 0, v___x_1741_);
v___x_1744_ = v___x_1732_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1741_);
lean_ctor_set(v_reuseFailAlloc_1745_, 1, v___x_1742_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
v_a_1724_ = v___x_1744_;
goto v___jp_1723_;
}
}
}
}
v___jp_1723_:
{
lean_object* v___x_1725_; 
v___x_1725_ = lean_nat_add(v_i_1719_, v_step_1722_);
lean_dec(v_i_1719_);
v_b_1718_ = v_a_1724_;
v_i_1719_ = v___x_1725_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg___boxed(lean_object* v_subst_1747_, lean_object* v_fvarIds_1748_, lean_object* v_range_1749_, lean_object* v_b_1750_, lean_object* v_i_1751_, lean_object* v___y_1752_){
_start:
{
lean_object* v_res_1753_; 
v_res_1753_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg(v_subst_1747_, v_fvarIds_1748_, v_range_1749_, v_b_1750_, v_i_1751_);
lean_dec_ref(v_range_1749_);
lean_dec_ref(v_fvarIds_1748_);
lean_dec_ref(v_subst_1747_);
return v_res_1753_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__0(lean_object* v___x_1754_, lean_object* v___x_1755_, lean_object* v_subst_1756_, lean_object* v_rule_1757_, lean_object* v_fvarIds_1758_, lean_object* v_x_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v_a_1770_; lean_object* v_fst_1771_; lean_object* v_snd_1772_; lean_object* v___x_1773_; 
v___x_1765_ = lean_array_get_size(v_fvarIds_1758_);
v___x_1766_ = lean_mk_empty_array_with_capacity(v___x_1765_);
lean_inc(v___x_1754_);
v___x_1767_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1767_, 0, v___x_1754_);
lean_ctor_set(v___x_1767_, 1, v___x_1765_);
lean_ctor_set(v___x_1767_, 2, v___x_1755_);
lean_inc_ref(v___x_1766_);
v___x_1768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1768_, 0, v___x_1766_);
lean_ctor_set(v___x_1768_, 1, v___x_1766_);
v___x_1769_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg(v_subst_1756_, v_fvarIds_1758_, v___x_1767_, v___x_1768_, v___x_1754_);
lean_dec_ref_known(v___x_1767_, 3);
v_a_1770_ = lean_ctor_get(v___x_1769_, 0);
lean_inc(v_a_1770_);
lean_dec_ref(v___x_1769_);
v_fst_1771_ = lean_ctor_get(v_a_1770_, 0);
lean_inc(v_fst_1771_);
v_snd_1772_ = lean_ctor_get(v_a_1770_, 1);
lean_inc(v_snd_1772_);
lean_dec(v_a_1770_);
v___x_1773_ = l_Lean_Meta_mkAppOptM_x27(v_rule_1757_, v_fst_1771_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
if (lean_obj_tag(v___x_1773_) == 0)
{
lean_object* v_a_1774_; uint8_t v___x_1775_; uint8_t v___x_1776_; uint8_t v___x_1777_; lean_object* v___x_1778_; 
v_a_1774_ = lean_ctor_get(v___x_1773_, 0);
lean_inc(v_a_1774_);
lean_dec_ref_known(v___x_1773_, 1);
v___x_1775_ = 0;
v___x_1776_ = 1;
v___x_1777_ = 1;
v___x_1778_ = l_Lean_Meta_mkLambdaFVars(v_snd_1772_, v_a_1774_, v___x_1775_, v___x_1776_, v___x_1775_, v___x_1776_, v___x_1777_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
lean_dec(v_snd_1772_);
return v___x_1778_;
}
else
{
lean_dec(v_snd_1772_);
return v___x_1773_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__0___boxed(lean_object* v___x_1779_, lean_object* v___x_1780_, lean_object* v_subst_1781_, lean_object* v_rule_1782_, lean_object* v_fvarIds_1783_, lean_object* v_x_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v_res_1790_; 
v_res_1790_ = lp_aesop_Aesop_Substitution_specializeRule___lam__0(v___x_1779_, v___x_1780_, v_subst_1781_, v_rule_1782_, v_fvarIds_1783_, v_x_1784_, v___y_1785_, v___y_1786_, v___y_1787_, v___y_1788_);
lean_dec(v___y_1788_);
lean_dec_ref(v___y_1787_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec_ref(v_x_1784_);
lean_dec_ref(v_fvarIds_1783_);
lean_dec_ref(v_subst_1781_);
return v_res_1790_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__1(lean_object* v_rule_1791_, lean_object* v_subst_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_){
_start:
{
lean_object* v___x_1798_; lean_object* v_a_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v_result_1803_; lean_object* v___x_1805_; uint8_t v_isShared_1806_; uint8_t v_isSharedCheck_1819_; 
lean_inc_ref(v_rule_1791_);
v___x_1798_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Substitution_openRuleType_spec__0___redArg(v_rule_1791_, v___y_1794_);
v_a_1799_ = lean_ctor_get(v___x_1798_, 0);
lean_inc(v_a_1799_);
lean_dec_ref(v___x_1798_);
v___x_1800_ = lean_unsigned_to_nat(0u);
v___x_1801_ = lean_obj_once(&lp_aesop_Aesop_Substitution_openRuleType___closed__3, &lp_aesop_Aesop_Substitution_openRuleType___closed__3_once, _init_lp_aesop_Aesop_Substitution_openRuleType___closed__3);
v___x_1802_ = l_Lean_collectLevelMVars(v___x_1801_, v_a_1799_);
v_result_1803_ = lean_ctor_get(v___x_1802_, 2);
v_isSharedCheck_1819_ = !lean_is_exclusive(v___x_1802_);
if (v_isSharedCheck_1819_ == 0)
{
lean_object* v_unused_1820_; lean_object* v_unused_1821_; 
v_unused_1820_ = lean_ctor_get(v___x_1802_, 1);
lean_dec(v_unused_1820_);
v_unused_1821_ = lean_ctor_get(v___x_1802_, 0);
lean_dec(v_unused_1821_);
v___x_1805_ = v___x_1802_;
v_isShared_1806_ = v_isSharedCheck_1819_;
goto v_resetjp_1804_;
}
else
{
lean_inc(v_result_1803_);
lean_dec(v___x_1802_);
v___x_1805_ = lean_box(0);
v_isShared_1806_ = v_isSharedCheck_1819_;
goto v_resetjp_1804_;
}
v_resetjp_1804_:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1810_; 
v___x_1807_ = lean_array_get_size(v_result_1803_);
v___x_1808_ = lean_unsigned_to_nat(1u);
if (v_isShared_1806_ == 0)
{
lean_ctor_set(v___x_1805_, 2, v___x_1808_);
lean_ctor_set(v___x_1805_, 1, v___x_1807_);
lean_ctor_set(v___x_1805_, 0, v___x_1800_);
v___x_1810_ = v___x_1805_;
goto v_reusejp_1809_;
}
else
{
lean_object* v_reuseFailAlloc_1818_; 
v_reuseFailAlloc_1818_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1818_, 0, v___x_1800_);
lean_ctor_set(v_reuseFailAlloc_1818_, 1, v___x_1807_);
lean_ctor_set(v_reuseFailAlloc_1818_, 2, v___x_1808_);
v___x_1810_ = v_reuseFailAlloc_1818_;
goto v_reusejp_1809_;
}
v_reusejp_1809_:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; 
v___x_1811_ = lean_box(0);
v___x_1812_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_openRuleType_spec__4___redArg(v_subst_1792_, v_result_1803_, v___x_1810_, v___x_1811_, v___x_1800_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_);
lean_dec_ref(v___x_1810_);
lean_dec_ref(v_result_1803_);
lean_dec_ref(v___x_1812_);
lean_inc(v___y_1796_);
lean_inc_ref(v___y_1795_);
lean_inc(v___y_1794_);
lean_inc_ref(v___y_1793_);
lean_inc_ref(v_rule_1791_);
v___x_1813_ = lean_infer_type(v_rule_1791_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_);
if (lean_obj_tag(v___x_1813_) == 0)
{
lean_object* v_a_1814_; lean_object* v___f_1815_; uint8_t v___x_1816_; lean_object* v___x_1817_; 
v_a_1814_ = lean_ctor_get(v___x_1813_, 0);
lean_inc(v_a_1814_);
lean_dec_ref_known(v___x_1813_, 1);
v___f_1815_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Substitution_specializeRule___lam__0___boxed), 11, 4);
lean_closure_set(v___f_1815_, 0, v___x_1800_);
lean_closure_set(v___f_1815_, 1, v___x_1808_);
lean_closure_set(v___f_1815_, 2, v_subst_1792_);
lean_closure_set(v___f_1815_, 3, v_rule_1791_);
v___x_1816_ = 0;
v___x_1817_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_Substitution_specializeRule_spec__1___redArg(v_a_1814_, v___f_1815_, v___x_1816_, v___x_1816_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
return v___x_1817_;
}
else
{
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
lean_dec_ref(v_subst_1792_);
lean_dec_ref(v_rule_1791_);
return v___x_1813_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___lam__1___boxed(lean_object* v_rule_1822_, lean_object* v_subst_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_){
_start:
{
lean_object* v_res_1829_; 
v_res_1829_ = lp_aesop_Aesop_Substitution_specializeRule___lam__1(v_rule_1822_, v_subst_1823_, v___y_1824_, v___y_1825_, v___y_1826_, v___y_1827_);
return v_res_1829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule(lean_object* v_rule_1830_, lean_object* v_subst_1831_, lean_object* v_a_1832_, lean_object* v_a_1833_, lean_object* v_a_1834_, lean_object* v_a_1835_){
_start:
{
lean_object* v___f_1837_; uint8_t v___x_1838_; lean_object* v___x_1839_; 
v___f_1837_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Substitution_specializeRule___lam__1___boxed), 7, 2);
lean_closure_set(v___f_1837_, 0, v_rule_1830_);
lean_closure_set(v___f_1837_, 1, v_subst_1831_);
v___x_1838_ = 0;
v___x_1839_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_Substitution_specializeRule_spec__2___redArg(v___f_1837_, v___x_1838_, v_a_1832_, v_a_1833_, v_a_1834_, v_a_1835_);
return v___x_1839_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Substitution_specializeRule___boxed(lean_object* v_rule_1840_, lean_object* v_subst_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_, lean_object* v_a_1844_, lean_object* v_a_1845_, lean_object* v_a_1846_){
_start:
{
lean_object* v_res_1847_; 
v_res_1847_ = lp_aesop_Aesop_Substitution_specializeRule(v_rule_1840_, v_subst_1841_, v_a_1842_, v_a_1843_, v_a_1844_, v_a_1845_);
lean_dec(v_a_1845_);
lean_dec_ref(v_a_1844_);
lean_dec(v_a_1843_);
lean_dec_ref(v_a_1842_);
return v_res_1847_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0(lean_object* v_subst_1848_, lean_object* v_fvarIds_1849_, lean_object* v_range_1850_, lean_object* v_b_1851_, lean_object* v_i_1852_, lean_object* v_hs_1853_, lean_object* v_hl_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
lean_object* v___x_1860_; 
v___x_1860_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___redArg(v_subst_1848_, v_fvarIds_1849_, v_range_1850_, v_b_1851_, v_i_1852_);
return v___x_1860_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0___boxed(lean_object* v_subst_1861_, lean_object* v_fvarIds_1862_, lean_object* v_range_1863_, lean_object* v_b_1864_, lean_object* v_i_1865_, lean_object* v_hs_1866_, lean_object* v_hl_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_){
_start:
{
lean_object* v_res_1873_; 
v_res_1873_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Substitution_specializeRule_spec__0(v_subst_1861_, v_fvarIds_1862_, v_range_1863_, v_b_1864_, v_i_1865_, v_hs_1866_, v_hl_1867_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
lean_dec(v___y_1871_);
lean_dec_ref(v___y_1870_);
lean_dec(v___y_1869_);
lean_dec_ref(v___y_1868_);
lean_dec_ref(v_range_1863_);
lean_dec_ref(v_fvarIds_1862_);
lean_dec_ref(v_subst_1861_);
return v_res_1873_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_openRuleType(lean_object* v_subst_x3f_1874_, lean_object* v_e_1875_, lean_object* v_a_1876_, lean_object* v_a_1877_, lean_object* v_a_1878_, lean_object* v_a_1879_){
_start:
{
if (lean_obj_tag(v_subst_x3f_1874_) == 0)
{
lean_object* v___x_1881_; 
lean_inc(v_a_1879_);
lean_inc_ref(v_a_1878_);
lean_inc(v_a_1877_);
lean_inc_ref(v_a_1876_);
v___x_1881_ = lean_infer_type(v_e_1875_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_object* v_a_1882_; lean_object* v___x_1883_; uint8_t v___x_1884_; lean_object* v___x_1885_; 
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
lean_inc(v_a_1882_);
lean_dec_ref_known(v___x_1881_, 1);
v___x_1883_ = lean_box(0);
v___x_1884_ = 0;
v___x_1885_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_1882_, v___x_1883_, v___x_1884_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_);
if (lean_obj_tag(v___x_1885_) == 0)
{
lean_object* v_a_1886_; lean_object* v___x_1888_; uint8_t v_isShared_1889_; uint8_t v_isSharedCheck_1905_; 
v_a_1886_ = lean_ctor_get(v___x_1885_, 0);
v_isSharedCheck_1905_ = !lean_is_exclusive(v___x_1885_);
if (v_isSharedCheck_1905_ == 0)
{
v___x_1888_ = v___x_1885_;
v_isShared_1889_ = v_isSharedCheck_1905_;
goto v_resetjp_1887_;
}
else
{
lean_inc(v_a_1886_);
lean_dec(v___x_1885_);
v___x_1888_ = lean_box(0);
v_isShared_1889_ = v_isSharedCheck_1905_;
goto v_resetjp_1887_;
}
v_resetjp_1887_:
{
lean_object* v_fst_1890_; lean_object* v_snd_1891_; lean_object* v___x_1893_; uint8_t v_isShared_1894_; uint8_t v_isSharedCheck_1904_; 
v_fst_1890_ = lean_ctor_get(v_a_1886_, 0);
v_snd_1891_ = lean_ctor_get(v_a_1886_, 1);
v_isSharedCheck_1904_ = !lean_is_exclusive(v_a_1886_);
if (v_isSharedCheck_1904_ == 0)
{
v___x_1893_ = v_a_1886_;
v_isShared_1894_ = v_isSharedCheck_1904_;
goto v_resetjp_1892_;
}
else
{
lean_inc(v_snd_1891_);
lean_inc(v_fst_1890_);
lean_dec(v_a_1886_);
v___x_1893_ = lean_box(0);
v_isShared_1894_ = v_isSharedCheck_1904_;
goto v_resetjp_1892_;
}
v_resetjp_1892_:
{
size_t v_sz_1895_; size_t v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1899_; 
v_sz_1895_ = lean_array_size(v_fst_1890_);
v___x_1896_ = ((size_t)0ULL);
v___x_1897_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Substitution_openRuleType_spec__1(v_sz_1895_, v___x_1896_, v_fst_1890_);
if (v_isShared_1894_ == 0)
{
lean_ctor_set(v___x_1893_, 0, v___x_1897_);
v___x_1899_ = v___x_1893_;
goto v_reusejp_1898_;
}
else
{
lean_object* v_reuseFailAlloc_1903_; 
v_reuseFailAlloc_1903_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1903_, 0, v___x_1897_);
lean_ctor_set(v_reuseFailAlloc_1903_, 1, v_snd_1891_);
v___x_1899_ = v_reuseFailAlloc_1903_;
goto v_reusejp_1898_;
}
v_reusejp_1898_:
{
lean_object* v___x_1901_; 
if (v_isShared_1889_ == 0)
{
lean_ctor_set(v___x_1888_, 0, v___x_1899_);
v___x_1901_ = v___x_1888_;
goto v_reusejp_1900_;
}
else
{
lean_object* v_reuseFailAlloc_1902_; 
v_reuseFailAlloc_1902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1902_, 0, v___x_1899_);
v___x_1901_ = v_reuseFailAlloc_1902_;
goto v_reusejp_1900_;
}
v_reusejp_1900_:
{
return v___x_1901_;
}
}
}
}
}
else
{
lean_object* v_a_1906_; lean_object* v___x_1908_; uint8_t v_isShared_1909_; uint8_t v_isSharedCheck_1913_; 
v_a_1906_ = lean_ctor_get(v___x_1885_, 0);
v_isSharedCheck_1913_ = !lean_is_exclusive(v___x_1885_);
if (v_isSharedCheck_1913_ == 0)
{
v___x_1908_ = v___x_1885_;
v_isShared_1909_ = v_isSharedCheck_1913_;
goto v_resetjp_1907_;
}
else
{
lean_inc(v_a_1906_);
lean_dec(v___x_1885_);
v___x_1908_ = lean_box(0);
v_isShared_1909_ = v_isSharedCheck_1913_;
goto v_resetjp_1907_;
}
v_resetjp_1907_:
{
lean_object* v___x_1911_; 
if (v_isShared_1909_ == 0)
{
v___x_1911_ = v___x_1908_;
goto v_reusejp_1910_;
}
else
{
lean_object* v_reuseFailAlloc_1912_; 
v_reuseFailAlloc_1912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1912_, 0, v_a_1906_);
v___x_1911_ = v_reuseFailAlloc_1912_;
goto v_reusejp_1910_;
}
v_reusejp_1910_:
{
return v___x_1911_;
}
}
}
}
else
{
lean_object* v_a_1914_; lean_object* v___x_1916_; uint8_t v_isShared_1917_; uint8_t v_isSharedCheck_1921_; 
v_a_1914_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1921_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1921_ == 0)
{
v___x_1916_ = v___x_1881_;
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
else
{
lean_inc(v_a_1914_);
lean_dec(v___x_1881_);
v___x_1916_ = lean_box(0);
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
v_resetjp_1915_:
{
lean_object* v___x_1919_; 
if (v_isShared_1917_ == 0)
{
v___x_1919_ = v___x_1916_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1920_; 
v_reuseFailAlloc_1920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1920_, 0, v_a_1914_);
v___x_1919_ = v_reuseFailAlloc_1920_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
return v___x_1919_;
}
}
}
}
else
{
lean_object* v_val_1922_; lean_object* v___x_1923_; 
v_val_1922_ = lean_ctor_get(v_subst_x3f_1874_, 0);
lean_inc(v_val_1922_);
lean_dec_ref_known(v_subst_x3f_1874_, 1);
v___x_1923_ = lp_aesop_Aesop_Substitution_openRuleType(v_e_1875_, v_val_1922_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_);
return v___x_1923_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_openRuleType___boxed(lean_object* v_subst_x3f_1924_, lean_object* v_e_1925_, lean_object* v_a_1926_, lean_object* v_a_1927_, lean_object* v_a_1928_, lean_object* v_a_1929_, lean_object* v_a_1930_){
_start:
{
lean_object* v_res_1931_; 
v_res_1931_ = lp_aesop_Aesop_openRuleType(v_subst_x3f_1924_, v_e_1925_, v_a_1926_, v_a_1927_, v_a_1928_, v_a_1929_);
lean_dec(v_a_1929_);
lean_dec_ref(v_a_1928_);
lean_dec(v_a_1927_);
lean_dec_ref(v_a_1926_);
return v_res_1931_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_LevelIndex(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_LevelIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_PremiseIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_LevelIndex(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_PremiseIndex(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_Substitution(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_LevelIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_PremiseIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_Substitution(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_Substitution(builtin);
}
#ifdef __cplusplus
}
#endif
