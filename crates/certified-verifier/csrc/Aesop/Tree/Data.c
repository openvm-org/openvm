// Lean compiler output
// Module: Aesop.Tree.Data
// Imports: public import Init public meta import Init public import Aesop.Tree.Data.ForwardRuleMatches public import Aesop.Tree.UnsafeQueue public import Aesop.Forward.State import Aesop.Constants import Batteries.Data.Array.Basic
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lp_aesop_Aesop_RegularRule_isSafe(lean_object*);
extern lean_object* lp_aesop_Aesop_nodeUnknownEmoji;
extern lean_object* lp_aesop_Aesop_nodeProvedEmoji;
extern lean_object* lp_aesop_Aesop_nodeUnprovableEmoji;
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_DeclNameGenerator_mkChild(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_UInt64_ofNat___boxed(lean_object*);
extern double lp_aesop_Aesop_unificationGoalPenalty;
double lean_float_of_nat(lean_object*);
double pow(double, double);
double lean_float_mul(double, double);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "no "};
static const lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "yes"};
static const lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(uint8_t);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalId_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalId;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_zero;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_dummy;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instLT;
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalId_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instDecidableRelLt___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_GoalId_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_reprFast, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_GoalId_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalId_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_GoalId_instToString = (const lean_object*)&lp_aesop_Aesop_GoalId_instToString___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_GoalId_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt64_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_GoalId_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalId_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_GoalId_instHashable = (const lean_object*)&lp_aesop_Aesop_GoalId_instHashable___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRappId_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRappId;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_zero;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_dummy;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instLT;
LEAN_EXPORT uint8_t lp_aesop_Aesop_RappId_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instDecidableRelLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RappId_instToString = (const lean_object*)&lp_aesop_Aesop_GoalId_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RappId_instHashable = (const lean_object*)&lp_aesop_Aesop_GoalId_instHashable___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIteration___aux__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIteration;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_one;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_none;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instToString___aux__1(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Iteration_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Iteration_instToString___aux__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Iteration_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_Iteration_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Iteration_instToString = (const lean_object*)&lp_aesop_Aesop_Iteration_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instLT;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instLE;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedNodeState_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedNodeState;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNodeState_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNodeState_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqNodeState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqNodeState_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqNodeState___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqNodeState___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqNodeState = (const lean_object*)&lp_aesop_Aesop_instBEqNodeState___closed__0_value;
static const lean_string_object lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "unknown"};
static const lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "proven"};
static const lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "unprovable"};
static const lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_NodeState_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_NodeState_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_NodeState_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_NodeState_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_NodeState_instToString = (const lean_object*)&lp_aesop_Aesop_NodeState_instToString___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnknown(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnknown___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isProven___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnprovable___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isIrrelevant(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalState_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalState;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalState_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalState_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqGoalState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqGoalState_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqGoalState___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqGoalState___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqGoalState = (const lean_object*)&lp_aesop_Aesop_instBEqGoalState___closed__0_value;
static const lean_string_object lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "provenByRuleApplication"};
static const lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "provenByNormalization"};
static const lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_GoalState_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_GoalState_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_GoalState_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalState_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_GoalState_instToString = (const lean_object*)&lp_aesop_Aesop_GoalState_instToString___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByRuleApplication(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByRuleApplication___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByNormalization(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByNormalization___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProven___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnprovable(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnprovable___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnknown(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnknown___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_toNodeState(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toNodeState___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isIrrelevant(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormalizationState_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormalizationState;
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isNormal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isNormal___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isProvenByNormalization___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalOrigin_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGoalOrigin;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_GoalOrigin_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "subgoal"};
static const lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalOrigin_toString___closed__0_value;
static const lean_string_object lp_aesop_Aesop_GoalOrigin_toString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "copy of "};
static const lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__1 = (const lean_object*)&lp_aesop_Aesop_GoalOrigin_toString___closed__1_value;
static const lean_string_object lp_aesop_Aesop_GoalOrigin_toString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = ", originally "};
static const lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__2 = (const lean_object*)&lp_aesop_Aesop_GoalOrigin_toString___closed__2_value;
static const lean_string_object lp_aesop_Aesop_GoalOrigin_toString___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "dropped mvar"};
static const lean_object* lp_aesop_Aesop_GoalOrigin_toString___closed__3 = (const lean_object*)&lp_aesop_Aesop_GoalOrigin_toString___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_toString(lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1 = (const lean_object*)&lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__0 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__1 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__2 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__3 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__4, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__4 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_treeImpl___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_treeImpl___lam__5___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_treeImpl___closed__5 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_treeImpl___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_treeImpl___closed__0_value),((lean_object*)&lp_aesop_Aesop_treeImpl___closed__1_value),((lean_object*)&lp_aesop_Aesop_treeImpl___closed__2_value),((lean_object*)&lp_aesop_Aesop_treeImpl___closed__3_value),((lean_object*)&lp_aesop_Aesop_treeImpl___closed__4_value),((lean_object*)&lp_aesop_Aesop_treeImpl___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_treeImpl___closed__6 = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__6_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_treeImpl = (const lean_object*)&lp_aesop_Aesop_treeImpl___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_mk(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_parent_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_goals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setGoals(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_state(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_state___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mk(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_id(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parent(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_children(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_origin(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_depth(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_state(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_state___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isForcedUnprovable(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isForcedUnprovable___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_preNormGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_normalizationState(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mvars(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardState(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardRuleMatches(lean_object*);
LEAN_EXPORT double lp_aesop_Aesop_Goal_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_successProbability___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_addedInIteration(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_lastExpandedInIteration(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_failedRapps(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_unsafeRulesSelected(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeRulesSelected___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setChildren(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setOrigin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setDepth(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setPreNormGoal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setNormalizationState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setMVars(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardRuleMatches(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability(double, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setAddedInIteration(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setLastExpandedInIteration(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeQueue(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setFailedRapps(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Goal_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Goal_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Goal_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Goal_instBEq = (const lean_object*)&lp_aesop_Aesop_Goal_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Goal_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Goal_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Goal_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Goal_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Goal_instHashable = (const lean_object*)&lp_aesop_Aesop_Goal_instHashable___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_mk(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_elim(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_modify(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_id(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parent(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_children(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_state(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_state___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevant___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_appliedRule(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_scriptSteps_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_originalSubgoals(lean_object*);
LEAN_EXPORT double lp_aesop_Aesop_Rapp_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_successProbability___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_metaState(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducedMVars(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_assignedMVars(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setId(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setParent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setChildren(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAppliedRule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setScriptSteps_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setOriginalSubgoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability(double, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIntroducedMVars(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAssignedMVars(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rapp_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rapp_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Rapp_instBEq = (const lean_object*)&lp_aesop_Aesop_Rapp_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rapp_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rapp_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rapp_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rapp_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Rapp_instHashable = (const lean_object*)&lp_aesop_Aesop_Rapp_instHashable___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isSafe(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isSafe___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoalAndMetaState_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Goal_safeRapps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Goal_safeRapps___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_safeRapps___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasSafeRapp(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasSafeRapp___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnsafeExhausted(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnsafeExhausted___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isExhausted(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isExhausted___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isActive(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isActive___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasProvableRapp(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasProvableRapp___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasMVar(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasMVar___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0___boxed(lean_object*);
LEAN_EXPORT double lp_aesop_Aesop_Goal_priority(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_priority___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isNormal___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_originalGoalId(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isRoot___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_introducesMVar(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducesMVar___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(uint8_t v_x_7_){
_start:
{
if (v_x_7_ == 0)
{
lean_object* v___x_8_; 
v___x_8_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__1));
return v___x_8_;
}
else
{
lean_object* v___x_9_; 
v___x_9_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___closed__3));
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo___boxed(lean_object* v_x_10_){
_start:
{
uint8_t v_x_34__boxed_11_; lean_object* v_res_12_; 
v_x_34__boxed_11_ = lean_unbox(v_x_10_);
v_res_12_ = lp_aesop___private_Aesop_Tree_Data_0__Bool_toYesNo(v_x_34__boxed_11_);
return v_res_12_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalId_default(void){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_unsigned_to_nat(0u);
return v___x_13_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalId(void){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_unsigned_to_nat(0u);
return v___x_14_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId_decEq(lean_object* v_x_15_, lean_object* v_x_16_){
_start:
{
uint8_t v___x_17_; 
v___x_17_ = lean_nat_dec_eq(v_x_15_, v_x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId_decEq___boxed(lean_object* v_x_18_, lean_object* v_x_19_){
_start:
{
uint8_t v_res_20_; lean_object* v_r_21_; 
v_res_20_ = lp_aesop_Aesop_instDecidableEqGoalId_decEq(v_x_18_, v_x_19_);
lean_dec(v_x_19_);
lean_dec(v_x_18_);
v_r_21_ = lean_box(v_res_20_);
return v_r_21_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqGoalId(lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
uint8_t v___x_24_; 
v___x_24_ = lean_nat_dec_eq(v_x_22_, v_x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqGoalId___boxed(lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
uint8_t v_res_27_; lean_object* v_r_28_; 
v_res_27_ = lp_aesop_Aesop_instDecidableEqGoalId(v_x_25_, v_x_26_);
lean_dec(v_x_26_);
lean_dec(v_x_25_);
v_r_28_ = lean_box(v_res_27_);
return v_r_28_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_zero(void){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_unsigned_to_nat(0u);
return v___x_29_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_one(void){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_unsigned_to_nat(1u);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ(lean_object* v_x_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_unsigned_to_nat(1u);
v___x_33_ = lean_nat_add(v_x_31_, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_succ___boxed(lean_object* v_x_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_aesop_Aesop_GoalId_succ(v_x_34_);
lean_dec(v_x_34_);
return v_res_35_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_dummy(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_cstr_to_nat("1000000000000000");
return v___x_36_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalId_instLT(void){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lean_box(0);
return v___x_37_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalId_instDecidableRelLt(lean_object* v_n_38_, lean_object* v_m_39_){
_start:
{
uint8_t v___x_40_; 
v___x_40_ = lean_nat_dec_lt(v_n_38_, v_m_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalId_instDecidableRelLt___boxed(lean_object* v_n_41_, lean_object* v_m_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_aesop_Aesop_GoalId_instDecidableRelLt(v_n_41_, v_m_42_);
lean_dec(v_m_42_);
lean_dec(v_n_41_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRappId_default(void){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_unsigned_to_nat(0u);
return v___x_49_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRappId(void){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_unsigned_to_nat(0u);
return v___x_50_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId_decEq(lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lean_nat_dec_eq(v_x_51_, v_x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId_decEq___boxed(lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
uint8_t v_res_56_; lean_object* v_r_57_; 
v_res_56_ = lp_aesop_Aesop_instDecidableEqRappId_decEq(v_x_54_, v_x_55_);
lean_dec(v_x_55_);
lean_dec(v_x_54_);
v_r_57_ = lean_box(v_res_56_);
return v_r_57_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instDecidableEqRappId(lean_object* v_x_58_, lean_object* v_x_59_){
_start:
{
uint8_t v___x_60_; 
v___x_60_ = lean_nat_dec_eq(v_x_58_, v_x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instDecidableEqRappId___boxed(lean_object* v_x_61_, lean_object* v_x_62_){
_start:
{
uint8_t v_res_63_; lean_object* v_r_64_; 
v_res_63_ = lp_aesop_Aesop_instDecidableEqRappId(v_x_61_, v_x_62_);
lean_dec(v_x_62_);
lean_dec(v_x_61_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_zero(void){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_unsigned_to_nat(0u);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ(lean_object* v_x_66_){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_nat_add(v_x_66_, v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_succ___boxed(lean_object* v_x_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_aesop_Aesop_RappId_succ(v_x_69_);
lean_dec(v_x_69_);
return v_res_70_;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_one(void){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_unsigned_to_nat(1u);
return v___x_71_;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_dummy(void){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_cstr_to_nat("1000000000000000");
return v___x_72_;
}
}
static lean_object* _init_lp_aesop_Aesop_RappId_instLT(void){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_box(0);
return v___x_73_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RappId_instDecidableRelLt(lean_object* v_n_74_, lean_object* v_m_75_){
_start:
{
uint8_t v___x_76_; 
v___x_76_ = lean_nat_dec_lt(v_n_74_, v_m_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappId_instDecidableRelLt___boxed(lean_object* v_n_77_, lean_object* v_m_78_){
_start:
{
uint8_t v_res_79_; lean_object* v_r_80_; 
v_res_79_ = lp_aesop_Aesop_RappId_instDecidableRelLt(v_n_77_, v_m_78_);
lean_dec(v_m_78_);
lean_dec(v_n_77_);
v_r_80_ = lean_box(v_res_79_);
return v_r_80_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIteration___aux__1(void){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_unsigned_to_nat(0u);
return v___x_83_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIteration(void){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lean_unsigned_to_nat(0u);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(lean_object* v_a_85_){
_start:
{
lean_inc(v_a_85_);
return v_a_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat___boxed(lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_toNat(v_a_86_);
lean_dec(v_a_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(lean_object* v_a_88_){
_start:
{
lean_inc(v_a_88_);
return v_a_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat___boxed(lean_object* v_a_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_aesop___private_Aesop_Tree_Data_0__Aesop_Iteration_ofNat(v_a_89_);
lean_dec(v_a_89_);
return v_res_90_;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_one(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_unsigned_to_nat(1u);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ(lean_object* v_i_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = lean_unsigned_to_nat(1u);
v___x_94_ = lean_nat_add(v_i_92_, v___x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_succ___boxed(lean_object* v_i_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_aesop_Aesop_Iteration_succ(v_i_95_);
lean_dec(v_i_95_);
return v_res_96_;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_none(void){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_unsigned_to_nat(0u);
return v___x_97_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq___aux__1(lean_object* v_a_98_, lean_object* v_b_99_){
_start:
{
uint8_t v___x_100_; 
v___x_100_ = lean_nat_dec_eq(v_a_98_, v_b_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___aux__1___boxed(lean_object* v_a_101_, lean_object* v_b_102_){
_start:
{
uint8_t v_res_103_; lean_object* v_r_104_; 
v_res_103_ = lp_aesop_Aesop_Iteration_instDecidableEq___aux__1(v_a_101_, v_b_102_);
lean_dec(v_b_102_);
lean_dec(v_a_101_);
v_r_104_ = lean_box(v_res_103_);
return v_r_104_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableEq(lean_object* v_a_105_, lean_object* v_b_106_){
_start:
{
uint8_t v___x_107_; 
v___x_107_ = lean_nat_dec_eq(v_a_105_, v_b_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableEq___boxed(lean_object* v_a_108_, lean_object* v_b_109_){
_start:
{
uint8_t v_res_110_; lean_object* v_r_111_; 
v_res_110_ = lp_aesop_Aesop_Iteration_instDecidableEq(v_a_108_, v_b_109_);
lean_dec(v_b_109_);
lean_dec(v_a_108_);
v_r_111_ = lean_box(v_res_110_);
return v_r_111_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instToString___aux__1(lean_object* v_n_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = l_Nat_reprFast(v_n_112_);
return v___x_113_;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_instLT(void){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
}
static lean_object* _init_lp_aesop_Aesop_Iteration_instLE(void){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lean_box(0);
return v___x_117_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt___aux__1(lean_object* v_n_118_, lean_object* v_m_119_){
_start:
{
uint8_t v___x_120_; 
v___x_120_ = lean_nat_dec_lt(v_n_118_, v_m_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___aux__1___boxed(lean_object* v_n_121_, lean_object* v_m_122_){
_start:
{
uint8_t v_res_123_; lean_object* v_r_124_; 
v_res_123_ = lp_aesop_Aesop_Iteration_instDecidableRelLt___aux__1(v_n_121_, v_m_122_);
lean_dec(v_m_122_);
lean_dec(v_n_121_);
v_r_124_ = lean_box(v_res_123_);
return v_r_124_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLt(lean_object* v_a_125_, lean_object* v_b_126_){
_start:
{
uint8_t v___x_127_; 
v___x_127_ = lean_nat_dec_lt(v_a_125_, v_b_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLt___boxed(lean_object* v_a_128_, lean_object* v_b_129_){
_start:
{
uint8_t v_res_130_; lean_object* v_r_131_; 
v_res_130_ = lp_aesop_Aesop_Iteration_instDecidableRelLt(v_a_128_, v_b_129_);
lean_dec(v_b_129_);
lean_dec(v_a_128_);
v_r_131_ = lean_box(v_res_130_);
return v_r_131_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe___aux__1(lean_object* v_n_132_, lean_object* v_m_133_){
_start:
{
uint8_t v___x_134_; 
v___x_134_ = lean_nat_dec_le(v_n_132_, v_m_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___aux__1___boxed(lean_object* v_n_135_, lean_object* v_m_136_){
_start:
{
uint8_t v_res_137_; lean_object* v_r_138_; 
v_res_137_ = lp_aesop_Aesop_Iteration_instDecidableRelLe___aux__1(v_n_135_, v_m_136_);
lean_dec(v_m_136_);
lean_dec(v_n_135_);
v_r_138_ = lean_box(v_res_137_);
return v_r_138_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Iteration_instDecidableRelLe(lean_object* v_a_139_, lean_object* v_b_140_){
_start:
{
uint8_t v___x_141_; 
v___x_141_ = lean_nat_dec_le(v_a_139_, v_b_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Iteration_instDecidableRelLe___boxed(lean_object* v_a_142_, lean_object* v_b_143_){
_start:
{
uint8_t v_res_144_; lean_object* v_r_145_; 
v_res_144_ = lp_aesop_Aesop_Iteration_instDecidableRelLe(v_a_142_, v_b_143_);
lean_dec(v_b_143_);
lean_dec(v_a_142_);
v_r_145_ = lean_box(v_res_144_);
return v_r_145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx(uint8_t v_x_146_){
_start:
{
switch(v_x_146_)
{
case 0:
{
lean_object* v___x_147_; 
v___x_147_ = lean_unsigned_to_nat(0u);
return v___x_147_;
}
case 1:
{
lean_object* v___x_148_; 
v___x_148_ = lean_unsigned_to_nat(1u);
return v___x_148_;
}
default: 
{
lean_object* v___x_149_; 
v___x_149_ = lean_unsigned_to_nat(2u);
return v___x_149_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorIdx___boxed(lean_object* v_x_150_){
_start:
{
uint8_t v_x_boxed_151_; lean_object* v_res_152_; 
v_x_boxed_151_ = lean_unbox(v_x_150_);
v_res_152_ = lp_aesop_Aesop_NodeState_ctorIdx(v_x_boxed_151_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg(lean_object* v_k_153_){
_start:
{
lean_inc(v_k_153_);
return v_k_153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___redArg___boxed(lean_object* v_k_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_aesop_Aesop_NodeState_ctorElim___redArg(v_k_154_);
lean_dec(v_k_154_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim(lean_object* v_motive_156_, lean_object* v_ctorIdx_157_, uint8_t v_t_158_, lean_object* v_h_159_, lean_object* v_k_160_){
_start:
{
lean_inc(v_k_160_);
return v_k_160_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_ctorElim___boxed(lean_object* v_motive_161_, lean_object* v_ctorIdx_162_, lean_object* v_t_163_, lean_object* v_h_164_, lean_object* v_k_165_){
_start:
{
uint8_t v_t_boxed_166_; lean_object* v_res_167_; 
v_t_boxed_166_ = lean_unbox(v_t_163_);
v_res_167_ = lp_aesop_Aesop_NodeState_ctorElim(v_motive_161_, v_ctorIdx_162_, v_t_boxed_166_, v_h_164_, v_k_165_);
lean_dec(v_k_165_);
lean_dec(v_ctorIdx_162_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg(lean_object* v_unknown_168_){
_start:
{
lean_inc(v_unknown_168_);
return v_unknown_168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___redArg___boxed(lean_object* v_unknown_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_aesop_Aesop_NodeState_unknown_elim___redArg(v_unknown_169_);
lean_dec(v_unknown_169_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim(lean_object* v_motive_171_, uint8_t v_t_172_, lean_object* v_h_173_, lean_object* v_unknown_174_){
_start:
{
lean_inc(v_unknown_174_);
return v_unknown_174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unknown_elim___boxed(lean_object* v_motive_175_, lean_object* v_t_176_, lean_object* v_h_177_, lean_object* v_unknown_178_){
_start:
{
uint8_t v_t_boxed_179_; lean_object* v_res_180_; 
v_t_boxed_179_ = lean_unbox(v_t_176_);
v_res_180_ = lp_aesop_Aesop_NodeState_unknown_elim(v_motive_175_, v_t_boxed_179_, v_h_177_, v_unknown_178_);
lean_dec(v_unknown_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg(lean_object* v_proven_181_){
_start:
{
lean_inc(v_proven_181_);
return v_proven_181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___redArg___boxed(lean_object* v_proven_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_aesop_Aesop_NodeState_proven_elim___redArg(v_proven_182_);
lean_dec(v_proven_182_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim(lean_object* v_motive_184_, uint8_t v_t_185_, lean_object* v_h_186_, lean_object* v_proven_187_){
_start:
{
lean_inc(v_proven_187_);
return v_proven_187_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_proven_elim___boxed(lean_object* v_motive_188_, lean_object* v_t_189_, lean_object* v_h_190_, lean_object* v_proven_191_){
_start:
{
uint8_t v_t_boxed_192_; lean_object* v_res_193_; 
v_t_boxed_192_ = lean_unbox(v_t_189_);
v_res_193_ = lp_aesop_Aesop_NodeState_proven_elim(v_motive_188_, v_t_boxed_192_, v_h_190_, v_proven_191_);
lean_dec(v_proven_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg(lean_object* v_unprovable_194_){
_start:
{
lean_inc(v_unprovable_194_);
return v_unprovable_194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___redArg___boxed(lean_object* v_unprovable_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_aesop_Aesop_NodeState_unprovable_elim___redArg(v_unprovable_195_);
lean_dec(v_unprovable_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim(lean_object* v_motive_197_, uint8_t v_t_198_, lean_object* v_h_199_, lean_object* v_unprovable_200_){
_start:
{
lean_inc(v_unprovable_200_);
return v_unprovable_200_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_unprovable_elim___boxed(lean_object* v_motive_201_, lean_object* v_t_202_, lean_object* v_h_203_, lean_object* v_unprovable_204_){
_start:
{
uint8_t v_t_boxed_205_; lean_object* v_res_206_; 
v_t_boxed_205_ = lean_unbox(v_t_202_);
v_res_206_ = lp_aesop_Aesop_NodeState_unprovable_elim(v_motive_201_, v_t_boxed_205_, v_h_203_, v_unprovable_204_);
lean_dec(v_unprovable_204_);
return v_res_206_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedNodeState_default(void){
_start:
{
uint8_t v___x_207_; 
v___x_207_ = 0;
return v___x_207_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedNodeState(void){
_start:
{
uint8_t v___x_208_; 
v___x_208_ = 0;
return v___x_208_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNodeState_beq(uint8_t v_x_209_, uint8_t v_y_210_){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_211_ = lp_aesop_Aesop_NodeState_ctorIdx(v_x_209_);
v___x_212_ = lp_aesop_Aesop_NodeState_ctorIdx(v_y_210_);
v___x_213_ = lean_nat_dec_eq(v___x_211_, v___x_212_);
lean_dec(v___x_212_);
lean_dec(v___x_211_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNodeState_beq___boxed(lean_object* v_x_214_, lean_object* v_y_215_){
_start:
{
uint8_t v_x_17__boxed_216_; uint8_t v_y_18__boxed_217_; uint8_t v_res_218_; lean_object* v_r_219_; 
v_x_17__boxed_216_ = lean_unbox(v_x_214_);
v_y_18__boxed_217_ = lean_unbox(v_y_215_);
v_res_218_ = lp_aesop_Aesop_instBEqNodeState_beq(v_x_17__boxed_216_, v_y_18__boxed_217_);
v_r_219_ = lean_box(v_res_218_);
return v_r_219_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0(uint8_t v_x_225_){
_start:
{
switch(v_x_225_)
{
case 0:
{
lean_object* v___x_226_; 
v___x_226_ = ((lean_object*)(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0));
return v___x_226_;
}
case 1:
{
lean_object* v___x_227_; 
v___x_227_ = ((lean_object*)(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__1));
return v___x_227_;
}
default: 
{
lean_object* v___x_228_; 
v___x_228_ = ((lean_object*)(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2));
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_instToString___lam__0___boxed(lean_object* v_x_229_){
_start:
{
uint8_t v_x_36__boxed_230_; lean_object* v_res_231_; 
v_x_36__boxed_230_ = lean_unbox(v_x_229_);
v_res_231_ = lp_aesop_Aesop_NodeState_instToString___lam__0(v_x_36__boxed_230_);
return v_res_231_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnknown(uint8_t v_x_234_){
_start:
{
if (v_x_234_ == 0)
{
uint8_t v___x_235_; 
v___x_235_ = 1;
return v___x_235_;
}
else
{
uint8_t v___x_236_; 
v___x_236_ = 0;
return v___x_236_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnknown___boxed(lean_object* v_x_237_){
_start:
{
uint8_t v_x_21__boxed_238_; uint8_t v_res_239_; lean_object* v_r_240_; 
v_x_21__boxed_238_ = lean_unbox(v_x_237_);
v_res_239_ = lp_aesop_Aesop_NodeState_isUnknown(v_x_21__boxed_238_);
v_r_240_ = lean_box(v_res_239_);
return v_r_240_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t v_x_241_){
_start:
{
if (v_x_241_ == 1)
{
uint8_t v___x_242_; 
v___x_242_ = 1;
return v___x_242_;
}
else
{
uint8_t v___x_243_; 
v___x_243_ = 0;
return v___x_243_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isProven___boxed(lean_object* v_x_244_){
_start:
{
uint8_t v_x_21__boxed_245_; uint8_t v_res_246_; lean_object* v_r_247_; 
v_x_21__boxed_245_ = lean_unbox(v_x_244_);
v_res_246_ = lp_aesop_Aesop_NodeState_isProven(v_x_21__boxed_245_);
v_r_247_ = lean_box(v_res_246_);
return v_r_247_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t v_x_248_){
_start:
{
if (v_x_248_ == 2)
{
uint8_t v___x_249_; 
v___x_249_ = 1;
return v___x_249_;
}
else
{
uint8_t v___x_250_; 
v___x_250_ = 0;
return v___x_250_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isUnprovable___boxed(lean_object* v_x_251_){
_start:
{
uint8_t v_x_21__boxed_252_; uint8_t v_res_253_; lean_object* v_r_254_; 
v_x_21__boxed_252_ = lean_unbox(v_x_251_);
v_res_253_ = lp_aesop_Aesop_NodeState_isUnprovable(v_x_21__boxed_252_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NodeState_isIrrelevant(uint8_t v_x_255_){
_start:
{
if (v_x_255_ == 0)
{
uint8_t v___x_256_; 
v___x_256_ = 0;
return v___x_256_;
}
else
{
uint8_t v___x_257_; 
v___x_257_ = 1;
return v___x_257_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_isIrrelevant___boxed(lean_object* v_x_258_){
_start:
{
uint8_t v_x_23__boxed_259_; uint8_t v_res_260_; lean_object* v_r_261_; 
v_x_23__boxed_259_ = lean_unbox(v_x_258_);
v_res_260_ = lp_aesop_Aesop_NodeState_isIrrelevant(v_x_23__boxed_259_);
v_r_261_ = lean_box(v_res_260_);
return v_r_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji(uint8_t v_x_262_){
_start:
{
switch(v_x_262_)
{
case 0:
{
lean_object* v___x_263_; 
v___x_263_ = lp_aesop_Aesop_nodeUnknownEmoji;
return v___x_263_;
}
case 1:
{
lean_object* v___x_264_; 
v___x_264_ = lp_aesop_Aesop_nodeProvedEmoji;
return v___x_264_;
}
default: 
{
lean_object* v___x_265_; 
v___x_265_ = lp_aesop_Aesop_nodeUnprovableEmoji;
return v___x_265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NodeState_toEmoji___boxed(lean_object* v_x_266_){
_start:
{
uint8_t v_x_25__boxed_267_; lean_object* v_res_268_; 
v_x_25__boxed_267_ = lean_unbox(v_x_266_);
v_res_268_ = lp_aesop_Aesop_NodeState_toEmoji(v_x_25__boxed_267_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx(uint8_t v_x_269_){
_start:
{
switch(v_x_269_)
{
case 0:
{
lean_object* v___x_270_; 
v___x_270_ = lean_unsigned_to_nat(0u);
return v___x_270_;
}
case 1:
{
lean_object* v___x_271_; 
v___x_271_ = lean_unsigned_to_nat(1u);
return v___x_271_;
}
case 2:
{
lean_object* v___x_272_; 
v___x_272_ = lean_unsigned_to_nat(2u);
return v___x_272_;
}
default: 
{
lean_object* v___x_273_; 
v___x_273_ = lean_unsigned_to_nat(3u);
return v___x_273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorIdx___boxed(lean_object* v_x_274_){
_start:
{
uint8_t v_x_boxed_275_; lean_object* v_res_276_; 
v_x_boxed_275_ = lean_unbox(v_x_274_);
v_res_276_ = lp_aesop_Aesop_GoalState_ctorIdx(v_x_boxed_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg(lean_object* v_k_277_){
_start:
{
lean_inc(v_k_277_);
return v_k_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___redArg___boxed(lean_object* v_k_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_aesop_Aesop_GoalState_ctorElim___redArg(v_k_278_);
lean_dec(v_k_278_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim(lean_object* v_motive_280_, lean_object* v_ctorIdx_281_, uint8_t v_t_282_, lean_object* v_h_283_, lean_object* v_k_284_){
_start:
{
lean_inc(v_k_284_);
return v_k_284_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_ctorElim___boxed(lean_object* v_motive_285_, lean_object* v_ctorIdx_286_, lean_object* v_t_287_, lean_object* v_h_288_, lean_object* v_k_289_){
_start:
{
uint8_t v_t_boxed_290_; lean_object* v_res_291_; 
v_t_boxed_290_ = lean_unbox(v_t_287_);
v_res_291_ = lp_aesop_Aesop_GoalState_ctorElim(v_motive_285_, v_ctorIdx_286_, v_t_boxed_290_, v_h_288_, v_k_289_);
lean_dec(v_k_289_);
lean_dec(v_ctorIdx_286_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg(lean_object* v_unknown_292_){
_start:
{
lean_inc(v_unknown_292_);
return v_unknown_292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___redArg___boxed(lean_object* v_unknown_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_aesop_Aesop_GoalState_unknown_elim___redArg(v_unknown_293_);
lean_dec(v_unknown_293_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim(lean_object* v_motive_295_, uint8_t v_t_296_, lean_object* v_h_297_, lean_object* v_unknown_298_){
_start:
{
lean_inc(v_unknown_298_);
return v_unknown_298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unknown_elim___boxed(lean_object* v_motive_299_, lean_object* v_t_300_, lean_object* v_h_301_, lean_object* v_unknown_302_){
_start:
{
uint8_t v_t_boxed_303_; lean_object* v_res_304_; 
v_t_boxed_303_ = lean_unbox(v_t_300_);
v_res_304_ = lp_aesop_Aesop_GoalState_unknown_elim(v_motive_299_, v_t_boxed_303_, v_h_301_, v_unknown_302_);
lean_dec(v_unknown_302_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(lean_object* v_provenByRuleApplication_305_){
_start:
{
lean_inc(v_provenByRuleApplication_305_);
return v_provenByRuleApplication_305_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg___boxed(lean_object* v_provenByRuleApplication_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___redArg(v_provenByRuleApplication_306_);
lean_dec(v_provenByRuleApplication_306_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(lean_object* v_motive_308_, uint8_t v_t_309_, lean_object* v_h_310_, lean_object* v_provenByRuleApplication_311_){
_start:
{
lean_inc(v_provenByRuleApplication_311_);
return v_provenByRuleApplication_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByRuleApplication_elim___boxed(lean_object* v_motive_312_, lean_object* v_t_313_, lean_object* v_h_314_, lean_object* v_provenByRuleApplication_315_){
_start:
{
uint8_t v_t_boxed_316_; lean_object* v_res_317_; 
v_t_boxed_316_ = lean_unbox(v_t_313_);
v_res_317_ = lp_aesop_Aesop_GoalState_provenByRuleApplication_elim(v_motive_312_, v_t_boxed_316_, v_h_314_, v_provenByRuleApplication_315_);
lean_dec(v_provenByRuleApplication_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(lean_object* v_provenByNormalization_318_){
_start:
{
lean_inc(v_provenByNormalization_318_);
return v_provenByNormalization_318_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg___boxed(lean_object* v_provenByNormalization_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_aesop_Aesop_GoalState_provenByNormalization_elim___redArg(v_provenByNormalization_319_);
lean_dec(v_provenByNormalization_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim(lean_object* v_motive_321_, uint8_t v_t_322_, lean_object* v_h_323_, lean_object* v_provenByNormalization_324_){
_start:
{
lean_inc(v_provenByNormalization_324_);
return v_provenByNormalization_324_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_provenByNormalization_elim___boxed(lean_object* v_motive_325_, lean_object* v_t_326_, lean_object* v_h_327_, lean_object* v_provenByNormalization_328_){
_start:
{
uint8_t v_t_boxed_329_; lean_object* v_res_330_; 
v_t_boxed_329_ = lean_unbox(v_t_326_);
v_res_330_ = lp_aesop_Aesop_GoalState_provenByNormalization_elim(v_motive_325_, v_t_boxed_329_, v_h_327_, v_provenByNormalization_328_);
lean_dec(v_provenByNormalization_328_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg(lean_object* v_unprovable_331_){
_start:
{
lean_inc(v_unprovable_331_);
return v_unprovable_331_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___redArg___boxed(lean_object* v_unprovable_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_aesop_Aesop_GoalState_unprovable_elim___redArg(v_unprovable_332_);
lean_dec(v_unprovable_332_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim(lean_object* v_motive_334_, uint8_t v_t_335_, lean_object* v_h_336_, lean_object* v_unprovable_337_){
_start:
{
lean_inc(v_unprovable_337_);
return v_unprovable_337_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_unprovable_elim___boxed(lean_object* v_motive_338_, lean_object* v_t_339_, lean_object* v_h_340_, lean_object* v_unprovable_341_){
_start:
{
uint8_t v_t_boxed_342_; lean_object* v_res_343_; 
v_t_boxed_342_ = lean_unbox(v_t_339_);
v_res_343_ = lp_aesop_Aesop_GoalState_unprovable_elim(v_motive_338_, v_t_boxed_342_, v_h_340_, v_unprovable_341_);
lean_dec(v_unprovable_341_);
return v_res_343_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalState_default(void){
_start:
{
uint8_t v___x_344_; 
v___x_344_ = 0;
return v___x_344_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalState(void){
_start:
{
uint8_t v___x_345_; 
v___x_345_ = 0;
return v___x_345_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqGoalState_beq(uint8_t v_x_346_, uint8_t v_y_347_){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; uint8_t v___x_350_; 
v___x_348_ = lp_aesop_Aesop_GoalState_ctorIdx(v_x_346_);
v___x_349_ = lp_aesop_Aesop_GoalState_ctorIdx(v_y_347_);
v___x_350_ = lean_nat_dec_eq(v___x_348_, v___x_349_);
lean_dec(v___x_349_);
lean_dec(v___x_348_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqGoalState_beq___boxed(lean_object* v_x_351_, lean_object* v_y_352_){
_start:
{
uint8_t v_x_17__boxed_353_; uint8_t v_y_18__boxed_354_; uint8_t v_res_355_; lean_object* v_r_356_; 
v_x_17__boxed_353_ = lean_unbox(v_x_351_);
v_y_18__boxed_354_ = lean_unbox(v_y_352_);
v_res_355_ = lp_aesop_Aesop_instBEqGoalState_beq(v_x_17__boxed_353_, v_y_18__boxed_354_);
v_r_356_ = lean_box(v_res_355_);
return v_r_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0(uint8_t v_x_361_){
_start:
{
switch(v_x_361_)
{
case 0:
{
lean_object* v___x_362_; 
v___x_362_ = ((lean_object*)(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__0));
return v___x_362_;
}
case 1:
{
lean_object* v___x_363_; 
v___x_363_ = ((lean_object*)(lp_aesop_Aesop_GoalState_instToString___lam__0___closed__0));
return v___x_363_;
}
case 2:
{
lean_object* v___x_364_; 
v___x_364_ = ((lean_object*)(lp_aesop_Aesop_GoalState_instToString___lam__0___closed__1));
return v___x_364_;
}
default: 
{
lean_object* v___x_365_; 
v___x_365_ = ((lean_object*)(lp_aesop_Aesop_NodeState_instToString___lam__0___closed__2));
return v___x_365_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_instToString___lam__0___boxed(lean_object* v_x_366_){
_start:
{
uint8_t v_x_44__boxed_367_; lean_object* v_res_368_; 
v_x_44__boxed_367_ = lean_unbox(v_x_366_);
v_res_368_ = lp_aesop_Aesop_GoalState_instToString___lam__0(v_x_44__boxed_367_);
return v_res_368_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByRuleApplication(uint8_t v_x_371_){
_start:
{
if (v_x_371_ == 1)
{
uint8_t v___x_372_; 
v___x_372_ = 1;
return v___x_372_;
}
else
{
uint8_t v___x_373_; 
v___x_373_ = 0;
return v___x_373_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByRuleApplication___boxed(lean_object* v_x_374_){
_start:
{
uint8_t v_x_21__boxed_375_; uint8_t v_res_376_; lean_object* v_r_377_; 
v_x_21__boxed_375_ = lean_unbox(v_x_374_);
v_res_376_ = lp_aesop_Aesop_GoalState_isProvenByRuleApplication(v_x_21__boxed_375_);
v_r_377_ = lean_box(v_res_376_);
return v_r_377_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProvenByNormalization(uint8_t v_x_378_){
_start:
{
if (v_x_378_ == 2)
{
uint8_t v___x_379_; 
v___x_379_ = 1;
return v___x_379_;
}
else
{
uint8_t v___x_380_; 
v___x_380_ = 0;
return v___x_380_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProvenByNormalization___boxed(lean_object* v_x_381_){
_start:
{
uint8_t v_x_21__boxed_382_; uint8_t v_res_383_; lean_object* v_r_384_; 
v_x_21__boxed_382_ = lean_unbox(v_x_381_);
v_res_383_ = lp_aesop_Aesop_GoalState_isProvenByNormalization(v_x_21__boxed_382_);
v_r_384_ = lean_box(v_res_383_);
return v_r_384_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t v_x_385_){
_start:
{
switch(v_x_385_)
{
case 1:
{
uint8_t v___x_386_; 
v___x_386_ = 1;
return v___x_386_;
}
case 2:
{
uint8_t v___x_387_; 
v___x_387_ = 1;
return v___x_387_;
}
default: 
{
uint8_t v___x_388_; 
v___x_388_ = 0;
return v___x_388_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isProven___boxed(lean_object* v_x_389_){
_start:
{
uint8_t v_x_26__boxed_390_; uint8_t v_res_391_; lean_object* v_r_392_; 
v_x_26__boxed_390_ = lean_unbox(v_x_389_);
v_res_391_ = lp_aesop_Aesop_GoalState_isProven(v_x_26__boxed_390_);
v_r_392_ = lean_box(v_res_391_);
return v_r_392_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnprovable(uint8_t v_x_393_){
_start:
{
if (v_x_393_ == 3)
{
uint8_t v___x_394_; 
v___x_394_ = 1;
return v___x_394_;
}
else
{
uint8_t v___x_395_; 
v___x_395_ = 0;
return v___x_395_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnprovable___boxed(lean_object* v_x_396_){
_start:
{
uint8_t v_x_21__boxed_397_; uint8_t v_res_398_; lean_object* v_r_399_; 
v_x_21__boxed_397_ = lean_unbox(v_x_396_);
v_res_398_ = lp_aesop_Aesop_GoalState_isUnprovable(v_x_21__boxed_397_);
v_r_399_ = lean_box(v_res_398_);
return v_r_399_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isUnknown(uint8_t v_x_400_){
_start:
{
if (v_x_400_ == 0)
{
uint8_t v___x_401_; 
v___x_401_ = 1;
return v___x_401_;
}
else
{
uint8_t v___x_402_; 
v___x_402_ = 0;
return v___x_402_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isUnknown___boxed(lean_object* v_x_403_){
_start:
{
uint8_t v_x_21__boxed_404_; uint8_t v_res_405_; lean_object* v_r_406_; 
v_x_21__boxed_404_ = lean_unbox(v_x_403_);
v_res_405_ = lp_aesop_Aesop_GoalState_isUnknown(v_x_21__boxed_404_);
v_r_406_ = lean_box(v_res_405_);
return v_r_406_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_toNodeState(uint8_t v_x_407_){
_start:
{
switch(v_x_407_)
{
case 0:
{
uint8_t v___x_408_; 
v___x_408_ = 0;
return v___x_408_;
}
case 3:
{
uint8_t v___x_409_; 
v___x_409_ = 2;
return v___x_409_;
}
default: 
{
uint8_t v___x_410_; 
v___x_410_ = 1;
return v___x_410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toNodeState___boxed(lean_object* v_x_411_){
_start:
{
uint8_t v_x_30__boxed_412_; uint8_t v_res_413_; lean_object* v_r_414_; 
v_x_30__boxed_412_ = lean_unbox(v_x_411_);
v_res_413_ = lp_aesop_Aesop_GoalState_toNodeState(v_x_30__boxed_412_);
v_r_414_ = lean_box(v_res_413_);
return v_r_414_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_GoalState_isIrrelevant(uint8_t v_s_415_){
_start:
{
uint8_t v___x_416_; uint8_t v___x_417_; 
v___x_416_ = lp_aesop_Aesop_GoalState_toNodeState(v_s_415_);
v___x_417_ = lp_aesop_Aesop_NodeState_isIrrelevant(v___x_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_isIrrelevant___boxed(lean_object* v_s_418_){
_start:
{
uint8_t v_s_boxed_419_; uint8_t v_res_420_; lean_object* v_r_421_; 
v_s_boxed_419_ = lean_unbox(v_s_418_);
v_res_420_ = lp_aesop_Aesop_GoalState_isIrrelevant(v_s_boxed_419_);
v_r_421_ = lean_box(v_res_420_);
return v_r_421_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji(uint8_t v_x_422_){
_start:
{
switch(v_x_422_)
{
case 0:
{
lean_object* v___x_423_; 
v___x_423_ = lp_aesop_Aesop_nodeUnknownEmoji;
return v___x_423_;
}
case 3:
{
lean_object* v___x_424_; 
v___x_424_ = lp_aesop_Aesop_nodeUnprovableEmoji;
return v___x_424_;
}
default: 
{
lean_object* v___x_425_; 
v___x_425_ = lp_aesop_Aesop_nodeProvedEmoji;
return v___x_425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalState_toEmoji___boxed(lean_object* v_x_426_){
_start:
{
uint8_t v_x_30__boxed_427_; lean_object* v_res_428_; 
v_x_30__boxed_427_ = lean_unbox(v_x_426_);
v_res_428_ = lp_aesop_Aesop_GoalState_toEmoji(v_x_30__boxed_427_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx(lean_object* v_x_429_){
_start:
{
switch(lean_obj_tag(v_x_429_))
{
case 0:
{
lean_object* v___x_430_; 
v___x_430_ = lean_unsigned_to_nat(0u);
return v___x_430_;
}
case 1:
{
lean_object* v___x_431_; 
v___x_431_ = lean_unsigned_to_nat(1u);
return v___x_431_;
}
default: 
{
lean_object* v___x_432_; 
v___x_432_ = lean_unsigned_to_nat(2u);
return v___x_432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorIdx___boxed(lean_object* v_x_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_aesop_Aesop_NormalizationState_ctorIdx(v_x_433_);
lean_dec(v_x_433_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___redArg(lean_object* v_t_435_, lean_object* v_k_436_){
_start:
{
switch(lean_obj_tag(v_t_435_))
{
case 0:
{
return v_k_436_;
}
case 1:
{
lean_object* v_postGoal_437_; lean_object* v_postState_438_; lean_object* v_script_439_; lean_object* v___x_440_; 
v_postGoal_437_ = lean_ctor_get(v_t_435_, 0);
lean_inc(v_postGoal_437_);
v_postState_438_ = lean_ctor_get(v_t_435_, 1);
lean_inc_ref(v_postState_438_);
v_script_439_ = lean_ctor_get(v_t_435_, 2);
lean_inc_ref(v_script_439_);
lean_dec_ref_known(v_t_435_, 3);
v___x_440_ = lean_apply_3(v_k_436_, v_postGoal_437_, v_postState_438_, v_script_439_);
return v___x_440_;
}
default: 
{
lean_object* v_postState_441_; lean_object* v_script_442_; lean_object* v___x_443_; 
v_postState_441_ = lean_ctor_get(v_t_435_, 0);
lean_inc_ref(v_postState_441_);
v_script_442_ = lean_ctor_get(v_t_435_, 1);
lean_inc_ref(v_script_442_);
lean_dec_ref_known(v_t_435_, 2);
v___x_443_ = lean_apply_2(v_k_436_, v_postState_441_, v_script_442_);
return v___x_443_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim(lean_object* v_motive_444_, lean_object* v_ctorIdx_445_, lean_object* v_t_446_, lean_object* v_h_447_, lean_object* v_k_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_446_, v_k_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_ctorElim___boxed(lean_object* v_motive_450_, lean_object* v_ctorIdx_451_, lean_object* v_t_452_, lean_object* v_h_453_, lean_object* v_k_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_aesop_Aesop_NormalizationState_ctorElim(v_motive_450_, v_ctorIdx_451_, v_t_452_, v_h_453_, v_k_454_);
lean_dec(v_ctorIdx_451_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim___redArg(lean_object* v_t_456_, lean_object* v_notNormal_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_456_, v_notNormal_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_notNormal_elim(lean_object* v_motive_459_, lean_object* v_t_460_, lean_object* v_h_461_, lean_object* v_notNormal_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_460_, v_notNormal_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim___redArg(lean_object* v_t_464_, lean_object* v_normal_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_464_, v_normal_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normal_elim(lean_object* v_motive_467_, lean_object* v_t_468_, lean_object* v_h_469_, lean_object* v_normal_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_468_, v_normal_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim___redArg(lean_object* v_t_472_, lean_object* v_provenByNormalization_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_472_, v_provenByNormalization_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_provenByNormalization_elim(lean_object* v_motive_475_, lean_object* v_t_476_, lean_object* v_h_477_, lean_object* v_provenByNormalization_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_aesop_Aesop_NormalizationState_ctorElim___redArg(v_t_476_, v_provenByNormalization_478_);
return v___x_479_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormalizationState_default(void){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lean_box(0);
return v___x_480_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormalizationState(void){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lean_box(0);
return v___x_481_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isNormal(lean_object* v_x_482_){
_start:
{
if (lean_obj_tag(v_x_482_) == 0)
{
uint8_t v___x_483_; 
v___x_483_ = 0;
return v___x_483_;
}
else
{
uint8_t v___x_484_; 
v___x_484_ = 1;
return v___x_484_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isNormal___boxed(lean_object* v_x_485_){
_start:
{
uint8_t v_res_486_; lean_object* v_r_487_; 
v_res_486_ = lp_aesop_Aesop_NormalizationState_isNormal(v_x_485_);
lean_dec(v_x_485_);
v_r_487_ = lean_box(v_res_486_);
return v_r_487_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object* v_x_488_){
_start:
{
if (lean_obj_tag(v_x_488_) == 2)
{
uint8_t v___x_489_; 
v___x_489_ = 1;
return v___x_489_;
}
else
{
uint8_t v___x_490_; 
v___x_490_ = 0;
return v___x_490_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_isProvenByNormalization___boxed(lean_object* v_x_491_){
_start:
{
uint8_t v_res_492_; lean_object* v_r_493_; 
v_res_492_ = lp_aesop_Aesop_NormalizationState_isProvenByNormalization(v_x_491_);
lean_dec(v_x_491_);
v_r_493_ = lean_box(v_res_492_);
return v_r_493_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object* v_x_494_){
_start:
{
if (lean_obj_tag(v_x_494_) == 1)
{
lean_object* v_postGoal_495_; lean_object* v___x_496_; 
v_postGoal_495_ = lean_ctor_get(v_x_494_, 0);
lean_inc(v_postGoal_495_);
v___x_496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_496_, 0, v_postGoal_495_);
return v___x_496_;
}
else
{
lean_object* v___x_497_; 
v___x_497_ = lean_box(0);
return v___x_497_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f___boxed(lean_object* v_x_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(v_x_498_);
lean_dec(v_x_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx(lean_object* v_x_500_){
_start:
{
switch(lean_obj_tag(v_x_500_))
{
case 0:
{
lean_object* v___x_501_; 
v___x_501_ = lean_unsigned_to_nat(0u);
return v___x_501_;
}
case 1:
{
lean_object* v___x_502_; 
v___x_502_ = lean_unsigned_to_nat(1u);
return v___x_502_;
}
default: 
{
lean_object* v___x_503_; 
v___x_503_ = lean_unsigned_to_nat(2u);
return v___x_503_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorIdx___boxed(lean_object* v_x_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_aesop_Aesop_GoalOrigin_ctorIdx(v_x_504_);
lean_dec(v_x_504_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(lean_object* v_t_506_, lean_object* v_k_507_){
_start:
{
if (lean_obj_tag(v_t_506_) == 1)
{
lean_object* v_from_508_; lean_object* v_rep_509_; lean_object* v___x_510_; 
v_from_508_ = lean_ctor_get(v_t_506_, 0);
lean_inc(v_from_508_);
v_rep_509_ = lean_ctor_get(v_t_506_, 1);
lean_inc(v_rep_509_);
lean_dec_ref_known(v_t_506_, 2);
v___x_510_ = lean_apply_2(v_k_507_, v_from_508_, v_rep_509_);
return v___x_510_;
}
else
{
lean_dec(v_t_506_);
return v_k_507_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim(lean_object* v_motive_511_, lean_object* v_ctorIdx_512_, lean_object* v_t_513_, lean_object* v_h_514_, lean_object* v_k_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_513_, v_k_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_ctorElim___boxed(lean_object* v_motive_517_, lean_object* v_ctorIdx_518_, lean_object* v_t_519_, lean_object* v_h_520_, lean_object* v_k_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_aesop_Aesop_GoalOrigin_ctorElim(v_motive_517_, v_ctorIdx_518_, v_t_519_, v_h_520_, v_k_521_);
lean_dec(v_ctorIdx_518_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim___redArg(lean_object* v_t_523_, lean_object* v_subgoal_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_523_, v_subgoal_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_subgoal_elim(lean_object* v_motive_526_, lean_object* v_t_527_, lean_object* v_h_528_, lean_object* v_subgoal_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_527_, v_subgoal_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim___redArg(lean_object* v_t_531_, lean_object* v_copied_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_531_, v_copied_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_copied_elim(lean_object* v_motive_534_, lean_object* v_t_535_, lean_object* v_h_536_, lean_object* v_copied_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_535_, v_copied_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim___redArg(lean_object* v_t_539_, lean_object* v_droppedMVar_540_){
_start:
{
lean_object* v___x_541_; 
v___x_541_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_539_, v_droppedMVar_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_droppedMVar_elim(lean_object* v_motive_542_, lean_object* v_t_543_, lean_object* v_h_544_, lean_object* v_droppedMVar_545_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_aesop_Aesop_GoalOrigin_ctorElim___redArg(v_t_543_, v_droppedMVar_545_);
return v___x_546_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalOrigin_default(void){
_start:
{
lean_object* v___x_547_; 
v___x_547_ = lean_box(0);
return v___x_547_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGoalOrigin(void){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lean_box(0);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(lean_object* v_x_549_){
_start:
{
if (lean_obj_tag(v_x_549_) == 1)
{
lean_object* v_rep_550_; lean_object* v___x_551_; 
v_rep_550_ = lean_ctor_get(v_x_549_, 1);
lean_inc(v_rep_550_);
v___x_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_551_, 0, v_rep_550_);
return v___x_551_;
}
else
{
lean_object* v___x_552_; 
v___x_552_ = lean_box(0);
return v___x_552_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f___boxed(lean_object* v_x_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(v_x_553_);
lean_dec(v_x_553_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalOrigin_toString(lean_object* v_x_559_){
_start:
{
switch(lean_obj_tag(v_x_559_))
{
case 0:
{
lean_object* v___x_560_; 
v___x_560_ = ((lean_object*)(lp_aesop_Aesop_GoalOrigin_toString___closed__0));
return v___x_560_;
}
case 1:
{
lean_object* v_from_561_; lean_object* v_rep_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v_from_561_ = lean_ctor_get(v_x_559_, 0);
lean_inc(v_from_561_);
v_rep_562_ = lean_ctor_get(v_x_559_, 1);
lean_inc(v_rep_562_);
lean_dec_ref_known(v_x_559_, 2);
v___x_563_ = ((lean_object*)(lp_aesop_Aesop_GoalOrigin_toString___closed__1));
v___x_564_ = l_Nat_reprFast(v_from_561_);
v___x_565_ = lean_string_append(v___x_563_, v___x_564_);
lean_dec_ref(v___x_564_);
v___x_566_ = ((lean_object*)(lp_aesop_Aesop_GoalOrigin_toString___closed__2));
v___x_567_ = lean_string_append(v___x_565_, v___x_566_);
v___x_568_ = l_Nat_reprFast(v_rep_562_);
v___x_569_ = lean_string_append(v___x_567_, v___x_568_);
lean_dec_ref(v___x_568_);
return v___x_569_;
}
default: 
{
lean_object* v___x_570_; 
v___x_570_ = ((lean_object*)(lp_aesop_Aesop_GoalOrigin_toString___closed__3));
return v___x_570_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_object* v_Goal_578_, lean_object* v_Rapp_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedMVarClusterData_default___closed__1));
return v___x_580_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0(void){
_start:
{
lean_object* v___x_581_; 
v___x_581_ = lp_aesop_Aesop_instInhabitedMVarClusterData_default(lean_box(0), lean_box(0));
return v___x_581_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMVarClusterData(lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0, &lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0_once, _init_lp_aesop_Aesop_instInhabitedMVarClusterData___closed__0);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__0(lean_object* v_d_585_){
_start:
{
lean_object* v___x_586_; 
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v_d_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1(lean_object* v_x_587_){
_start:
{
lean_object* v_d_588_; 
v_d_588_ = lean_ctor_get(v_x_587_, 0);
lean_inc_ref(v_d_588_);
return v_d_588_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__1___boxed(lean_object* v_x_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_aesop_Aesop_treeImpl___lam__1(v_x_589_);
lean_dec_ref(v_x_589_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__2(lean_object* v_d_591_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_592_, 0, v_d_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3(lean_object* v_x_593_){
_start:
{
lean_object* v_d_594_; 
v_d_594_ = lean_ctor_get(v_x_593_, 0);
lean_inc_ref(v_d_594_);
return v_d_594_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__3___boxed(lean_object* v_x_595_){
_start:
{
lean_object* v_res_596_; 
v_res_596_ = lp_aesop_Aesop_treeImpl___lam__3(v_x_595_);
lean_dec_ref(v_x_595_);
return v_res_596_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__4(lean_object* v_d_597_){
_start:
{
lean_object* v___x_598_; 
v___x_598_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_598_, 0, v_d_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5(lean_object* v_x_599_){
_start:
{
lean_object* v_d_600_; 
v_d_600_ = lean_ctor_get(v_x_599_, 0);
lean_inc_ref(v_d_600_);
return v_d_600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeImpl___lam__5___boxed(lean_object* v_x_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_aesop_Aesop_treeImpl___lam__5(v_x_601_);
lean_dec_ref(v_x_601_);
return v_res_602_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_mk(lean_object* v_a_617_){
_start:
{
lean_object* v___x_618_; lean_object* v_introMVarCluster_619_; lean_object* v___x_620_; 
v___x_618_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_619_ = lean_ctor_get(v___x_618_, 4);
lean_inc(v_introMVarCluster_619_);
v___x_620_ = lean_apply_1(v_introMVarCluster_619_, v_a_617_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_elim(lean_object* v_a_621_){
_start:
{
lean_object* v___x_622_; lean_object* v_elimMVarCluster_623_; lean_object* v___x_624_; 
v___x_622_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_623_ = lean_ctor_get(v___x_622_, 5);
lean_inc_ref(v_elimMVarCluster_623_);
v___x_624_ = lean_apply_1(v_elimMVarCluster_623_, v_a_621_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_modify(lean_object* v_f_625_, lean_object* v_c_626_){
_start:
{
lean_object* v___x_627_; lean_object* v_introMVarCluster_628_; lean_object* v_elimMVarCluster_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; 
v___x_627_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_628_ = lean_ctor_get(v___x_627_, 4);
v_elimMVarCluster_629_ = lean_ctor_get(v___x_627_, 5);
lean_inc_ref(v_elimMVarCluster_629_);
v___x_630_ = lean_apply_1(v_elimMVarCluster_629_, v_c_626_);
v___x_631_ = lean_apply_1(v_f_625_, v___x_630_);
lean_inc(v_introMVarCluster_628_);
v___x_632_ = lean_apply_1(v_introMVarCluster_628_, v___x_631_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_parent_x3f(lean_object* v_c_633_){
_start:
{
lean_object* v___x_634_; lean_object* v_elimMVarCluster_635_; lean_object* v___x_636_; lean_object* v_parent_x3f_637_; 
v___x_634_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_635_ = lean_ctor_get(v___x_634_, 5);
lean_inc_ref(v_elimMVarCluster_635_);
v___x_636_ = lean_apply_1(v_elimMVarCluster_635_, v_c_633_);
v_parent_x3f_637_ = lean_ctor_get(v___x_636_, 0);
lean_inc(v_parent_x3f_637_);
lean_dec_ref(v___x_636_);
return v_parent_x3f_637_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setParent(lean_object* v_parent_x3f_638_, lean_object* v_c_639_){
_start:
{
lean_object* v___x_640_; lean_object* v_introMVarCluster_641_; lean_object* v_elimMVarCluster_642_; lean_object* v___x_643_; lean_object* v_goals_644_; uint8_t v_isIrrelevant_645_; uint8_t v_state_646_; lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_654_; 
v___x_640_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_641_ = lean_ctor_get(v___x_640_, 4);
v_elimMVarCluster_642_ = lean_ctor_get(v___x_640_, 5);
lean_inc_ref(v_elimMVarCluster_642_);
v___x_643_ = lean_apply_1(v_elimMVarCluster_642_, v_c_639_);
v_goals_644_ = lean_ctor_get(v___x_643_, 1);
v_isIrrelevant_645_ = lean_ctor_get_uint8(v___x_643_, sizeof(void*)*2);
v_state_646_ = lean_ctor_get_uint8(v___x_643_, sizeof(void*)*2 + 1);
v_isSharedCheck_654_ = !lean_is_exclusive(v___x_643_);
if (v_isSharedCheck_654_ == 0)
{
lean_object* v_unused_655_; 
v_unused_655_ = lean_ctor_get(v___x_643_, 0);
lean_dec(v_unused_655_);
v___x_648_ = v___x_643_;
v_isShared_649_ = v_isSharedCheck_654_;
goto v_resetjp_647_;
}
else
{
lean_inc(v_goals_644_);
lean_dec(v___x_643_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_654_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v___x_651_; 
if (v_isShared_649_ == 0)
{
lean_ctor_set(v___x_648_, 0, v_parent_x3f_638_);
v___x_651_ = v___x_648_;
goto v_reusejp_650_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v_parent_x3f_638_);
lean_ctor_set(v_reuseFailAlloc_653_, 1, v_goals_644_);
lean_ctor_set_uint8(v_reuseFailAlloc_653_, sizeof(void*)*2, v_isIrrelevant_645_);
lean_ctor_set_uint8(v_reuseFailAlloc_653_, sizeof(void*)*2 + 1, v_state_646_);
v___x_651_ = v_reuseFailAlloc_653_;
goto v_reusejp_650_;
}
v_reusejp_650_:
{
lean_object* v___x_652_; 
lean_inc(v_introMVarCluster_641_);
v___x_652_ = lean_apply_1(v_introMVarCluster_641_, v___x_651_);
return v___x_652_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_goals(lean_object* v_c_656_){
_start:
{
lean_object* v___x_657_; lean_object* v_elimMVarCluster_658_; lean_object* v___x_659_; lean_object* v_goals_660_; 
v___x_657_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_658_ = lean_ctor_get(v___x_657_, 5);
lean_inc_ref(v_elimMVarCluster_658_);
v___x_659_ = lean_apply_1(v_elimMVarCluster_658_, v_c_656_);
v_goals_660_ = lean_ctor_get(v___x_659_, 1);
lean_inc_ref(v_goals_660_);
lean_dec_ref(v___x_659_);
return v_goals_660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setGoals(lean_object* v_goals_661_, lean_object* v_c_662_){
_start:
{
lean_object* v___x_663_; lean_object* v_introMVarCluster_664_; lean_object* v_elimMVarCluster_665_; lean_object* v___x_666_; lean_object* v_parent_x3f_667_; uint8_t v_isIrrelevant_668_; uint8_t v_state_669_; lean_object* v___x_671_; uint8_t v_isShared_672_; uint8_t v_isSharedCheck_677_; 
v___x_663_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_664_ = lean_ctor_get(v___x_663_, 4);
v_elimMVarCluster_665_ = lean_ctor_get(v___x_663_, 5);
lean_inc_ref(v_elimMVarCluster_665_);
v___x_666_ = lean_apply_1(v_elimMVarCluster_665_, v_c_662_);
v_parent_x3f_667_ = lean_ctor_get(v___x_666_, 0);
v_isIrrelevant_668_ = lean_ctor_get_uint8(v___x_666_, sizeof(void*)*2);
v_state_669_ = lean_ctor_get_uint8(v___x_666_, sizeof(void*)*2 + 1);
v_isSharedCheck_677_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_677_ == 0)
{
lean_object* v_unused_678_; 
v_unused_678_ = lean_ctor_get(v___x_666_, 1);
lean_dec(v_unused_678_);
v___x_671_ = v___x_666_;
v_isShared_672_ = v_isSharedCheck_677_;
goto v_resetjp_670_;
}
else
{
lean_inc(v_parent_x3f_667_);
lean_dec(v___x_666_);
v___x_671_ = lean_box(0);
v_isShared_672_ = v_isSharedCheck_677_;
goto v_resetjp_670_;
}
v_resetjp_670_:
{
lean_object* v___x_674_; 
if (v_isShared_672_ == 0)
{
lean_ctor_set(v___x_671_, 1, v_goals_661_);
v___x_674_ = v___x_671_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_676_; 
v_reuseFailAlloc_676_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_676_, 0, v_parent_x3f_667_);
lean_ctor_set(v_reuseFailAlloc_676_, 1, v_goals_661_);
lean_ctor_set_uint8(v_reuseFailAlloc_676_, sizeof(void*)*2, v_isIrrelevant_668_);
lean_ctor_set_uint8(v_reuseFailAlloc_676_, sizeof(void*)*2 + 1, v_state_669_);
v___x_674_ = v_reuseFailAlloc_676_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
lean_object* v___x_675_; 
lean_inc(v_introMVarCluster_664_);
v___x_675_ = lean_apply_1(v_introMVarCluster_664_, v___x_674_);
return v___x_675_;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevant(lean_object* v_c_679_){
_start:
{
lean_object* v___x_680_; lean_object* v_elimMVarCluster_681_; lean_object* v___x_682_; uint8_t v_isIrrelevant_683_; 
v___x_680_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_681_ = lean_ctor_get(v___x_680_, 5);
lean_inc_ref(v_elimMVarCluster_681_);
v___x_682_ = lean_apply_1(v_elimMVarCluster_681_, v_c_679_);
v_isIrrelevant_683_ = lean_ctor_get_uint8(v___x_682_, sizeof(void*)*2);
lean_dec_ref(v___x_682_);
return v_isIrrelevant_683_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevant___boxed(lean_object* v_c_684_){
_start:
{
uint8_t v_res_685_; lean_object* v_r_686_; 
v_res_685_ = lp_aesop_Aesop_MVarCluster_isIrrelevant(v_c_684_);
v_r_686_ = lean_box(v_res_685_);
return v_r_686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant(uint8_t v_isIrrelevant_687_, lean_object* v_c_688_){
_start:
{
lean_object* v___x_689_; lean_object* v_introMVarCluster_690_; lean_object* v_elimMVarCluster_691_; lean_object* v___x_692_; lean_object* v_parent_x3f_693_; lean_object* v_goals_694_; uint8_t v_state_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_703_; 
v___x_689_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_690_ = lean_ctor_get(v___x_689_, 4);
v_elimMVarCluster_691_ = lean_ctor_get(v___x_689_, 5);
lean_inc_ref(v_elimMVarCluster_691_);
v___x_692_ = lean_apply_1(v_elimMVarCluster_691_, v_c_688_);
v_parent_x3f_693_ = lean_ctor_get(v___x_692_, 0);
v_goals_694_ = lean_ctor_get(v___x_692_, 1);
v_state_695_ = lean_ctor_get_uint8(v___x_692_, sizeof(void*)*2 + 1);
v_isSharedCheck_703_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_703_ == 0)
{
v___x_697_ = v___x_692_;
v_isShared_698_ = v_isSharedCheck_703_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_goals_694_);
lean_inc(v_parent_x3f_693_);
lean_dec(v___x_692_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_703_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
lean_object* v___x_700_; 
if (v_isShared_698_ == 0)
{
v___x_700_ = v___x_697_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_702_; 
v_reuseFailAlloc_702_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_702_, 0, v_parent_x3f_693_);
lean_ctor_set(v_reuseFailAlloc_702_, 1, v_goals_694_);
lean_ctor_set_uint8(v_reuseFailAlloc_702_, sizeof(void*)*2 + 1, v_state_695_);
v___x_700_ = v_reuseFailAlloc_702_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
lean_object* v___x_701_; 
lean_ctor_set_uint8(v___x_700_, sizeof(void*)*2, v_isIrrelevant_687_);
lean_inc(v_introMVarCluster_690_);
v___x_701_ = lean_apply_1(v_introMVarCluster_690_, v___x_700_);
return v___x_701_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setIsIrrelevant___boxed(lean_object* v_isIrrelevant_704_, lean_object* v_c_705_){
_start:
{
uint8_t v_isIrrelevant_boxed_706_; lean_object* v_res_707_; 
v_isIrrelevant_boxed_706_ = lean_unbox(v_isIrrelevant_704_);
v_res_707_ = lp_aesop_Aesop_MVarCluster_setIsIrrelevant(v_isIrrelevant_boxed_706_, v_c_705_);
return v_res_707_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_state(lean_object* v_c_708_){
_start:
{
lean_object* v___x_709_; lean_object* v_elimMVarCluster_710_; lean_object* v___x_711_; uint8_t v_state_712_; 
v___x_709_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_710_ = lean_ctor_get(v___x_709_, 5);
lean_inc_ref(v_elimMVarCluster_710_);
v___x_711_ = lean_apply_1(v_elimMVarCluster_710_, v_c_708_);
v_state_712_ = lean_ctor_get_uint8(v___x_711_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_711_);
return v_state_712_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_state___boxed(lean_object* v_c_713_){
_start:
{
uint8_t v_res_714_; lean_object* v_r_715_; 
v_res_714_ = lp_aesop_Aesop_MVarCluster_state(v_c_713_);
v_r_715_ = lean_box(v_res_714_);
return v_r_715_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState(uint8_t v_state_716_, lean_object* v_c_717_){
_start:
{
lean_object* v___x_718_; lean_object* v_introMVarCluster_719_; lean_object* v_elimMVarCluster_720_; lean_object* v___x_721_; lean_object* v_parent_x3f_722_; lean_object* v_goals_723_; uint8_t v_isIrrelevant_724_; lean_object* v___x_726_; uint8_t v_isShared_727_; uint8_t v_isSharedCheck_732_; 
v___x_718_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introMVarCluster_719_ = lean_ctor_get(v___x_718_, 4);
v_elimMVarCluster_720_ = lean_ctor_get(v___x_718_, 5);
lean_inc_ref(v_elimMVarCluster_720_);
v___x_721_ = lean_apply_1(v_elimMVarCluster_720_, v_c_717_);
v_parent_x3f_722_ = lean_ctor_get(v___x_721_, 0);
v_goals_723_ = lean_ctor_get(v___x_721_, 1);
v_isIrrelevant_724_ = lean_ctor_get_uint8(v___x_721_, sizeof(void*)*2);
v_isSharedCheck_732_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_732_ == 0)
{
v___x_726_ = v___x_721_;
v_isShared_727_ = v_isSharedCheck_732_;
goto v_resetjp_725_;
}
else
{
lean_inc(v_goals_723_);
lean_inc(v_parent_x3f_722_);
lean_dec(v___x_721_);
v___x_726_ = lean_box(0);
v_isShared_727_ = v_isSharedCheck_732_;
goto v_resetjp_725_;
}
v_resetjp_725_:
{
lean_object* v___x_729_; 
if (v_isShared_727_ == 0)
{
v___x_729_ = v___x_726_;
goto v_reusejp_728_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_parent_x3f_722_);
lean_ctor_set(v_reuseFailAlloc_731_, 1, v_goals_723_);
lean_ctor_set_uint8(v_reuseFailAlloc_731_, sizeof(void*)*2, v_isIrrelevant_724_);
v___x_729_ = v_reuseFailAlloc_731_;
goto v_reusejp_728_;
}
v_reusejp_728_:
{
lean_object* v___x_730_; 
lean_ctor_set_uint8(v___x_729_, sizeof(void*)*2 + 1, v_state_716_);
lean_inc(v_introMVarCluster_719_);
v___x_730_ = lean_apply_1(v_introMVarCluster_719_, v___x_729_);
return v___x_730_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_setState___boxed(lean_object* v_state_733_, lean_object* v_c_734_){
_start:
{
uint8_t v_state_boxed_735_; lean_object* v_res_736_; 
v_state_boxed_735_ = lean_unbox(v_state_733_);
v_res_736_ = lp_aesop_Aesop_MVarCluster_setState(v_state_boxed_735_, v_c_734_);
return v_res_736_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mk(lean_object* v_a_737_){
_start:
{
lean_object* v___x_738_; lean_object* v_introGoal_739_; lean_object* v___x_740_; 
v___x_738_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_739_ = lean_ctor_get(v___x_738_, 0);
lean_inc(v_introGoal_739_);
v___x_740_ = lean_apply_1(v_introGoal_739_, v_a_737_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_elim(lean_object* v_a_741_){
_start:
{
lean_object* v___x_742_; lean_object* v_elimGoal_743_; lean_object* v___x_744_; 
v___x_742_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_743_ = lean_ctor_get(v___x_742_, 1);
lean_inc_ref(v_elimGoal_743_);
v___x_744_ = lean_apply_1(v_elimGoal_743_, v_a_741_);
return v___x_744_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_modify(lean_object* v_f_745_, lean_object* v_g_746_){
_start:
{
lean_object* v___x_747_; lean_object* v_introGoal_748_; lean_object* v_elimGoal_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; 
v___x_747_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_748_ = lean_ctor_get(v___x_747_, 0);
v_elimGoal_749_ = lean_ctor_get(v___x_747_, 1);
lean_inc_ref(v_elimGoal_749_);
v___x_750_ = lean_apply_1(v_elimGoal_749_, v_g_746_);
v___x_751_ = lean_apply_1(v_f_745_, v___x_750_);
lean_inc(v_introGoal_748_);
v___x_752_ = lean_apply_1(v_introGoal_748_, v___x_751_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_id(lean_object* v_g_753_){
_start:
{
lean_object* v___x_754_; lean_object* v_elimGoal_755_; lean_object* v___x_756_; lean_object* v_id_757_; 
v___x_754_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_755_ = lean_ctor_get(v___x_754_, 1);
lean_inc_ref(v_elimGoal_755_);
v___x_756_ = lean_apply_1(v_elimGoal_755_, v_g_753_);
v_id_757_ = lean_ctor_get(v___x_756_, 0);
lean_inc(v_id_757_);
lean_dec_ref(v___x_756_);
return v_id_757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parent(lean_object* v_g_758_){
_start:
{
lean_object* v___x_759_; lean_object* v_elimGoal_760_; lean_object* v___x_761_; lean_object* v_parent_762_; 
v___x_759_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_760_ = lean_ctor_get(v___x_759_, 1);
lean_inc_ref(v_elimGoal_760_);
v___x_761_ = lean_apply_1(v_elimGoal_760_, v_g_758_);
v_parent_762_ = lean_ctor_get(v___x_761_, 1);
lean_inc(v_parent_762_);
lean_dec_ref(v___x_761_);
return v_parent_762_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_children(lean_object* v_g_763_){
_start:
{
lean_object* v___x_764_; lean_object* v_elimGoal_765_; lean_object* v___x_766_; lean_object* v_children_767_; 
v___x_764_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_765_ = lean_ctor_get(v___x_764_, 1);
lean_inc_ref(v_elimGoal_765_);
v___x_766_ = lean_apply_1(v_elimGoal_765_, v_g_763_);
v_children_767_ = lean_ctor_get(v___x_766_, 2);
lean_inc_ref(v_children_767_);
lean_dec_ref(v___x_766_);
return v_children_767_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_origin(lean_object* v_g_768_){
_start:
{
lean_object* v___x_769_; lean_object* v_elimGoal_770_; lean_object* v___x_771_; lean_object* v_origin_772_; 
v___x_769_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_770_ = lean_ctor_get(v___x_769_, 1);
lean_inc_ref(v_elimGoal_770_);
v___x_771_ = lean_apply_1(v_elimGoal_770_, v_g_768_);
v_origin_772_ = lean_ctor_get(v___x_771_, 3);
lean_inc(v_origin_772_);
lean_dec_ref(v___x_771_);
return v_origin_772_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_depth(lean_object* v_g_773_){
_start:
{
lean_object* v___x_774_; lean_object* v_elimGoal_775_; lean_object* v___x_776_; lean_object* v_depth_777_; 
v___x_774_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_775_ = lean_ctor_get(v___x_774_, 1);
lean_inc_ref(v_elimGoal_775_);
v___x_776_ = lean_apply_1(v_elimGoal_775_, v_g_773_);
v_depth_777_ = lean_ctor_get(v___x_776_, 4);
lean_inc(v_depth_777_);
lean_dec_ref(v___x_776_);
return v_depth_777_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_state(lean_object* v_g_778_){
_start:
{
lean_object* v___x_779_; lean_object* v_elimGoal_780_; lean_object* v___x_781_; uint8_t v_state_782_; 
v___x_779_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_780_ = lean_ctor_get(v___x_779_, 1);
lean_inc_ref(v_elimGoal_780_);
v___x_781_ = lean_apply_1(v_elimGoal_780_, v_g_778_);
v_state_782_ = lean_ctor_get_uint8(v___x_781_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_781_);
return v_state_782_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_state___boxed(lean_object* v_g_783_){
_start:
{
uint8_t v_res_784_; lean_object* v_r_785_; 
v_res_784_ = lp_aesop_Aesop_Goal_state(v_g_783_);
v_r_785_ = lean_box(v_res_784_);
return v_r_785_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevant(lean_object* v_g_786_){
_start:
{
lean_object* v___x_787_; lean_object* v_elimGoal_788_; lean_object* v___x_789_; uint8_t v_isIrrelevant_790_; 
v___x_787_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_788_ = lean_ctor_get(v___x_787_, 1);
lean_inc_ref(v_elimGoal_788_);
v___x_789_ = lean_apply_1(v_elimGoal_788_, v_g_786_);
v_isIrrelevant_790_ = lean_ctor_get_uint8(v___x_789_, sizeof(void*)*14 + 9);
lean_dec_ref(v___x_789_);
return v_isIrrelevant_790_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevant___boxed(lean_object* v_g_791_){
_start:
{
uint8_t v_res_792_; lean_object* v_r_793_; 
v_res_792_ = lp_aesop_Aesop_Goal_isIrrelevant(v_g_791_);
v_r_793_ = lean_box(v_res_792_);
return v_r_793_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isForcedUnprovable(lean_object* v_g_794_){
_start:
{
lean_object* v___x_795_; lean_object* v_elimGoal_796_; lean_object* v___x_797_; uint8_t v_isForcedUnprovable_798_; 
v___x_795_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_796_ = lean_ctor_get(v___x_795_, 1);
lean_inc_ref(v_elimGoal_796_);
v___x_797_ = lean_apply_1(v_elimGoal_796_, v_g_794_);
v_isForcedUnprovable_798_ = lean_ctor_get_uint8(v___x_797_, sizeof(void*)*14 + 10);
lean_dec_ref(v___x_797_);
return v_isForcedUnprovable_798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isForcedUnprovable___boxed(lean_object* v_g_799_){
_start:
{
uint8_t v_res_800_; lean_object* v_r_801_; 
v_res_800_ = lp_aesop_Aesop_Goal_isForcedUnprovable(v_g_799_);
v_r_801_ = lean_box(v_res_800_);
return v_r_801_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_preNormGoal(lean_object* v_g_802_){
_start:
{
lean_object* v___x_803_; lean_object* v_elimGoal_804_; lean_object* v___x_805_; lean_object* v_preNormGoal_806_; 
v___x_803_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_804_ = lean_ctor_get(v___x_803_, 1);
lean_inc_ref(v_elimGoal_804_);
v___x_805_ = lean_apply_1(v_elimGoal_804_, v_g_802_);
v_preNormGoal_806_ = lean_ctor_get(v___x_805_, 5);
lean_inc(v_preNormGoal_806_);
lean_dec_ref(v___x_805_);
return v_preNormGoal_806_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_normalizationState(lean_object* v_g_807_){
_start:
{
lean_object* v___x_808_; lean_object* v_elimGoal_809_; lean_object* v___x_810_; lean_object* v_normalizationState_811_; 
v___x_808_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_809_ = lean_ctor_get(v___x_808_, 1);
lean_inc_ref(v_elimGoal_809_);
v___x_810_ = lean_apply_1(v_elimGoal_809_, v_g_807_);
v_normalizationState_811_ = lean_ctor_get(v___x_810_, 6);
lean_inc(v_normalizationState_811_);
lean_dec_ref(v___x_810_);
return v_normalizationState_811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_mvars(lean_object* v_g_812_){
_start:
{
lean_object* v___x_813_; lean_object* v_elimGoal_814_; lean_object* v___x_815_; lean_object* v_mvars_816_; 
v___x_813_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_814_ = lean_ctor_get(v___x_813_, 1);
lean_inc_ref(v_elimGoal_814_);
v___x_815_ = lean_apply_1(v_elimGoal_814_, v_g_812_);
v_mvars_816_ = lean_ctor_get(v___x_815_, 7);
lean_inc_ref(v_mvars_816_);
lean_dec_ref(v___x_815_);
return v_mvars_816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardState(lean_object* v_g_817_){
_start:
{
lean_object* v___x_818_; lean_object* v_elimGoal_819_; lean_object* v___x_820_; lean_object* v_forwardState_821_; 
v___x_818_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_819_ = lean_ctor_get(v___x_818_, 1);
lean_inc_ref(v_elimGoal_819_);
v___x_820_ = lean_apply_1(v_elimGoal_819_, v_g_817_);
v_forwardState_821_ = lean_ctor_get(v___x_820_, 8);
lean_inc_ref(v_forwardState_821_);
lean_dec_ref(v___x_820_);
return v_forwardState_821_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_forwardRuleMatches(lean_object* v_g_822_){
_start:
{
lean_object* v___x_823_; lean_object* v_elimGoal_824_; lean_object* v___x_825_; lean_object* v_forwardRuleMatches_826_; 
v___x_823_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_824_ = lean_ctor_get(v___x_823_, 1);
lean_inc_ref(v_elimGoal_824_);
v___x_825_ = lean_apply_1(v_elimGoal_824_, v_g_822_);
v_forwardRuleMatches_826_ = lean_ctor_get(v___x_825_, 9);
lean_inc_ref(v_forwardRuleMatches_826_);
lean_dec_ref(v___x_825_);
return v_forwardRuleMatches_826_;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Goal_successProbability(lean_object* v_g_827_){
_start:
{
lean_object* v___x_828_; lean_object* v_elimGoal_829_; lean_object* v___x_830_; double v_successProbability_831_; 
v___x_828_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_829_ = lean_ctor_get(v___x_828_, 1);
lean_inc_ref(v_elimGoal_829_);
v___x_830_ = lean_apply_1(v_elimGoal_829_, v_g_827_);
v_successProbability_831_ = lean_ctor_get_float(v___x_830_, sizeof(void*)*14);
lean_dec_ref(v___x_830_);
return v_successProbability_831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_successProbability___boxed(lean_object* v_g_832_){
_start:
{
double v_res_833_; lean_object* v_r_834_; 
v_res_833_ = lp_aesop_Aesop_Goal_successProbability(v_g_832_);
v_r_834_ = lean_box_float(v_res_833_);
return v_r_834_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_addedInIteration(lean_object* v_g_835_){
_start:
{
lean_object* v___x_836_; lean_object* v_elimGoal_837_; lean_object* v___x_838_; lean_object* v_addedInIteration_839_; 
v___x_836_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_837_ = lean_ctor_get(v___x_836_, 1);
lean_inc_ref(v_elimGoal_837_);
v___x_838_ = lean_apply_1(v_elimGoal_837_, v_g_835_);
v_addedInIteration_839_ = lean_ctor_get(v___x_838_, 10);
lean_inc(v_addedInIteration_839_);
lean_dec_ref(v___x_838_);
return v_addedInIteration_839_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_lastExpandedInIteration(lean_object* v_g_840_){
_start:
{
lean_object* v___x_841_; lean_object* v_elimGoal_842_; lean_object* v___x_843_; lean_object* v_lastExpandedInIteration_844_; 
v___x_841_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_842_ = lean_ctor_get(v___x_841_, 1);
lean_inc_ref(v_elimGoal_842_);
v___x_843_ = lean_apply_1(v_elimGoal_842_, v_g_840_);
v_lastExpandedInIteration_844_ = lean_ctor_get(v___x_843_, 11);
lean_inc(v_lastExpandedInIteration_844_);
lean_dec_ref(v___x_843_);
return v_lastExpandedInIteration_844_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_failedRapps(lean_object* v_g_845_){
_start:
{
lean_object* v___x_846_; lean_object* v_elimGoal_847_; lean_object* v___x_848_; lean_object* v_failedRapps_849_; 
v___x_846_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_847_ = lean_ctor_get(v___x_846_, 1);
lean_inc_ref(v_elimGoal_847_);
v___x_848_ = lean_apply_1(v_elimGoal_847_, v_g_845_);
v_failedRapps_849_ = lean_ctor_get(v___x_848_, 13);
lean_inc_ref(v_failedRapps_849_);
lean_dec_ref(v___x_848_);
return v_failedRapps_849_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_unsafeRulesSelected(lean_object* v_g_850_){
_start:
{
lean_object* v___x_851_; lean_object* v_elimGoal_852_; lean_object* v___x_853_; uint8_t v_unsafeRulesSelected_854_; 
v___x_851_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_852_ = lean_ctor_get(v___x_851_, 1);
lean_inc_ref(v_elimGoal_852_);
v___x_853_ = lean_apply_1(v_elimGoal_852_, v_g_850_);
v_unsafeRulesSelected_854_ = lean_ctor_get_uint8(v___x_853_, sizeof(void*)*14 + 11);
lean_dec_ref(v___x_853_);
return v_unsafeRulesSelected_854_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeRulesSelected___boxed(lean_object* v_g_855_){
_start:
{
uint8_t v_res_856_; lean_object* v_r_857_; 
v_res_856_ = lp_aesop_Aesop_Goal_unsafeRulesSelected(v_g_855_);
v_r_857_ = lean_box(v_res_856_);
return v_r_857_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue(lean_object* v_g_858_){
_start:
{
lean_object* v___x_859_; lean_object* v_elimGoal_860_; lean_object* v___x_861_; lean_object* v_unsafeQueue_862_; 
v___x_859_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_860_ = lean_ctor_get(v___x_859_, 1);
lean_inc_ref(v_elimGoal_860_);
v___x_861_ = lean_apply_1(v_elimGoal_860_, v_g_858_);
v_unsafeQueue_862_ = lean_ctor_get(v___x_861_, 12);
lean_inc_ref(v_unsafeQueue_862_);
lean_dec_ref(v___x_861_);
return v_unsafeQueue_862_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_unsafeQueue_x3f(lean_object* v_g_863_){
_start:
{
lean_object* v___x_864_; lean_object* v_elimGoal_865_; lean_object* v___x_866_; uint8_t v_unsafeRulesSelected_867_; 
v___x_864_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_865_ = lean_ctor_get(v___x_864_, 1);
lean_inc_ref(v_elimGoal_865_);
v___x_866_ = lean_apply_1(v_elimGoal_865_, v_g_863_);
v_unsafeRulesSelected_867_ = lean_ctor_get_uint8(v___x_866_, sizeof(void*)*14 + 11);
if (v_unsafeRulesSelected_867_ == 0)
{
lean_object* v___x_868_; 
lean_dec_ref(v___x_866_);
v___x_868_ = lean_box(0);
return v___x_868_;
}
else
{
lean_object* v_unsafeQueue_869_; lean_object* v___x_870_; 
v_unsafeQueue_869_ = lean_ctor_get(v___x_866_, 12);
lean_inc_ref(v_unsafeQueue_869_);
lean_dec_ref(v___x_866_);
v___x_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_870_, 0, v_unsafeQueue_869_);
return v___x_870_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setId(lean_object* v_id_871_, lean_object* v_g_872_){
_start:
{
lean_object* v___x_873_; lean_object* v_introGoal_874_; lean_object* v_elimGoal_875_; lean_object* v___x_876_; lean_object* v_parent_877_; lean_object* v_children_878_; lean_object* v_origin_879_; lean_object* v_depth_880_; uint8_t v_state_881_; uint8_t v_isIrrelevant_882_; uint8_t v_isForcedUnprovable_883_; lean_object* v_preNormGoal_884_; lean_object* v_normalizationState_885_; lean_object* v_mvars_886_; lean_object* v_forwardState_887_; lean_object* v_forwardRuleMatches_888_; double v_successProbability_889_; lean_object* v_addedInIteration_890_; lean_object* v_lastExpandedInIteration_891_; uint8_t v_unsafeRulesSelected_892_; lean_object* v_unsafeQueue_893_; lean_object* v_failedRapps_894_; lean_object* v___x_896_; uint8_t v_isShared_897_; uint8_t v_isSharedCheck_902_; 
v___x_873_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_874_ = lean_ctor_get(v___x_873_, 0);
v_elimGoal_875_ = lean_ctor_get(v___x_873_, 1);
lean_inc_ref(v_elimGoal_875_);
v___x_876_ = lean_apply_1(v_elimGoal_875_, v_g_872_);
v_parent_877_ = lean_ctor_get(v___x_876_, 1);
v_children_878_ = lean_ctor_get(v___x_876_, 2);
v_origin_879_ = lean_ctor_get(v___x_876_, 3);
v_depth_880_ = lean_ctor_get(v___x_876_, 4);
v_state_881_ = lean_ctor_get_uint8(v___x_876_, sizeof(void*)*14 + 8);
v_isIrrelevant_882_ = lean_ctor_get_uint8(v___x_876_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_883_ = lean_ctor_get_uint8(v___x_876_, sizeof(void*)*14 + 10);
v_preNormGoal_884_ = lean_ctor_get(v___x_876_, 5);
v_normalizationState_885_ = lean_ctor_get(v___x_876_, 6);
v_mvars_886_ = lean_ctor_get(v___x_876_, 7);
v_forwardState_887_ = lean_ctor_get(v___x_876_, 8);
v_forwardRuleMatches_888_ = lean_ctor_get(v___x_876_, 9);
v_successProbability_889_ = lean_ctor_get_float(v___x_876_, sizeof(void*)*14);
v_addedInIteration_890_ = lean_ctor_get(v___x_876_, 10);
v_lastExpandedInIteration_891_ = lean_ctor_get(v___x_876_, 11);
v_unsafeRulesSelected_892_ = lean_ctor_get_uint8(v___x_876_, sizeof(void*)*14 + 11);
v_unsafeQueue_893_ = lean_ctor_get(v___x_876_, 12);
v_failedRapps_894_ = lean_ctor_get(v___x_876_, 13);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_876_);
if (v_isSharedCheck_902_ == 0)
{
lean_object* v_unused_903_; 
v_unused_903_ = lean_ctor_get(v___x_876_, 0);
lean_dec(v_unused_903_);
v___x_896_ = v___x_876_;
v_isShared_897_ = v_isSharedCheck_902_;
goto v_resetjp_895_;
}
else
{
lean_inc(v_failedRapps_894_);
lean_inc(v_unsafeQueue_893_);
lean_inc(v_lastExpandedInIteration_891_);
lean_inc(v_addedInIteration_890_);
lean_inc(v_forwardRuleMatches_888_);
lean_inc(v_forwardState_887_);
lean_inc(v_mvars_886_);
lean_inc(v_normalizationState_885_);
lean_inc(v_preNormGoal_884_);
lean_inc(v_depth_880_);
lean_inc(v_origin_879_);
lean_inc(v_children_878_);
lean_inc(v_parent_877_);
lean_dec(v___x_876_);
v___x_896_ = lean_box(0);
v_isShared_897_ = v_isSharedCheck_902_;
goto v_resetjp_895_;
}
v_resetjp_895_:
{
lean_object* v___x_899_; 
if (v_isShared_897_ == 0)
{
lean_ctor_set(v___x_896_, 0, v_id_871_);
v___x_899_ = v___x_896_;
goto v_reusejp_898_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_id_871_);
lean_ctor_set(v_reuseFailAlloc_901_, 1, v_parent_877_);
lean_ctor_set(v_reuseFailAlloc_901_, 2, v_children_878_);
lean_ctor_set(v_reuseFailAlloc_901_, 3, v_origin_879_);
lean_ctor_set(v_reuseFailAlloc_901_, 4, v_depth_880_);
lean_ctor_set(v_reuseFailAlloc_901_, 5, v_preNormGoal_884_);
lean_ctor_set(v_reuseFailAlloc_901_, 6, v_normalizationState_885_);
lean_ctor_set(v_reuseFailAlloc_901_, 7, v_mvars_886_);
lean_ctor_set(v_reuseFailAlloc_901_, 8, v_forwardState_887_);
lean_ctor_set(v_reuseFailAlloc_901_, 9, v_forwardRuleMatches_888_);
lean_ctor_set(v_reuseFailAlloc_901_, 10, v_addedInIteration_890_);
lean_ctor_set(v_reuseFailAlloc_901_, 11, v_lastExpandedInIteration_891_);
lean_ctor_set(v_reuseFailAlloc_901_, 12, v_unsafeQueue_893_);
lean_ctor_set(v_reuseFailAlloc_901_, 13, v_failedRapps_894_);
lean_ctor_set_uint8(v_reuseFailAlloc_901_, sizeof(void*)*14 + 8, v_state_881_);
lean_ctor_set_uint8(v_reuseFailAlloc_901_, sizeof(void*)*14 + 9, v_isIrrelevant_882_);
lean_ctor_set_uint8(v_reuseFailAlloc_901_, sizeof(void*)*14 + 10, v_isForcedUnprovable_883_);
lean_ctor_set_float(v_reuseFailAlloc_901_, sizeof(void*)*14, v_successProbability_889_);
lean_ctor_set_uint8(v_reuseFailAlloc_901_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_892_);
v___x_899_ = v_reuseFailAlloc_901_;
goto v_reusejp_898_;
}
v_reusejp_898_:
{
lean_object* v___x_900_; 
lean_inc(v_introGoal_874_);
v___x_900_ = lean_apply_1(v_introGoal_874_, v___x_899_);
return v___x_900_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setParent(lean_object* v_parent_904_, lean_object* v_g_905_){
_start:
{
lean_object* v___x_906_; lean_object* v_introGoal_907_; lean_object* v_elimGoal_908_; lean_object* v___x_909_; lean_object* v_id_910_; lean_object* v_children_911_; lean_object* v_origin_912_; lean_object* v_depth_913_; uint8_t v_state_914_; uint8_t v_isIrrelevant_915_; uint8_t v_isForcedUnprovable_916_; lean_object* v_preNormGoal_917_; lean_object* v_normalizationState_918_; lean_object* v_mvars_919_; lean_object* v_forwardState_920_; lean_object* v_forwardRuleMatches_921_; double v_successProbability_922_; lean_object* v_addedInIteration_923_; lean_object* v_lastExpandedInIteration_924_; uint8_t v_unsafeRulesSelected_925_; lean_object* v_unsafeQueue_926_; lean_object* v_failedRapps_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_935_; 
v___x_906_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_907_ = lean_ctor_get(v___x_906_, 0);
v_elimGoal_908_ = lean_ctor_get(v___x_906_, 1);
lean_inc_ref(v_elimGoal_908_);
v___x_909_ = lean_apply_1(v_elimGoal_908_, v_g_905_);
v_id_910_ = lean_ctor_get(v___x_909_, 0);
v_children_911_ = lean_ctor_get(v___x_909_, 2);
v_origin_912_ = lean_ctor_get(v___x_909_, 3);
v_depth_913_ = lean_ctor_get(v___x_909_, 4);
v_state_914_ = lean_ctor_get_uint8(v___x_909_, sizeof(void*)*14 + 8);
v_isIrrelevant_915_ = lean_ctor_get_uint8(v___x_909_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_916_ = lean_ctor_get_uint8(v___x_909_, sizeof(void*)*14 + 10);
v_preNormGoal_917_ = lean_ctor_get(v___x_909_, 5);
v_normalizationState_918_ = lean_ctor_get(v___x_909_, 6);
v_mvars_919_ = lean_ctor_get(v___x_909_, 7);
v_forwardState_920_ = lean_ctor_get(v___x_909_, 8);
v_forwardRuleMatches_921_ = lean_ctor_get(v___x_909_, 9);
v_successProbability_922_ = lean_ctor_get_float(v___x_909_, sizeof(void*)*14);
v_addedInIteration_923_ = lean_ctor_get(v___x_909_, 10);
v_lastExpandedInIteration_924_ = lean_ctor_get(v___x_909_, 11);
v_unsafeRulesSelected_925_ = lean_ctor_get_uint8(v___x_909_, sizeof(void*)*14 + 11);
v_unsafeQueue_926_ = lean_ctor_get(v___x_909_, 12);
v_failedRapps_927_ = lean_ctor_get(v___x_909_, 13);
v_isSharedCheck_935_ = !lean_is_exclusive(v___x_909_);
if (v_isSharedCheck_935_ == 0)
{
lean_object* v_unused_936_; 
v_unused_936_ = lean_ctor_get(v___x_909_, 1);
lean_dec(v_unused_936_);
v___x_929_ = v___x_909_;
v_isShared_930_ = v_isSharedCheck_935_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_failedRapps_927_);
lean_inc(v_unsafeQueue_926_);
lean_inc(v_lastExpandedInIteration_924_);
lean_inc(v_addedInIteration_923_);
lean_inc(v_forwardRuleMatches_921_);
lean_inc(v_forwardState_920_);
lean_inc(v_mvars_919_);
lean_inc(v_normalizationState_918_);
lean_inc(v_preNormGoal_917_);
lean_inc(v_depth_913_);
lean_inc(v_origin_912_);
lean_inc(v_children_911_);
lean_inc(v_id_910_);
lean_dec(v___x_909_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_935_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 1, v_parent_904_);
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v_id_910_);
lean_ctor_set(v_reuseFailAlloc_934_, 1, v_parent_904_);
lean_ctor_set(v_reuseFailAlloc_934_, 2, v_children_911_);
lean_ctor_set(v_reuseFailAlloc_934_, 3, v_origin_912_);
lean_ctor_set(v_reuseFailAlloc_934_, 4, v_depth_913_);
lean_ctor_set(v_reuseFailAlloc_934_, 5, v_preNormGoal_917_);
lean_ctor_set(v_reuseFailAlloc_934_, 6, v_normalizationState_918_);
lean_ctor_set(v_reuseFailAlloc_934_, 7, v_mvars_919_);
lean_ctor_set(v_reuseFailAlloc_934_, 8, v_forwardState_920_);
lean_ctor_set(v_reuseFailAlloc_934_, 9, v_forwardRuleMatches_921_);
lean_ctor_set(v_reuseFailAlloc_934_, 10, v_addedInIteration_923_);
lean_ctor_set(v_reuseFailAlloc_934_, 11, v_lastExpandedInIteration_924_);
lean_ctor_set(v_reuseFailAlloc_934_, 12, v_unsafeQueue_926_);
lean_ctor_set(v_reuseFailAlloc_934_, 13, v_failedRapps_927_);
lean_ctor_set_uint8(v_reuseFailAlloc_934_, sizeof(void*)*14 + 8, v_state_914_);
lean_ctor_set_uint8(v_reuseFailAlloc_934_, sizeof(void*)*14 + 9, v_isIrrelevant_915_);
lean_ctor_set_uint8(v_reuseFailAlloc_934_, sizeof(void*)*14 + 10, v_isForcedUnprovable_916_);
lean_ctor_set_float(v_reuseFailAlloc_934_, sizeof(void*)*14, v_successProbability_922_);
lean_ctor_set_uint8(v_reuseFailAlloc_934_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_925_);
v___x_932_ = v_reuseFailAlloc_934_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
lean_object* v___x_933_; 
lean_inc(v_introGoal_907_);
v___x_933_ = lean_apply_1(v_introGoal_907_, v___x_932_);
return v___x_933_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setChildren(lean_object* v_children_937_, lean_object* v_g_938_){
_start:
{
lean_object* v___x_939_; lean_object* v_introGoal_940_; lean_object* v_elimGoal_941_; lean_object* v___x_942_; lean_object* v_id_943_; lean_object* v_parent_944_; lean_object* v_origin_945_; lean_object* v_depth_946_; uint8_t v_state_947_; uint8_t v_isIrrelevant_948_; uint8_t v_isForcedUnprovable_949_; lean_object* v_preNormGoal_950_; lean_object* v_normalizationState_951_; lean_object* v_mvars_952_; lean_object* v_forwardState_953_; lean_object* v_forwardRuleMatches_954_; double v_successProbability_955_; lean_object* v_addedInIteration_956_; lean_object* v_lastExpandedInIteration_957_; uint8_t v_unsafeRulesSelected_958_; lean_object* v_unsafeQueue_959_; lean_object* v_failedRapps_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_968_; 
v___x_939_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_940_ = lean_ctor_get(v___x_939_, 0);
v_elimGoal_941_ = lean_ctor_get(v___x_939_, 1);
lean_inc_ref(v_elimGoal_941_);
v___x_942_ = lean_apply_1(v_elimGoal_941_, v_g_938_);
v_id_943_ = lean_ctor_get(v___x_942_, 0);
v_parent_944_ = lean_ctor_get(v___x_942_, 1);
v_origin_945_ = lean_ctor_get(v___x_942_, 3);
v_depth_946_ = lean_ctor_get(v___x_942_, 4);
v_state_947_ = lean_ctor_get_uint8(v___x_942_, sizeof(void*)*14 + 8);
v_isIrrelevant_948_ = lean_ctor_get_uint8(v___x_942_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_949_ = lean_ctor_get_uint8(v___x_942_, sizeof(void*)*14 + 10);
v_preNormGoal_950_ = lean_ctor_get(v___x_942_, 5);
v_normalizationState_951_ = lean_ctor_get(v___x_942_, 6);
v_mvars_952_ = lean_ctor_get(v___x_942_, 7);
v_forwardState_953_ = lean_ctor_get(v___x_942_, 8);
v_forwardRuleMatches_954_ = lean_ctor_get(v___x_942_, 9);
v_successProbability_955_ = lean_ctor_get_float(v___x_942_, sizeof(void*)*14);
v_addedInIteration_956_ = lean_ctor_get(v___x_942_, 10);
v_lastExpandedInIteration_957_ = lean_ctor_get(v___x_942_, 11);
v_unsafeRulesSelected_958_ = lean_ctor_get_uint8(v___x_942_, sizeof(void*)*14 + 11);
v_unsafeQueue_959_ = lean_ctor_get(v___x_942_, 12);
v_failedRapps_960_ = lean_ctor_get(v___x_942_, 13);
v_isSharedCheck_968_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_968_ == 0)
{
lean_object* v_unused_969_; 
v_unused_969_ = lean_ctor_get(v___x_942_, 2);
lean_dec(v_unused_969_);
v___x_962_ = v___x_942_;
v_isShared_963_ = v_isSharedCheck_968_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_failedRapps_960_);
lean_inc(v_unsafeQueue_959_);
lean_inc(v_lastExpandedInIteration_957_);
lean_inc(v_addedInIteration_956_);
lean_inc(v_forwardRuleMatches_954_);
lean_inc(v_forwardState_953_);
lean_inc(v_mvars_952_);
lean_inc(v_normalizationState_951_);
lean_inc(v_preNormGoal_950_);
lean_inc(v_depth_946_);
lean_inc(v_origin_945_);
lean_inc(v_parent_944_);
lean_inc(v_id_943_);
lean_dec(v___x_942_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_968_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_965_; 
if (v_isShared_963_ == 0)
{
lean_ctor_set(v___x_962_, 2, v_children_937_);
v___x_965_ = v___x_962_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_967_; 
v_reuseFailAlloc_967_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_967_, 0, v_id_943_);
lean_ctor_set(v_reuseFailAlloc_967_, 1, v_parent_944_);
lean_ctor_set(v_reuseFailAlloc_967_, 2, v_children_937_);
lean_ctor_set(v_reuseFailAlloc_967_, 3, v_origin_945_);
lean_ctor_set(v_reuseFailAlloc_967_, 4, v_depth_946_);
lean_ctor_set(v_reuseFailAlloc_967_, 5, v_preNormGoal_950_);
lean_ctor_set(v_reuseFailAlloc_967_, 6, v_normalizationState_951_);
lean_ctor_set(v_reuseFailAlloc_967_, 7, v_mvars_952_);
lean_ctor_set(v_reuseFailAlloc_967_, 8, v_forwardState_953_);
lean_ctor_set(v_reuseFailAlloc_967_, 9, v_forwardRuleMatches_954_);
lean_ctor_set(v_reuseFailAlloc_967_, 10, v_addedInIteration_956_);
lean_ctor_set(v_reuseFailAlloc_967_, 11, v_lastExpandedInIteration_957_);
lean_ctor_set(v_reuseFailAlloc_967_, 12, v_unsafeQueue_959_);
lean_ctor_set(v_reuseFailAlloc_967_, 13, v_failedRapps_960_);
lean_ctor_set_uint8(v_reuseFailAlloc_967_, sizeof(void*)*14 + 8, v_state_947_);
lean_ctor_set_uint8(v_reuseFailAlloc_967_, sizeof(void*)*14 + 9, v_isIrrelevant_948_);
lean_ctor_set_uint8(v_reuseFailAlloc_967_, sizeof(void*)*14 + 10, v_isForcedUnprovable_949_);
lean_ctor_set_float(v_reuseFailAlloc_967_, sizeof(void*)*14, v_successProbability_955_);
lean_ctor_set_uint8(v_reuseFailAlloc_967_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_958_);
v___x_965_ = v_reuseFailAlloc_967_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
lean_object* v___x_966_; 
lean_inc(v_introGoal_940_);
v___x_966_ = lean_apply_1(v_introGoal_940_, v___x_965_);
return v___x_966_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setOrigin(lean_object* v_origin_970_, lean_object* v_g_971_){
_start:
{
lean_object* v___x_972_; lean_object* v_introGoal_973_; lean_object* v_elimGoal_974_; lean_object* v___x_975_; lean_object* v_id_976_; lean_object* v_parent_977_; lean_object* v_children_978_; lean_object* v_depth_979_; uint8_t v_state_980_; uint8_t v_isIrrelevant_981_; uint8_t v_isForcedUnprovable_982_; lean_object* v_preNormGoal_983_; lean_object* v_normalizationState_984_; lean_object* v_mvars_985_; lean_object* v_forwardState_986_; lean_object* v_forwardRuleMatches_987_; double v_successProbability_988_; lean_object* v_addedInIteration_989_; lean_object* v_lastExpandedInIteration_990_; uint8_t v_unsafeRulesSelected_991_; lean_object* v_unsafeQueue_992_; lean_object* v_failedRapps_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1001_; 
v___x_972_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_973_ = lean_ctor_get(v___x_972_, 0);
v_elimGoal_974_ = lean_ctor_get(v___x_972_, 1);
lean_inc_ref(v_elimGoal_974_);
v___x_975_ = lean_apply_1(v_elimGoal_974_, v_g_971_);
v_id_976_ = lean_ctor_get(v___x_975_, 0);
v_parent_977_ = lean_ctor_get(v___x_975_, 1);
v_children_978_ = lean_ctor_get(v___x_975_, 2);
v_depth_979_ = lean_ctor_get(v___x_975_, 4);
v_state_980_ = lean_ctor_get_uint8(v___x_975_, sizeof(void*)*14 + 8);
v_isIrrelevant_981_ = lean_ctor_get_uint8(v___x_975_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_982_ = lean_ctor_get_uint8(v___x_975_, sizeof(void*)*14 + 10);
v_preNormGoal_983_ = lean_ctor_get(v___x_975_, 5);
v_normalizationState_984_ = lean_ctor_get(v___x_975_, 6);
v_mvars_985_ = lean_ctor_get(v___x_975_, 7);
v_forwardState_986_ = lean_ctor_get(v___x_975_, 8);
v_forwardRuleMatches_987_ = lean_ctor_get(v___x_975_, 9);
v_successProbability_988_ = lean_ctor_get_float(v___x_975_, sizeof(void*)*14);
v_addedInIteration_989_ = lean_ctor_get(v___x_975_, 10);
v_lastExpandedInIteration_990_ = lean_ctor_get(v___x_975_, 11);
v_unsafeRulesSelected_991_ = lean_ctor_get_uint8(v___x_975_, sizeof(void*)*14 + 11);
v_unsafeQueue_992_ = lean_ctor_get(v___x_975_, 12);
v_failedRapps_993_ = lean_ctor_get(v___x_975_, 13);
v_isSharedCheck_1001_ = !lean_is_exclusive(v___x_975_);
if (v_isSharedCheck_1001_ == 0)
{
lean_object* v_unused_1002_; 
v_unused_1002_ = lean_ctor_get(v___x_975_, 3);
lean_dec(v_unused_1002_);
v___x_995_ = v___x_975_;
v_isShared_996_ = v_isSharedCheck_1001_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_failedRapps_993_);
lean_inc(v_unsafeQueue_992_);
lean_inc(v_lastExpandedInIteration_990_);
lean_inc(v_addedInIteration_989_);
lean_inc(v_forwardRuleMatches_987_);
lean_inc(v_forwardState_986_);
lean_inc(v_mvars_985_);
lean_inc(v_normalizationState_984_);
lean_inc(v_preNormGoal_983_);
lean_inc(v_depth_979_);
lean_inc(v_children_978_);
lean_inc(v_parent_977_);
lean_inc(v_id_976_);
lean_dec(v___x_975_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1001_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v___x_998_; 
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 3, v_origin_970_);
v___x_998_ = v___x_995_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v_id_976_);
lean_ctor_set(v_reuseFailAlloc_1000_, 1, v_parent_977_);
lean_ctor_set(v_reuseFailAlloc_1000_, 2, v_children_978_);
lean_ctor_set(v_reuseFailAlloc_1000_, 3, v_origin_970_);
lean_ctor_set(v_reuseFailAlloc_1000_, 4, v_depth_979_);
lean_ctor_set(v_reuseFailAlloc_1000_, 5, v_preNormGoal_983_);
lean_ctor_set(v_reuseFailAlloc_1000_, 6, v_normalizationState_984_);
lean_ctor_set(v_reuseFailAlloc_1000_, 7, v_mvars_985_);
lean_ctor_set(v_reuseFailAlloc_1000_, 8, v_forwardState_986_);
lean_ctor_set(v_reuseFailAlloc_1000_, 9, v_forwardRuleMatches_987_);
lean_ctor_set(v_reuseFailAlloc_1000_, 10, v_addedInIteration_989_);
lean_ctor_set(v_reuseFailAlloc_1000_, 11, v_lastExpandedInIteration_990_);
lean_ctor_set(v_reuseFailAlloc_1000_, 12, v_unsafeQueue_992_);
lean_ctor_set(v_reuseFailAlloc_1000_, 13, v_failedRapps_993_);
lean_ctor_set_uint8(v_reuseFailAlloc_1000_, sizeof(void*)*14 + 8, v_state_980_);
lean_ctor_set_uint8(v_reuseFailAlloc_1000_, sizeof(void*)*14 + 9, v_isIrrelevant_981_);
lean_ctor_set_uint8(v_reuseFailAlloc_1000_, sizeof(void*)*14 + 10, v_isForcedUnprovable_982_);
lean_ctor_set_float(v_reuseFailAlloc_1000_, sizeof(void*)*14, v_successProbability_988_);
lean_ctor_set_uint8(v_reuseFailAlloc_1000_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_991_);
v___x_998_ = v_reuseFailAlloc_1000_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
lean_object* v___x_999_; 
lean_inc(v_introGoal_973_);
v___x_999_ = lean_apply_1(v_introGoal_973_, v___x_998_);
return v___x_999_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setDepth(lean_object* v_depth_1003_, lean_object* v_g_1004_){
_start:
{
lean_object* v___x_1005_; lean_object* v_introGoal_1006_; lean_object* v_elimGoal_1007_; lean_object* v___x_1008_; lean_object* v_id_1009_; lean_object* v_parent_1010_; lean_object* v_children_1011_; lean_object* v_origin_1012_; uint8_t v_state_1013_; uint8_t v_isIrrelevant_1014_; uint8_t v_isForcedUnprovable_1015_; lean_object* v_preNormGoal_1016_; lean_object* v_normalizationState_1017_; lean_object* v_mvars_1018_; lean_object* v_forwardState_1019_; lean_object* v_forwardRuleMatches_1020_; double v_successProbability_1021_; lean_object* v_addedInIteration_1022_; lean_object* v_lastExpandedInIteration_1023_; uint8_t v_unsafeRulesSelected_1024_; lean_object* v_unsafeQueue_1025_; lean_object* v_failedRapps_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1034_; 
v___x_1005_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1006_ = lean_ctor_get(v___x_1005_, 0);
v_elimGoal_1007_ = lean_ctor_get(v___x_1005_, 1);
lean_inc_ref(v_elimGoal_1007_);
v___x_1008_ = lean_apply_1(v_elimGoal_1007_, v_g_1004_);
v_id_1009_ = lean_ctor_get(v___x_1008_, 0);
v_parent_1010_ = lean_ctor_get(v___x_1008_, 1);
v_children_1011_ = lean_ctor_get(v___x_1008_, 2);
v_origin_1012_ = lean_ctor_get(v___x_1008_, 3);
v_state_1013_ = lean_ctor_get_uint8(v___x_1008_, sizeof(void*)*14 + 8);
v_isIrrelevant_1014_ = lean_ctor_get_uint8(v___x_1008_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1015_ = lean_ctor_get_uint8(v___x_1008_, sizeof(void*)*14 + 10);
v_preNormGoal_1016_ = lean_ctor_get(v___x_1008_, 5);
v_normalizationState_1017_ = lean_ctor_get(v___x_1008_, 6);
v_mvars_1018_ = lean_ctor_get(v___x_1008_, 7);
v_forwardState_1019_ = lean_ctor_get(v___x_1008_, 8);
v_forwardRuleMatches_1020_ = lean_ctor_get(v___x_1008_, 9);
v_successProbability_1021_ = lean_ctor_get_float(v___x_1008_, sizeof(void*)*14);
v_addedInIteration_1022_ = lean_ctor_get(v___x_1008_, 10);
v_lastExpandedInIteration_1023_ = lean_ctor_get(v___x_1008_, 11);
v_unsafeRulesSelected_1024_ = lean_ctor_get_uint8(v___x_1008_, sizeof(void*)*14 + 11);
v_unsafeQueue_1025_ = lean_ctor_get(v___x_1008_, 12);
v_failedRapps_1026_ = lean_ctor_get(v___x_1008_, 13);
v_isSharedCheck_1034_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1034_ == 0)
{
lean_object* v_unused_1035_; 
v_unused_1035_ = lean_ctor_get(v___x_1008_, 4);
lean_dec(v_unused_1035_);
v___x_1028_ = v___x_1008_;
v_isShared_1029_ = v_isSharedCheck_1034_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_failedRapps_1026_);
lean_inc(v_unsafeQueue_1025_);
lean_inc(v_lastExpandedInIteration_1023_);
lean_inc(v_addedInIteration_1022_);
lean_inc(v_forwardRuleMatches_1020_);
lean_inc(v_forwardState_1019_);
lean_inc(v_mvars_1018_);
lean_inc(v_normalizationState_1017_);
lean_inc(v_preNormGoal_1016_);
lean_inc(v_origin_1012_);
lean_inc(v_children_1011_);
lean_inc(v_parent_1010_);
lean_inc(v_id_1009_);
lean_dec(v___x_1008_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1034_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___x_1031_; 
if (v_isShared_1029_ == 0)
{
lean_ctor_set(v___x_1028_, 4, v_depth_1003_);
v___x_1031_ = v___x_1028_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1033_; 
v_reuseFailAlloc_1033_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1033_, 0, v_id_1009_);
lean_ctor_set(v_reuseFailAlloc_1033_, 1, v_parent_1010_);
lean_ctor_set(v_reuseFailAlloc_1033_, 2, v_children_1011_);
lean_ctor_set(v_reuseFailAlloc_1033_, 3, v_origin_1012_);
lean_ctor_set(v_reuseFailAlloc_1033_, 4, v_depth_1003_);
lean_ctor_set(v_reuseFailAlloc_1033_, 5, v_preNormGoal_1016_);
lean_ctor_set(v_reuseFailAlloc_1033_, 6, v_normalizationState_1017_);
lean_ctor_set(v_reuseFailAlloc_1033_, 7, v_mvars_1018_);
lean_ctor_set(v_reuseFailAlloc_1033_, 8, v_forwardState_1019_);
lean_ctor_set(v_reuseFailAlloc_1033_, 9, v_forwardRuleMatches_1020_);
lean_ctor_set(v_reuseFailAlloc_1033_, 10, v_addedInIteration_1022_);
lean_ctor_set(v_reuseFailAlloc_1033_, 11, v_lastExpandedInIteration_1023_);
lean_ctor_set(v_reuseFailAlloc_1033_, 12, v_unsafeQueue_1025_);
lean_ctor_set(v_reuseFailAlloc_1033_, 13, v_failedRapps_1026_);
lean_ctor_set_uint8(v_reuseFailAlloc_1033_, sizeof(void*)*14 + 8, v_state_1013_);
lean_ctor_set_uint8(v_reuseFailAlloc_1033_, sizeof(void*)*14 + 9, v_isIrrelevant_1014_);
lean_ctor_set_uint8(v_reuseFailAlloc_1033_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1015_);
lean_ctor_set_float(v_reuseFailAlloc_1033_, sizeof(void*)*14, v_successProbability_1021_);
lean_ctor_set_uint8(v_reuseFailAlloc_1033_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1024_);
v___x_1031_ = v_reuseFailAlloc_1033_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
lean_object* v___x_1032_; 
lean_inc(v_introGoal_1006_);
v___x_1032_ = lean_apply_1(v_introGoal_1006_, v___x_1031_);
return v___x_1032_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant(uint8_t v_isIrrelevant_1036_, lean_object* v_g_1037_){
_start:
{
lean_object* v___x_1038_; lean_object* v_introGoal_1039_; lean_object* v_elimGoal_1040_; lean_object* v___x_1041_; lean_object* v_id_1042_; lean_object* v_parent_1043_; lean_object* v_children_1044_; lean_object* v_origin_1045_; lean_object* v_depth_1046_; uint8_t v_state_1047_; uint8_t v_isForcedUnprovable_1048_; lean_object* v_preNormGoal_1049_; lean_object* v_normalizationState_1050_; lean_object* v_mvars_1051_; lean_object* v_forwardState_1052_; lean_object* v_forwardRuleMatches_1053_; double v_successProbability_1054_; lean_object* v_addedInIteration_1055_; lean_object* v_lastExpandedInIteration_1056_; uint8_t v_unsafeRulesSelected_1057_; lean_object* v_unsafeQueue_1058_; lean_object* v_failedRapps_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1067_; 
v___x_1038_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1039_ = lean_ctor_get(v___x_1038_, 0);
v_elimGoal_1040_ = lean_ctor_get(v___x_1038_, 1);
lean_inc_ref(v_elimGoal_1040_);
v___x_1041_ = lean_apply_1(v_elimGoal_1040_, v_g_1037_);
v_id_1042_ = lean_ctor_get(v___x_1041_, 0);
v_parent_1043_ = lean_ctor_get(v___x_1041_, 1);
v_children_1044_ = lean_ctor_get(v___x_1041_, 2);
v_origin_1045_ = lean_ctor_get(v___x_1041_, 3);
v_depth_1046_ = lean_ctor_get(v___x_1041_, 4);
v_state_1047_ = lean_ctor_get_uint8(v___x_1041_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_1048_ = lean_ctor_get_uint8(v___x_1041_, sizeof(void*)*14 + 10);
v_preNormGoal_1049_ = lean_ctor_get(v___x_1041_, 5);
v_normalizationState_1050_ = lean_ctor_get(v___x_1041_, 6);
v_mvars_1051_ = lean_ctor_get(v___x_1041_, 7);
v_forwardState_1052_ = lean_ctor_get(v___x_1041_, 8);
v_forwardRuleMatches_1053_ = lean_ctor_get(v___x_1041_, 9);
v_successProbability_1054_ = lean_ctor_get_float(v___x_1041_, sizeof(void*)*14);
v_addedInIteration_1055_ = lean_ctor_get(v___x_1041_, 10);
v_lastExpandedInIteration_1056_ = lean_ctor_get(v___x_1041_, 11);
v_unsafeRulesSelected_1057_ = lean_ctor_get_uint8(v___x_1041_, sizeof(void*)*14 + 11);
v_unsafeQueue_1058_ = lean_ctor_get(v___x_1041_, 12);
v_failedRapps_1059_ = lean_ctor_get(v___x_1041_, 13);
v_isSharedCheck_1067_ = !lean_is_exclusive(v___x_1041_);
if (v_isSharedCheck_1067_ == 0)
{
v___x_1061_ = v___x_1041_;
v_isShared_1062_ = v_isSharedCheck_1067_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_failedRapps_1059_);
lean_inc(v_unsafeQueue_1058_);
lean_inc(v_lastExpandedInIteration_1056_);
lean_inc(v_addedInIteration_1055_);
lean_inc(v_forwardRuleMatches_1053_);
lean_inc(v_forwardState_1052_);
lean_inc(v_mvars_1051_);
lean_inc(v_normalizationState_1050_);
lean_inc(v_preNormGoal_1049_);
lean_inc(v_depth_1046_);
lean_inc(v_origin_1045_);
lean_inc(v_children_1044_);
lean_inc(v_parent_1043_);
lean_inc(v_id_1042_);
lean_dec(v___x_1041_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1067_;
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
lean_object* v_reuseFailAlloc_1066_; 
v_reuseFailAlloc_1066_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1066_, 0, v_id_1042_);
lean_ctor_set(v_reuseFailAlloc_1066_, 1, v_parent_1043_);
lean_ctor_set(v_reuseFailAlloc_1066_, 2, v_children_1044_);
lean_ctor_set(v_reuseFailAlloc_1066_, 3, v_origin_1045_);
lean_ctor_set(v_reuseFailAlloc_1066_, 4, v_depth_1046_);
lean_ctor_set(v_reuseFailAlloc_1066_, 5, v_preNormGoal_1049_);
lean_ctor_set(v_reuseFailAlloc_1066_, 6, v_normalizationState_1050_);
lean_ctor_set(v_reuseFailAlloc_1066_, 7, v_mvars_1051_);
lean_ctor_set(v_reuseFailAlloc_1066_, 8, v_forwardState_1052_);
lean_ctor_set(v_reuseFailAlloc_1066_, 9, v_forwardRuleMatches_1053_);
lean_ctor_set(v_reuseFailAlloc_1066_, 10, v_addedInIteration_1055_);
lean_ctor_set(v_reuseFailAlloc_1066_, 11, v_lastExpandedInIteration_1056_);
lean_ctor_set(v_reuseFailAlloc_1066_, 12, v_unsafeQueue_1058_);
lean_ctor_set(v_reuseFailAlloc_1066_, 13, v_failedRapps_1059_);
lean_ctor_set_uint8(v_reuseFailAlloc_1066_, sizeof(void*)*14 + 8, v_state_1047_);
lean_ctor_set_uint8(v_reuseFailAlloc_1066_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1048_);
lean_ctor_set_float(v_reuseFailAlloc_1066_, sizeof(void*)*14, v_successProbability_1054_);
lean_ctor_set_uint8(v_reuseFailAlloc_1066_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1057_);
v___x_1064_ = v_reuseFailAlloc_1066_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
lean_object* v___x_1065_; 
lean_ctor_set_uint8(v___x_1064_, sizeof(void*)*14 + 9, v_isIrrelevant_1036_);
lean_inc(v_introGoal_1039_);
v___x_1065_ = lean_apply_1(v_introGoal_1039_, v___x_1064_);
return v___x_1065_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsIrrelevant___boxed(lean_object* v_isIrrelevant_1068_, lean_object* v_g_1069_){
_start:
{
uint8_t v_isIrrelevant_boxed_1070_; lean_object* v_res_1071_; 
v_isIrrelevant_boxed_1070_ = lean_unbox(v_isIrrelevant_1068_);
v_res_1071_ = lp_aesop_Aesop_Goal_setIsIrrelevant(v_isIrrelevant_boxed_1070_, v_g_1069_);
return v_res_1071_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable(uint8_t v_isForcedUnprovable_1072_, lean_object* v_g_1073_){
_start:
{
lean_object* v___x_1074_; lean_object* v_introGoal_1075_; lean_object* v_elimGoal_1076_; lean_object* v___x_1077_; lean_object* v_id_1078_; lean_object* v_parent_1079_; lean_object* v_children_1080_; lean_object* v_origin_1081_; lean_object* v_depth_1082_; uint8_t v_state_1083_; uint8_t v_isIrrelevant_1084_; lean_object* v_preNormGoal_1085_; lean_object* v_normalizationState_1086_; lean_object* v_mvars_1087_; lean_object* v_forwardState_1088_; lean_object* v_forwardRuleMatches_1089_; double v_successProbability_1090_; lean_object* v_addedInIteration_1091_; lean_object* v_lastExpandedInIteration_1092_; uint8_t v_unsafeRulesSelected_1093_; lean_object* v_unsafeQueue_1094_; lean_object* v_failedRapps_1095_; lean_object* v___x_1097_; uint8_t v_isShared_1098_; uint8_t v_isSharedCheck_1103_; 
v___x_1074_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1075_ = lean_ctor_get(v___x_1074_, 0);
v_elimGoal_1076_ = lean_ctor_get(v___x_1074_, 1);
lean_inc_ref(v_elimGoal_1076_);
v___x_1077_ = lean_apply_1(v_elimGoal_1076_, v_g_1073_);
v_id_1078_ = lean_ctor_get(v___x_1077_, 0);
v_parent_1079_ = lean_ctor_get(v___x_1077_, 1);
v_children_1080_ = lean_ctor_get(v___x_1077_, 2);
v_origin_1081_ = lean_ctor_get(v___x_1077_, 3);
v_depth_1082_ = lean_ctor_get(v___x_1077_, 4);
v_state_1083_ = lean_ctor_get_uint8(v___x_1077_, sizeof(void*)*14 + 8);
v_isIrrelevant_1084_ = lean_ctor_get_uint8(v___x_1077_, sizeof(void*)*14 + 9);
v_preNormGoal_1085_ = lean_ctor_get(v___x_1077_, 5);
v_normalizationState_1086_ = lean_ctor_get(v___x_1077_, 6);
v_mvars_1087_ = lean_ctor_get(v___x_1077_, 7);
v_forwardState_1088_ = lean_ctor_get(v___x_1077_, 8);
v_forwardRuleMatches_1089_ = lean_ctor_get(v___x_1077_, 9);
v_successProbability_1090_ = lean_ctor_get_float(v___x_1077_, sizeof(void*)*14);
v_addedInIteration_1091_ = lean_ctor_get(v___x_1077_, 10);
v_lastExpandedInIteration_1092_ = lean_ctor_get(v___x_1077_, 11);
v_unsafeRulesSelected_1093_ = lean_ctor_get_uint8(v___x_1077_, sizeof(void*)*14 + 11);
v_unsafeQueue_1094_ = lean_ctor_get(v___x_1077_, 12);
v_failedRapps_1095_ = lean_ctor_get(v___x_1077_, 13);
v_isSharedCheck_1103_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1103_ == 0)
{
v___x_1097_ = v___x_1077_;
v_isShared_1098_ = v_isSharedCheck_1103_;
goto v_resetjp_1096_;
}
else
{
lean_inc(v_failedRapps_1095_);
lean_inc(v_unsafeQueue_1094_);
lean_inc(v_lastExpandedInIteration_1092_);
lean_inc(v_addedInIteration_1091_);
lean_inc(v_forwardRuleMatches_1089_);
lean_inc(v_forwardState_1088_);
lean_inc(v_mvars_1087_);
lean_inc(v_normalizationState_1086_);
lean_inc(v_preNormGoal_1085_);
lean_inc(v_depth_1082_);
lean_inc(v_origin_1081_);
lean_inc(v_children_1080_);
lean_inc(v_parent_1079_);
lean_inc(v_id_1078_);
lean_dec(v___x_1077_);
v___x_1097_ = lean_box(0);
v_isShared_1098_ = v_isSharedCheck_1103_;
goto v_resetjp_1096_;
}
v_resetjp_1096_:
{
lean_object* v___x_1100_; 
if (v_isShared_1098_ == 0)
{
v___x_1100_ = v___x_1097_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v_id_1078_);
lean_ctor_set(v_reuseFailAlloc_1102_, 1, v_parent_1079_);
lean_ctor_set(v_reuseFailAlloc_1102_, 2, v_children_1080_);
lean_ctor_set(v_reuseFailAlloc_1102_, 3, v_origin_1081_);
lean_ctor_set(v_reuseFailAlloc_1102_, 4, v_depth_1082_);
lean_ctor_set(v_reuseFailAlloc_1102_, 5, v_preNormGoal_1085_);
lean_ctor_set(v_reuseFailAlloc_1102_, 6, v_normalizationState_1086_);
lean_ctor_set(v_reuseFailAlloc_1102_, 7, v_mvars_1087_);
lean_ctor_set(v_reuseFailAlloc_1102_, 8, v_forwardState_1088_);
lean_ctor_set(v_reuseFailAlloc_1102_, 9, v_forwardRuleMatches_1089_);
lean_ctor_set(v_reuseFailAlloc_1102_, 10, v_addedInIteration_1091_);
lean_ctor_set(v_reuseFailAlloc_1102_, 11, v_lastExpandedInIteration_1092_);
lean_ctor_set(v_reuseFailAlloc_1102_, 12, v_unsafeQueue_1094_);
lean_ctor_set(v_reuseFailAlloc_1102_, 13, v_failedRapps_1095_);
lean_ctor_set_uint8(v_reuseFailAlloc_1102_, sizeof(void*)*14 + 8, v_state_1083_);
lean_ctor_set_uint8(v_reuseFailAlloc_1102_, sizeof(void*)*14 + 9, v_isIrrelevant_1084_);
lean_ctor_set_float(v_reuseFailAlloc_1102_, sizeof(void*)*14, v_successProbability_1090_);
lean_ctor_set_uint8(v_reuseFailAlloc_1102_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1093_);
v___x_1100_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
lean_object* v___x_1101_; 
lean_ctor_set_uint8(v___x_1100_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1072_);
lean_inc(v_introGoal_1075_);
v___x_1101_ = lean_apply_1(v_introGoal_1075_, v___x_1100_);
return v___x_1101_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setIsForcedUnprovable___boxed(lean_object* v_isForcedUnprovable_1104_, lean_object* v_g_1105_){
_start:
{
uint8_t v_isForcedUnprovable_boxed_1106_; lean_object* v_res_1107_; 
v_isForcedUnprovable_boxed_1106_ = lean_unbox(v_isForcedUnprovable_1104_);
v_res_1107_ = lp_aesop_Aesop_Goal_setIsForcedUnprovable(v_isForcedUnprovable_boxed_1106_, v_g_1105_);
return v_res_1107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setPreNormGoal(lean_object* v_preNormGoal_1108_, lean_object* v_g_1109_){
_start:
{
lean_object* v___x_1110_; lean_object* v_introGoal_1111_; lean_object* v_elimGoal_1112_; lean_object* v___x_1113_; lean_object* v_id_1114_; lean_object* v_parent_1115_; lean_object* v_children_1116_; lean_object* v_origin_1117_; lean_object* v_depth_1118_; uint8_t v_state_1119_; uint8_t v_isIrrelevant_1120_; uint8_t v_isForcedUnprovable_1121_; lean_object* v_normalizationState_1122_; lean_object* v_mvars_1123_; lean_object* v_forwardState_1124_; lean_object* v_forwardRuleMatches_1125_; double v_successProbability_1126_; lean_object* v_addedInIteration_1127_; lean_object* v_lastExpandedInIteration_1128_; uint8_t v_unsafeRulesSelected_1129_; lean_object* v_unsafeQueue_1130_; lean_object* v_failedRapps_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1139_; 
v___x_1110_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1111_ = lean_ctor_get(v___x_1110_, 0);
v_elimGoal_1112_ = lean_ctor_get(v___x_1110_, 1);
lean_inc_ref(v_elimGoal_1112_);
v___x_1113_ = lean_apply_1(v_elimGoal_1112_, v_g_1109_);
v_id_1114_ = lean_ctor_get(v___x_1113_, 0);
v_parent_1115_ = lean_ctor_get(v___x_1113_, 1);
v_children_1116_ = lean_ctor_get(v___x_1113_, 2);
v_origin_1117_ = lean_ctor_get(v___x_1113_, 3);
v_depth_1118_ = lean_ctor_get(v___x_1113_, 4);
v_state_1119_ = lean_ctor_get_uint8(v___x_1113_, sizeof(void*)*14 + 8);
v_isIrrelevant_1120_ = lean_ctor_get_uint8(v___x_1113_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1121_ = lean_ctor_get_uint8(v___x_1113_, sizeof(void*)*14 + 10);
v_normalizationState_1122_ = lean_ctor_get(v___x_1113_, 6);
v_mvars_1123_ = lean_ctor_get(v___x_1113_, 7);
v_forwardState_1124_ = lean_ctor_get(v___x_1113_, 8);
v_forwardRuleMatches_1125_ = lean_ctor_get(v___x_1113_, 9);
v_successProbability_1126_ = lean_ctor_get_float(v___x_1113_, sizeof(void*)*14);
v_addedInIteration_1127_ = lean_ctor_get(v___x_1113_, 10);
v_lastExpandedInIteration_1128_ = lean_ctor_get(v___x_1113_, 11);
v_unsafeRulesSelected_1129_ = lean_ctor_get_uint8(v___x_1113_, sizeof(void*)*14 + 11);
v_unsafeQueue_1130_ = lean_ctor_get(v___x_1113_, 12);
v_failedRapps_1131_ = lean_ctor_get(v___x_1113_, 13);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1139_ == 0)
{
lean_object* v_unused_1140_; 
v_unused_1140_ = lean_ctor_get(v___x_1113_, 5);
lean_dec(v_unused_1140_);
v___x_1133_ = v___x_1113_;
v_isShared_1134_ = v_isSharedCheck_1139_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_failedRapps_1131_);
lean_inc(v_unsafeQueue_1130_);
lean_inc(v_lastExpandedInIteration_1128_);
lean_inc(v_addedInIteration_1127_);
lean_inc(v_forwardRuleMatches_1125_);
lean_inc(v_forwardState_1124_);
lean_inc(v_mvars_1123_);
lean_inc(v_normalizationState_1122_);
lean_inc(v_depth_1118_);
lean_inc(v_origin_1117_);
lean_inc(v_children_1116_);
lean_inc(v_parent_1115_);
lean_inc(v_id_1114_);
lean_dec(v___x_1113_);
v___x_1133_ = lean_box(0);
v_isShared_1134_ = v_isSharedCheck_1139_;
goto v_resetjp_1132_;
}
v_resetjp_1132_:
{
lean_object* v___x_1136_; 
if (v_isShared_1134_ == 0)
{
lean_ctor_set(v___x_1133_, 5, v_preNormGoal_1108_);
v___x_1136_ = v___x_1133_;
goto v_reusejp_1135_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_id_1114_);
lean_ctor_set(v_reuseFailAlloc_1138_, 1, v_parent_1115_);
lean_ctor_set(v_reuseFailAlloc_1138_, 2, v_children_1116_);
lean_ctor_set(v_reuseFailAlloc_1138_, 3, v_origin_1117_);
lean_ctor_set(v_reuseFailAlloc_1138_, 4, v_depth_1118_);
lean_ctor_set(v_reuseFailAlloc_1138_, 5, v_preNormGoal_1108_);
lean_ctor_set(v_reuseFailAlloc_1138_, 6, v_normalizationState_1122_);
lean_ctor_set(v_reuseFailAlloc_1138_, 7, v_mvars_1123_);
lean_ctor_set(v_reuseFailAlloc_1138_, 8, v_forwardState_1124_);
lean_ctor_set(v_reuseFailAlloc_1138_, 9, v_forwardRuleMatches_1125_);
lean_ctor_set(v_reuseFailAlloc_1138_, 10, v_addedInIteration_1127_);
lean_ctor_set(v_reuseFailAlloc_1138_, 11, v_lastExpandedInIteration_1128_);
lean_ctor_set(v_reuseFailAlloc_1138_, 12, v_unsafeQueue_1130_);
lean_ctor_set(v_reuseFailAlloc_1138_, 13, v_failedRapps_1131_);
lean_ctor_set_uint8(v_reuseFailAlloc_1138_, sizeof(void*)*14 + 8, v_state_1119_);
lean_ctor_set_uint8(v_reuseFailAlloc_1138_, sizeof(void*)*14 + 9, v_isIrrelevant_1120_);
lean_ctor_set_uint8(v_reuseFailAlloc_1138_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1121_);
lean_ctor_set_float(v_reuseFailAlloc_1138_, sizeof(void*)*14, v_successProbability_1126_);
lean_ctor_set_uint8(v_reuseFailAlloc_1138_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1129_);
v___x_1136_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1135_;
}
v_reusejp_1135_:
{
lean_object* v___x_1137_; 
lean_inc(v_introGoal_1111_);
v___x_1137_ = lean_apply_1(v_introGoal_1111_, v___x_1136_);
return v___x_1137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setNormalizationState(lean_object* v_normalizationState_1141_, lean_object* v_g_1142_){
_start:
{
lean_object* v___x_1143_; lean_object* v_introGoal_1144_; lean_object* v_elimGoal_1145_; lean_object* v___x_1146_; lean_object* v_id_1147_; lean_object* v_parent_1148_; lean_object* v_children_1149_; lean_object* v_origin_1150_; lean_object* v_depth_1151_; uint8_t v_state_1152_; uint8_t v_isIrrelevant_1153_; uint8_t v_isForcedUnprovable_1154_; lean_object* v_preNormGoal_1155_; lean_object* v_mvars_1156_; lean_object* v_forwardState_1157_; lean_object* v_forwardRuleMatches_1158_; double v_successProbability_1159_; lean_object* v_addedInIteration_1160_; lean_object* v_lastExpandedInIteration_1161_; uint8_t v_unsafeRulesSelected_1162_; lean_object* v_unsafeQueue_1163_; lean_object* v_failedRapps_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1172_; 
v___x_1143_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1144_ = lean_ctor_get(v___x_1143_, 0);
v_elimGoal_1145_ = lean_ctor_get(v___x_1143_, 1);
lean_inc_ref(v_elimGoal_1145_);
v___x_1146_ = lean_apply_1(v_elimGoal_1145_, v_g_1142_);
v_id_1147_ = lean_ctor_get(v___x_1146_, 0);
v_parent_1148_ = lean_ctor_get(v___x_1146_, 1);
v_children_1149_ = lean_ctor_get(v___x_1146_, 2);
v_origin_1150_ = lean_ctor_get(v___x_1146_, 3);
v_depth_1151_ = lean_ctor_get(v___x_1146_, 4);
v_state_1152_ = lean_ctor_get_uint8(v___x_1146_, sizeof(void*)*14 + 8);
v_isIrrelevant_1153_ = lean_ctor_get_uint8(v___x_1146_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1154_ = lean_ctor_get_uint8(v___x_1146_, sizeof(void*)*14 + 10);
v_preNormGoal_1155_ = lean_ctor_get(v___x_1146_, 5);
v_mvars_1156_ = lean_ctor_get(v___x_1146_, 7);
v_forwardState_1157_ = lean_ctor_get(v___x_1146_, 8);
v_forwardRuleMatches_1158_ = lean_ctor_get(v___x_1146_, 9);
v_successProbability_1159_ = lean_ctor_get_float(v___x_1146_, sizeof(void*)*14);
v_addedInIteration_1160_ = lean_ctor_get(v___x_1146_, 10);
v_lastExpandedInIteration_1161_ = lean_ctor_get(v___x_1146_, 11);
v_unsafeRulesSelected_1162_ = lean_ctor_get_uint8(v___x_1146_, sizeof(void*)*14 + 11);
v_unsafeQueue_1163_ = lean_ctor_get(v___x_1146_, 12);
v_failedRapps_1164_ = lean_ctor_get(v___x_1146_, 13);
v_isSharedCheck_1172_ = !lean_is_exclusive(v___x_1146_);
if (v_isSharedCheck_1172_ == 0)
{
lean_object* v_unused_1173_; 
v_unused_1173_ = lean_ctor_get(v___x_1146_, 6);
lean_dec(v_unused_1173_);
v___x_1166_ = v___x_1146_;
v_isShared_1167_ = v_isSharedCheck_1172_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_failedRapps_1164_);
lean_inc(v_unsafeQueue_1163_);
lean_inc(v_lastExpandedInIteration_1161_);
lean_inc(v_addedInIteration_1160_);
lean_inc(v_forwardRuleMatches_1158_);
lean_inc(v_forwardState_1157_);
lean_inc(v_mvars_1156_);
lean_inc(v_preNormGoal_1155_);
lean_inc(v_depth_1151_);
lean_inc(v_origin_1150_);
lean_inc(v_children_1149_);
lean_inc(v_parent_1148_);
lean_inc(v_id_1147_);
lean_dec(v___x_1146_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1172_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1169_; 
if (v_isShared_1167_ == 0)
{
lean_ctor_set(v___x_1166_, 6, v_normalizationState_1141_);
v___x_1169_ = v___x_1166_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_id_1147_);
lean_ctor_set(v_reuseFailAlloc_1171_, 1, v_parent_1148_);
lean_ctor_set(v_reuseFailAlloc_1171_, 2, v_children_1149_);
lean_ctor_set(v_reuseFailAlloc_1171_, 3, v_origin_1150_);
lean_ctor_set(v_reuseFailAlloc_1171_, 4, v_depth_1151_);
lean_ctor_set(v_reuseFailAlloc_1171_, 5, v_preNormGoal_1155_);
lean_ctor_set(v_reuseFailAlloc_1171_, 6, v_normalizationState_1141_);
lean_ctor_set(v_reuseFailAlloc_1171_, 7, v_mvars_1156_);
lean_ctor_set(v_reuseFailAlloc_1171_, 8, v_forwardState_1157_);
lean_ctor_set(v_reuseFailAlloc_1171_, 9, v_forwardRuleMatches_1158_);
lean_ctor_set(v_reuseFailAlloc_1171_, 10, v_addedInIteration_1160_);
lean_ctor_set(v_reuseFailAlloc_1171_, 11, v_lastExpandedInIteration_1161_);
lean_ctor_set(v_reuseFailAlloc_1171_, 12, v_unsafeQueue_1163_);
lean_ctor_set(v_reuseFailAlloc_1171_, 13, v_failedRapps_1164_);
lean_ctor_set_uint8(v_reuseFailAlloc_1171_, sizeof(void*)*14 + 8, v_state_1152_);
lean_ctor_set_uint8(v_reuseFailAlloc_1171_, sizeof(void*)*14 + 9, v_isIrrelevant_1153_);
lean_ctor_set_uint8(v_reuseFailAlloc_1171_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1154_);
lean_ctor_set_float(v_reuseFailAlloc_1171_, sizeof(void*)*14, v_successProbability_1159_);
lean_ctor_set_uint8(v_reuseFailAlloc_1171_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1162_);
v___x_1169_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
lean_object* v___x_1170_; 
lean_inc(v_introGoal_1144_);
v___x_1170_ = lean_apply_1(v_introGoal_1144_, v___x_1169_);
return v___x_1170_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setMVars(lean_object* v_mvars_1174_, lean_object* v_g_1175_){
_start:
{
lean_object* v___x_1176_; lean_object* v_introGoal_1177_; lean_object* v_elimGoal_1178_; lean_object* v___x_1179_; lean_object* v_id_1180_; lean_object* v_parent_1181_; lean_object* v_children_1182_; lean_object* v_origin_1183_; lean_object* v_depth_1184_; uint8_t v_state_1185_; uint8_t v_isIrrelevant_1186_; uint8_t v_isForcedUnprovable_1187_; lean_object* v_preNormGoal_1188_; lean_object* v_normalizationState_1189_; lean_object* v_forwardState_1190_; lean_object* v_forwardRuleMatches_1191_; double v_successProbability_1192_; lean_object* v_addedInIteration_1193_; lean_object* v_lastExpandedInIteration_1194_; uint8_t v_unsafeRulesSelected_1195_; lean_object* v_unsafeQueue_1196_; lean_object* v_failedRapps_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1205_; 
v___x_1176_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1177_ = lean_ctor_get(v___x_1176_, 0);
v_elimGoal_1178_ = lean_ctor_get(v___x_1176_, 1);
lean_inc_ref(v_elimGoal_1178_);
v___x_1179_ = lean_apply_1(v_elimGoal_1178_, v_g_1175_);
v_id_1180_ = lean_ctor_get(v___x_1179_, 0);
v_parent_1181_ = lean_ctor_get(v___x_1179_, 1);
v_children_1182_ = lean_ctor_get(v___x_1179_, 2);
v_origin_1183_ = lean_ctor_get(v___x_1179_, 3);
v_depth_1184_ = lean_ctor_get(v___x_1179_, 4);
v_state_1185_ = lean_ctor_get_uint8(v___x_1179_, sizeof(void*)*14 + 8);
v_isIrrelevant_1186_ = lean_ctor_get_uint8(v___x_1179_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1187_ = lean_ctor_get_uint8(v___x_1179_, sizeof(void*)*14 + 10);
v_preNormGoal_1188_ = lean_ctor_get(v___x_1179_, 5);
v_normalizationState_1189_ = lean_ctor_get(v___x_1179_, 6);
v_forwardState_1190_ = lean_ctor_get(v___x_1179_, 8);
v_forwardRuleMatches_1191_ = lean_ctor_get(v___x_1179_, 9);
v_successProbability_1192_ = lean_ctor_get_float(v___x_1179_, sizeof(void*)*14);
v_addedInIteration_1193_ = lean_ctor_get(v___x_1179_, 10);
v_lastExpandedInIteration_1194_ = lean_ctor_get(v___x_1179_, 11);
v_unsafeRulesSelected_1195_ = lean_ctor_get_uint8(v___x_1179_, sizeof(void*)*14 + 11);
v_unsafeQueue_1196_ = lean_ctor_get(v___x_1179_, 12);
v_failedRapps_1197_ = lean_ctor_get(v___x_1179_, 13);
v_isSharedCheck_1205_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1205_ == 0)
{
lean_object* v_unused_1206_; 
v_unused_1206_ = lean_ctor_get(v___x_1179_, 7);
lean_dec(v_unused_1206_);
v___x_1199_ = v___x_1179_;
v_isShared_1200_ = v_isSharedCheck_1205_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_failedRapps_1197_);
lean_inc(v_unsafeQueue_1196_);
lean_inc(v_lastExpandedInIteration_1194_);
lean_inc(v_addedInIteration_1193_);
lean_inc(v_forwardRuleMatches_1191_);
lean_inc(v_forwardState_1190_);
lean_inc(v_normalizationState_1189_);
lean_inc(v_preNormGoal_1188_);
lean_inc(v_depth_1184_);
lean_inc(v_origin_1183_);
lean_inc(v_children_1182_);
lean_inc(v_parent_1181_);
lean_inc(v_id_1180_);
lean_dec(v___x_1179_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1205_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1202_; 
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 7, v_mvars_1174_);
v___x_1202_ = v___x_1199_;
goto v_reusejp_1201_;
}
else
{
lean_object* v_reuseFailAlloc_1204_; 
v_reuseFailAlloc_1204_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1204_, 0, v_id_1180_);
lean_ctor_set(v_reuseFailAlloc_1204_, 1, v_parent_1181_);
lean_ctor_set(v_reuseFailAlloc_1204_, 2, v_children_1182_);
lean_ctor_set(v_reuseFailAlloc_1204_, 3, v_origin_1183_);
lean_ctor_set(v_reuseFailAlloc_1204_, 4, v_depth_1184_);
lean_ctor_set(v_reuseFailAlloc_1204_, 5, v_preNormGoal_1188_);
lean_ctor_set(v_reuseFailAlloc_1204_, 6, v_normalizationState_1189_);
lean_ctor_set(v_reuseFailAlloc_1204_, 7, v_mvars_1174_);
lean_ctor_set(v_reuseFailAlloc_1204_, 8, v_forwardState_1190_);
lean_ctor_set(v_reuseFailAlloc_1204_, 9, v_forwardRuleMatches_1191_);
lean_ctor_set(v_reuseFailAlloc_1204_, 10, v_addedInIteration_1193_);
lean_ctor_set(v_reuseFailAlloc_1204_, 11, v_lastExpandedInIteration_1194_);
lean_ctor_set(v_reuseFailAlloc_1204_, 12, v_unsafeQueue_1196_);
lean_ctor_set(v_reuseFailAlloc_1204_, 13, v_failedRapps_1197_);
lean_ctor_set_uint8(v_reuseFailAlloc_1204_, sizeof(void*)*14 + 8, v_state_1185_);
lean_ctor_set_uint8(v_reuseFailAlloc_1204_, sizeof(void*)*14 + 9, v_isIrrelevant_1186_);
lean_ctor_set_uint8(v_reuseFailAlloc_1204_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1187_);
lean_ctor_set_float(v_reuseFailAlloc_1204_, sizeof(void*)*14, v_successProbability_1192_);
lean_ctor_set_uint8(v_reuseFailAlloc_1204_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1195_);
v___x_1202_ = v_reuseFailAlloc_1204_;
goto v_reusejp_1201_;
}
v_reusejp_1201_:
{
lean_object* v___x_1203_; 
lean_inc(v_introGoal_1177_);
v___x_1203_ = lean_apply_1(v_introGoal_1177_, v___x_1202_);
return v___x_1203_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardState(lean_object* v_forwardState_1207_, lean_object* v_g_1208_){
_start:
{
lean_object* v___x_1209_; lean_object* v_introGoal_1210_; lean_object* v_elimGoal_1211_; lean_object* v___x_1212_; lean_object* v_id_1213_; lean_object* v_parent_1214_; lean_object* v_children_1215_; lean_object* v_origin_1216_; lean_object* v_depth_1217_; uint8_t v_state_1218_; uint8_t v_isIrrelevant_1219_; uint8_t v_isForcedUnprovable_1220_; lean_object* v_preNormGoal_1221_; lean_object* v_normalizationState_1222_; lean_object* v_mvars_1223_; lean_object* v_forwardRuleMatches_1224_; double v_successProbability_1225_; lean_object* v_addedInIteration_1226_; lean_object* v_lastExpandedInIteration_1227_; uint8_t v_unsafeRulesSelected_1228_; lean_object* v_unsafeQueue_1229_; lean_object* v_failedRapps_1230_; lean_object* v___x_1232_; uint8_t v_isShared_1233_; uint8_t v_isSharedCheck_1238_; 
v___x_1209_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1210_ = lean_ctor_get(v___x_1209_, 0);
v_elimGoal_1211_ = lean_ctor_get(v___x_1209_, 1);
lean_inc_ref(v_elimGoal_1211_);
v___x_1212_ = lean_apply_1(v_elimGoal_1211_, v_g_1208_);
v_id_1213_ = lean_ctor_get(v___x_1212_, 0);
v_parent_1214_ = lean_ctor_get(v___x_1212_, 1);
v_children_1215_ = lean_ctor_get(v___x_1212_, 2);
v_origin_1216_ = lean_ctor_get(v___x_1212_, 3);
v_depth_1217_ = lean_ctor_get(v___x_1212_, 4);
v_state_1218_ = lean_ctor_get_uint8(v___x_1212_, sizeof(void*)*14 + 8);
v_isIrrelevant_1219_ = lean_ctor_get_uint8(v___x_1212_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1220_ = lean_ctor_get_uint8(v___x_1212_, sizeof(void*)*14 + 10);
v_preNormGoal_1221_ = lean_ctor_get(v___x_1212_, 5);
v_normalizationState_1222_ = lean_ctor_get(v___x_1212_, 6);
v_mvars_1223_ = lean_ctor_get(v___x_1212_, 7);
v_forwardRuleMatches_1224_ = lean_ctor_get(v___x_1212_, 9);
v_successProbability_1225_ = lean_ctor_get_float(v___x_1212_, sizeof(void*)*14);
v_addedInIteration_1226_ = lean_ctor_get(v___x_1212_, 10);
v_lastExpandedInIteration_1227_ = lean_ctor_get(v___x_1212_, 11);
v_unsafeRulesSelected_1228_ = lean_ctor_get_uint8(v___x_1212_, sizeof(void*)*14 + 11);
v_unsafeQueue_1229_ = lean_ctor_get(v___x_1212_, 12);
v_failedRapps_1230_ = lean_ctor_get(v___x_1212_, 13);
v_isSharedCheck_1238_ = !lean_is_exclusive(v___x_1212_);
if (v_isSharedCheck_1238_ == 0)
{
lean_object* v_unused_1239_; 
v_unused_1239_ = lean_ctor_get(v___x_1212_, 8);
lean_dec(v_unused_1239_);
v___x_1232_ = v___x_1212_;
v_isShared_1233_ = v_isSharedCheck_1238_;
goto v_resetjp_1231_;
}
else
{
lean_inc(v_failedRapps_1230_);
lean_inc(v_unsafeQueue_1229_);
lean_inc(v_lastExpandedInIteration_1227_);
lean_inc(v_addedInIteration_1226_);
lean_inc(v_forwardRuleMatches_1224_);
lean_inc(v_mvars_1223_);
lean_inc(v_normalizationState_1222_);
lean_inc(v_preNormGoal_1221_);
lean_inc(v_depth_1217_);
lean_inc(v_origin_1216_);
lean_inc(v_children_1215_);
lean_inc(v_parent_1214_);
lean_inc(v_id_1213_);
lean_dec(v___x_1212_);
v___x_1232_ = lean_box(0);
v_isShared_1233_ = v_isSharedCheck_1238_;
goto v_resetjp_1231_;
}
v_resetjp_1231_:
{
lean_object* v___x_1235_; 
if (v_isShared_1233_ == 0)
{
lean_ctor_set(v___x_1232_, 8, v_forwardState_1207_);
v___x_1235_ = v___x_1232_;
goto v_reusejp_1234_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v_id_1213_);
lean_ctor_set(v_reuseFailAlloc_1237_, 1, v_parent_1214_);
lean_ctor_set(v_reuseFailAlloc_1237_, 2, v_children_1215_);
lean_ctor_set(v_reuseFailAlloc_1237_, 3, v_origin_1216_);
lean_ctor_set(v_reuseFailAlloc_1237_, 4, v_depth_1217_);
lean_ctor_set(v_reuseFailAlloc_1237_, 5, v_preNormGoal_1221_);
lean_ctor_set(v_reuseFailAlloc_1237_, 6, v_normalizationState_1222_);
lean_ctor_set(v_reuseFailAlloc_1237_, 7, v_mvars_1223_);
lean_ctor_set(v_reuseFailAlloc_1237_, 8, v_forwardState_1207_);
lean_ctor_set(v_reuseFailAlloc_1237_, 9, v_forwardRuleMatches_1224_);
lean_ctor_set(v_reuseFailAlloc_1237_, 10, v_addedInIteration_1226_);
lean_ctor_set(v_reuseFailAlloc_1237_, 11, v_lastExpandedInIteration_1227_);
lean_ctor_set(v_reuseFailAlloc_1237_, 12, v_unsafeQueue_1229_);
lean_ctor_set(v_reuseFailAlloc_1237_, 13, v_failedRapps_1230_);
lean_ctor_set_uint8(v_reuseFailAlloc_1237_, sizeof(void*)*14 + 8, v_state_1218_);
lean_ctor_set_uint8(v_reuseFailAlloc_1237_, sizeof(void*)*14 + 9, v_isIrrelevant_1219_);
lean_ctor_set_uint8(v_reuseFailAlloc_1237_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1220_);
lean_ctor_set_float(v_reuseFailAlloc_1237_, sizeof(void*)*14, v_successProbability_1225_);
lean_ctor_set_uint8(v_reuseFailAlloc_1237_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1228_);
v___x_1235_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1234_;
}
v_reusejp_1234_:
{
lean_object* v___x_1236_; 
lean_inc(v_introGoal_1210_);
v___x_1236_ = lean_apply_1(v_introGoal_1210_, v___x_1235_);
return v___x_1236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setForwardRuleMatches(lean_object* v_forwardRuleMatches_1240_, lean_object* v_g_1241_){
_start:
{
lean_object* v___x_1242_; lean_object* v_introGoal_1243_; lean_object* v_elimGoal_1244_; lean_object* v___x_1245_; lean_object* v_id_1246_; lean_object* v_parent_1247_; lean_object* v_children_1248_; lean_object* v_origin_1249_; lean_object* v_depth_1250_; uint8_t v_state_1251_; uint8_t v_isIrrelevant_1252_; uint8_t v_isForcedUnprovable_1253_; lean_object* v_preNormGoal_1254_; lean_object* v_normalizationState_1255_; lean_object* v_mvars_1256_; lean_object* v_forwardState_1257_; double v_successProbability_1258_; lean_object* v_addedInIteration_1259_; lean_object* v_lastExpandedInIteration_1260_; uint8_t v_unsafeRulesSelected_1261_; lean_object* v_unsafeQueue_1262_; lean_object* v_failedRapps_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1271_; 
v___x_1242_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1243_ = lean_ctor_get(v___x_1242_, 0);
v_elimGoal_1244_ = lean_ctor_get(v___x_1242_, 1);
lean_inc_ref(v_elimGoal_1244_);
v___x_1245_ = lean_apply_1(v_elimGoal_1244_, v_g_1241_);
v_id_1246_ = lean_ctor_get(v___x_1245_, 0);
v_parent_1247_ = lean_ctor_get(v___x_1245_, 1);
v_children_1248_ = lean_ctor_get(v___x_1245_, 2);
v_origin_1249_ = lean_ctor_get(v___x_1245_, 3);
v_depth_1250_ = lean_ctor_get(v___x_1245_, 4);
v_state_1251_ = lean_ctor_get_uint8(v___x_1245_, sizeof(void*)*14 + 8);
v_isIrrelevant_1252_ = lean_ctor_get_uint8(v___x_1245_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1253_ = lean_ctor_get_uint8(v___x_1245_, sizeof(void*)*14 + 10);
v_preNormGoal_1254_ = lean_ctor_get(v___x_1245_, 5);
v_normalizationState_1255_ = lean_ctor_get(v___x_1245_, 6);
v_mvars_1256_ = lean_ctor_get(v___x_1245_, 7);
v_forwardState_1257_ = lean_ctor_get(v___x_1245_, 8);
v_successProbability_1258_ = lean_ctor_get_float(v___x_1245_, sizeof(void*)*14);
v_addedInIteration_1259_ = lean_ctor_get(v___x_1245_, 10);
v_lastExpandedInIteration_1260_ = lean_ctor_get(v___x_1245_, 11);
v_unsafeRulesSelected_1261_ = lean_ctor_get_uint8(v___x_1245_, sizeof(void*)*14 + 11);
v_unsafeQueue_1262_ = lean_ctor_get(v___x_1245_, 12);
v_failedRapps_1263_ = lean_ctor_get(v___x_1245_, 13);
v_isSharedCheck_1271_ = !lean_is_exclusive(v___x_1245_);
if (v_isSharedCheck_1271_ == 0)
{
lean_object* v_unused_1272_; 
v_unused_1272_ = lean_ctor_get(v___x_1245_, 9);
lean_dec(v_unused_1272_);
v___x_1265_ = v___x_1245_;
v_isShared_1266_ = v_isSharedCheck_1271_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_failedRapps_1263_);
lean_inc(v_unsafeQueue_1262_);
lean_inc(v_lastExpandedInIteration_1260_);
lean_inc(v_addedInIteration_1259_);
lean_inc(v_forwardState_1257_);
lean_inc(v_mvars_1256_);
lean_inc(v_normalizationState_1255_);
lean_inc(v_preNormGoal_1254_);
lean_inc(v_depth_1250_);
lean_inc(v_origin_1249_);
lean_inc(v_children_1248_);
lean_inc(v_parent_1247_);
lean_inc(v_id_1246_);
lean_dec(v___x_1245_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1271_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___x_1268_; 
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 9, v_forwardRuleMatches_1240_);
v___x_1268_ = v___x_1265_;
goto v_reusejp_1267_;
}
else
{
lean_object* v_reuseFailAlloc_1270_; 
v_reuseFailAlloc_1270_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1270_, 0, v_id_1246_);
lean_ctor_set(v_reuseFailAlloc_1270_, 1, v_parent_1247_);
lean_ctor_set(v_reuseFailAlloc_1270_, 2, v_children_1248_);
lean_ctor_set(v_reuseFailAlloc_1270_, 3, v_origin_1249_);
lean_ctor_set(v_reuseFailAlloc_1270_, 4, v_depth_1250_);
lean_ctor_set(v_reuseFailAlloc_1270_, 5, v_preNormGoal_1254_);
lean_ctor_set(v_reuseFailAlloc_1270_, 6, v_normalizationState_1255_);
lean_ctor_set(v_reuseFailAlloc_1270_, 7, v_mvars_1256_);
lean_ctor_set(v_reuseFailAlloc_1270_, 8, v_forwardState_1257_);
lean_ctor_set(v_reuseFailAlloc_1270_, 9, v_forwardRuleMatches_1240_);
lean_ctor_set(v_reuseFailAlloc_1270_, 10, v_addedInIteration_1259_);
lean_ctor_set(v_reuseFailAlloc_1270_, 11, v_lastExpandedInIteration_1260_);
lean_ctor_set(v_reuseFailAlloc_1270_, 12, v_unsafeQueue_1262_);
lean_ctor_set(v_reuseFailAlloc_1270_, 13, v_failedRapps_1263_);
lean_ctor_set_uint8(v_reuseFailAlloc_1270_, sizeof(void*)*14 + 8, v_state_1251_);
lean_ctor_set_uint8(v_reuseFailAlloc_1270_, sizeof(void*)*14 + 9, v_isIrrelevant_1252_);
lean_ctor_set_uint8(v_reuseFailAlloc_1270_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1253_);
lean_ctor_set_float(v_reuseFailAlloc_1270_, sizeof(void*)*14, v_successProbability_1258_);
lean_ctor_set_uint8(v_reuseFailAlloc_1270_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1261_);
v___x_1268_ = v_reuseFailAlloc_1270_;
goto v_reusejp_1267_;
}
v_reusejp_1267_:
{
lean_object* v___x_1269_; 
lean_inc(v_introGoal_1243_);
v___x_1269_ = lean_apply_1(v_introGoal_1243_, v___x_1268_);
return v___x_1269_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability(double v_successProbability_1273_, lean_object* v_g_1274_){
_start:
{
lean_object* v___x_1275_; lean_object* v_introGoal_1276_; lean_object* v_elimGoal_1277_; lean_object* v___x_1278_; lean_object* v_id_1279_; lean_object* v_parent_1280_; lean_object* v_children_1281_; lean_object* v_origin_1282_; lean_object* v_depth_1283_; uint8_t v_state_1284_; uint8_t v_isIrrelevant_1285_; uint8_t v_isForcedUnprovable_1286_; lean_object* v_preNormGoal_1287_; lean_object* v_normalizationState_1288_; lean_object* v_mvars_1289_; lean_object* v_forwardState_1290_; lean_object* v_forwardRuleMatches_1291_; lean_object* v_addedInIteration_1292_; lean_object* v_lastExpandedInIteration_1293_; uint8_t v_unsafeRulesSelected_1294_; lean_object* v_unsafeQueue_1295_; lean_object* v_failedRapps_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1304_; 
v___x_1275_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1276_ = lean_ctor_get(v___x_1275_, 0);
v_elimGoal_1277_ = lean_ctor_get(v___x_1275_, 1);
lean_inc_ref(v_elimGoal_1277_);
v___x_1278_ = lean_apply_1(v_elimGoal_1277_, v_g_1274_);
v_id_1279_ = lean_ctor_get(v___x_1278_, 0);
v_parent_1280_ = lean_ctor_get(v___x_1278_, 1);
v_children_1281_ = lean_ctor_get(v___x_1278_, 2);
v_origin_1282_ = lean_ctor_get(v___x_1278_, 3);
v_depth_1283_ = lean_ctor_get(v___x_1278_, 4);
v_state_1284_ = lean_ctor_get_uint8(v___x_1278_, sizeof(void*)*14 + 8);
v_isIrrelevant_1285_ = lean_ctor_get_uint8(v___x_1278_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1286_ = lean_ctor_get_uint8(v___x_1278_, sizeof(void*)*14 + 10);
v_preNormGoal_1287_ = lean_ctor_get(v___x_1278_, 5);
v_normalizationState_1288_ = lean_ctor_get(v___x_1278_, 6);
v_mvars_1289_ = lean_ctor_get(v___x_1278_, 7);
v_forwardState_1290_ = lean_ctor_get(v___x_1278_, 8);
v_forwardRuleMatches_1291_ = lean_ctor_get(v___x_1278_, 9);
v_addedInIteration_1292_ = lean_ctor_get(v___x_1278_, 10);
v_lastExpandedInIteration_1293_ = lean_ctor_get(v___x_1278_, 11);
v_unsafeRulesSelected_1294_ = lean_ctor_get_uint8(v___x_1278_, sizeof(void*)*14 + 11);
v_unsafeQueue_1295_ = lean_ctor_get(v___x_1278_, 12);
v_failedRapps_1296_ = lean_ctor_get(v___x_1278_, 13);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1278_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1298_ = v___x_1278_;
v_isShared_1299_ = v_isSharedCheck_1304_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_failedRapps_1296_);
lean_inc(v_unsafeQueue_1295_);
lean_inc(v_lastExpandedInIteration_1293_);
lean_inc(v_addedInIteration_1292_);
lean_inc(v_forwardRuleMatches_1291_);
lean_inc(v_forwardState_1290_);
lean_inc(v_mvars_1289_);
lean_inc(v_normalizationState_1288_);
lean_inc(v_preNormGoal_1287_);
lean_inc(v_depth_1283_);
lean_inc(v_origin_1282_);
lean_inc(v_children_1281_);
lean_inc(v_parent_1280_);
lean_inc(v_id_1279_);
lean_dec(v___x_1278_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1304_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
lean_object* v___x_1301_; 
if (v_isShared_1299_ == 0)
{
v___x_1301_ = v___x_1298_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_id_1279_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v_parent_1280_);
lean_ctor_set(v_reuseFailAlloc_1303_, 2, v_children_1281_);
lean_ctor_set(v_reuseFailAlloc_1303_, 3, v_origin_1282_);
lean_ctor_set(v_reuseFailAlloc_1303_, 4, v_depth_1283_);
lean_ctor_set(v_reuseFailAlloc_1303_, 5, v_preNormGoal_1287_);
lean_ctor_set(v_reuseFailAlloc_1303_, 6, v_normalizationState_1288_);
lean_ctor_set(v_reuseFailAlloc_1303_, 7, v_mvars_1289_);
lean_ctor_set(v_reuseFailAlloc_1303_, 8, v_forwardState_1290_);
lean_ctor_set(v_reuseFailAlloc_1303_, 9, v_forwardRuleMatches_1291_);
lean_ctor_set(v_reuseFailAlloc_1303_, 10, v_addedInIteration_1292_);
lean_ctor_set(v_reuseFailAlloc_1303_, 11, v_lastExpandedInIteration_1293_);
lean_ctor_set(v_reuseFailAlloc_1303_, 12, v_unsafeQueue_1295_);
lean_ctor_set(v_reuseFailAlloc_1303_, 13, v_failedRapps_1296_);
lean_ctor_set_uint8(v_reuseFailAlloc_1303_, sizeof(void*)*14 + 8, v_state_1284_);
lean_ctor_set_uint8(v_reuseFailAlloc_1303_, sizeof(void*)*14 + 9, v_isIrrelevant_1285_);
lean_ctor_set_uint8(v_reuseFailAlloc_1303_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1286_);
lean_ctor_set_uint8(v_reuseFailAlloc_1303_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1294_);
v___x_1301_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
lean_object* v___x_1302_; 
lean_ctor_set_float(v___x_1301_, sizeof(void*)*14, v_successProbability_1273_);
lean_inc(v_introGoal_1276_);
v___x_1302_ = lean_apply_1(v_introGoal_1276_, v___x_1301_);
return v___x_1302_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setSuccessProbability___boxed(lean_object* v_successProbability_1305_, lean_object* v_g_1306_){
_start:
{
double v_successProbability_boxed_1307_; lean_object* v_res_1308_; 
v_successProbability_boxed_1307_ = lean_unbox_float(v_successProbability_1305_);
lean_dec_ref(v_successProbability_1305_);
v_res_1308_ = lp_aesop_Aesop_Goal_setSuccessProbability(v_successProbability_boxed_1307_, v_g_1306_);
return v_res_1308_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setAddedInIteration(lean_object* v_addedInIteration_1309_, lean_object* v_g_1310_){
_start:
{
lean_object* v___x_1311_; lean_object* v_introGoal_1312_; lean_object* v_elimGoal_1313_; lean_object* v___x_1314_; lean_object* v_id_1315_; lean_object* v_parent_1316_; lean_object* v_children_1317_; lean_object* v_origin_1318_; lean_object* v_depth_1319_; uint8_t v_state_1320_; uint8_t v_isIrrelevant_1321_; uint8_t v_isForcedUnprovable_1322_; lean_object* v_preNormGoal_1323_; lean_object* v_normalizationState_1324_; lean_object* v_mvars_1325_; lean_object* v_forwardState_1326_; lean_object* v_forwardRuleMatches_1327_; double v_successProbability_1328_; lean_object* v_lastExpandedInIteration_1329_; uint8_t v_unsafeRulesSelected_1330_; lean_object* v_unsafeQueue_1331_; lean_object* v_failedRapps_1332_; lean_object* v___x_1334_; uint8_t v_isShared_1335_; uint8_t v_isSharedCheck_1340_; 
v___x_1311_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1312_ = lean_ctor_get(v___x_1311_, 0);
v_elimGoal_1313_ = lean_ctor_get(v___x_1311_, 1);
lean_inc_ref(v_elimGoal_1313_);
v___x_1314_ = lean_apply_1(v_elimGoal_1313_, v_g_1310_);
v_id_1315_ = lean_ctor_get(v___x_1314_, 0);
v_parent_1316_ = lean_ctor_get(v___x_1314_, 1);
v_children_1317_ = lean_ctor_get(v___x_1314_, 2);
v_origin_1318_ = lean_ctor_get(v___x_1314_, 3);
v_depth_1319_ = lean_ctor_get(v___x_1314_, 4);
v_state_1320_ = lean_ctor_get_uint8(v___x_1314_, sizeof(void*)*14 + 8);
v_isIrrelevant_1321_ = lean_ctor_get_uint8(v___x_1314_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1322_ = lean_ctor_get_uint8(v___x_1314_, sizeof(void*)*14 + 10);
v_preNormGoal_1323_ = lean_ctor_get(v___x_1314_, 5);
v_normalizationState_1324_ = lean_ctor_get(v___x_1314_, 6);
v_mvars_1325_ = lean_ctor_get(v___x_1314_, 7);
v_forwardState_1326_ = lean_ctor_get(v___x_1314_, 8);
v_forwardRuleMatches_1327_ = lean_ctor_get(v___x_1314_, 9);
v_successProbability_1328_ = lean_ctor_get_float(v___x_1314_, sizeof(void*)*14);
v_lastExpandedInIteration_1329_ = lean_ctor_get(v___x_1314_, 11);
v_unsafeRulesSelected_1330_ = lean_ctor_get_uint8(v___x_1314_, sizeof(void*)*14 + 11);
v_unsafeQueue_1331_ = lean_ctor_get(v___x_1314_, 12);
v_failedRapps_1332_ = lean_ctor_get(v___x_1314_, 13);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1314_);
if (v_isSharedCheck_1340_ == 0)
{
lean_object* v_unused_1341_; 
v_unused_1341_ = lean_ctor_get(v___x_1314_, 10);
lean_dec(v_unused_1341_);
v___x_1334_ = v___x_1314_;
v_isShared_1335_ = v_isSharedCheck_1340_;
goto v_resetjp_1333_;
}
else
{
lean_inc(v_failedRapps_1332_);
lean_inc(v_unsafeQueue_1331_);
lean_inc(v_lastExpandedInIteration_1329_);
lean_inc(v_forwardRuleMatches_1327_);
lean_inc(v_forwardState_1326_);
lean_inc(v_mvars_1325_);
lean_inc(v_normalizationState_1324_);
lean_inc(v_preNormGoal_1323_);
lean_inc(v_depth_1319_);
lean_inc(v_origin_1318_);
lean_inc(v_children_1317_);
lean_inc(v_parent_1316_);
lean_inc(v_id_1315_);
lean_dec(v___x_1314_);
v___x_1334_ = lean_box(0);
v_isShared_1335_ = v_isSharedCheck_1340_;
goto v_resetjp_1333_;
}
v_resetjp_1333_:
{
lean_object* v___x_1337_; 
if (v_isShared_1335_ == 0)
{
lean_ctor_set(v___x_1334_, 10, v_addedInIteration_1309_);
v___x_1337_ = v___x_1334_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_id_1315_);
lean_ctor_set(v_reuseFailAlloc_1339_, 1, v_parent_1316_);
lean_ctor_set(v_reuseFailAlloc_1339_, 2, v_children_1317_);
lean_ctor_set(v_reuseFailAlloc_1339_, 3, v_origin_1318_);
lean_ctor_set(v_reuseFailAlloc_1339_, 4, v_depth_1319_);
lean_ctor_set(v_reuseFailAlloc_1339_, 5, v_preNormGoal_1323_);
lean_ctor_set(v_reuseFailAlloc_1339_, 6, v_normalizationState_1324_);
lean_ctor_set(v_reuseFailAlloc_1339_, 7, v_mvars_1325_);
lean_ctor_set(v_reuseFailAlloc_1339_, 8, v_forwardState_1326_);
lean_ctor_set(v_reuseFailAlloc_1339_, 9, v_forwardRuleMatches_1327_);
lean_ctor_set(v_reuseFailAlloc_1339_, 10, v_addedInIteration_1309_);
lean_ctor_set(v_reuseFailAlloc_1339_, 11, v_lastExpandedInIteration_1329_);
lean_ctor_set(v_reuseFailAlloc_1339_, 12, v_unsafeQueue_1331_);
lean_ctor_set(v_reuseFailAlloc_1339_, 13, v_failedRapps_1332_);
lean_ctor_set_uint8(v_reuseFailAlloc_1339_, sizeof(void*)*14 + 8, v_state_1320_);
lean_ctor_set_uint8(v_reuseFailAlloc_1339_, sizeof(void*)*14 + 9, v_isIrrelevant_1321_);
lean_ctor_set_uint8(v_reuseFailAlloc_1339_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1322_);
lean_ctor_set_float(v_reuseFailAlloc_1339_, sizeof(void*)*14, v_successProbability_1328_);
lean_ctor_set_uint8(v_reuseFailAlloc_1339_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1330_);
v___x_1337_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
lean_object* v___x_1338_; 
lean_inc(v_introGoal_1312_);
v___x_1338_ = lean_apply_1(v_introGoal_1312_, v___x_1337_);
return v___x_1338_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setLastExpandedInIteration(lean_object* v_lastExpandedInIteration_1342_, lean_object* v_g_1343_){
_start:
{
lean_object* v___x_1344_; lean_object* v_introGoal_1345_; lean_object* v_elimGoal_1346_; lean_object* v___x_1347_; lean_object* v_id_1348_; lean_object* v_parent_1349_; lean_object* v_children_1350_; lean_object* v_origin_1351_; lean_object* v_depth_1352_; uint8_t v_state_1353_; uint8_t v_isIrrelevant_1354_; uint8_t v_isForcedUnprovable_1355_; lean_object* v_preNormGoal_1356_; lean_object* v_normalizationState_1357_; lean_object* v_mvars_1358_; lean_object* v_forwardState_1359_; lean_object* v_forwardRuleMatches_1360_; double v_successProbability_1361_; lean_object* v_addedInIteration_1362_; uint8_t v_unsafeRulesSelected_1363_; lean_object* v_unsafeQueue_1364_; lean_object* v_failedRapps_1365_; lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1373_; 
v___x_1344_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1345_ = lean_ctor_get(v___x_1344_, 0);
v_elimGoal_1346_ = lean_ctor_get(v___x_1344_, 1);
lean_inc_ref(v_elimGoal_1346_);
v___x_1347_ = lean_apply_1(v_elimGoal_1346_, v_g_1343_);
v_id_1348_ = lean_ctor_get(v___x_1347_, 0);
v_parent_1349_ = lean_ctor_get(v___x_1347_, 1);
v_children_1350_ = lean_ctor_get(v___x_1347_, 2);
v_origin_1351_ = lean_ctor_get(v___x_1347_, 3);
v_depth_1352_ = lean_ctor_get(v___x_1347_, 4);
v_state_1353_ = lean_ctor_get_uint8(v___x_1347_, sizeof(void*)*14 + 8);
v_isIrrelevant_1354_ = lean_ctor_get_uint8(v___x_1347_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1355_ = lean_ctor_get_uint8(v___x_1347_, sizeof(void*)*14 + 10);
v_preNormGoal_1356_ = lean_ctor_get(v___x_1347_, 5);
v_normalizationState_1357_ = lean_ctor_get(v___x_1347_, 6);
v_mvars_1358_ = lean_ctor_get(v___x_1347_, 7);
v_forwardState_1359_ = lean_ctor_get(v___x_1347_, 8);
v_forwardRuleMatches_1360_ = lean_ctor_get(v___x_1347_, 9);
v_successProbability_1361_ = lean_ctor_get_float(v___x_1347_, sizeof(void*)*14);
v_addedInIteration_1362_ = lean_ctor_get(v___x_1347_, 10);
v_unsafeRulesSelected_1363_ = lean_ctor_get_uint8(v___x_1347_, sizeof(void*)*14 + 11);
v_unsafeQueue_1364_ = lean_ctor_get(v___x_1347_, 12);
v_failedRapps_1365_ = lean_ctor_get(v___x_1347_, 13);
v_isSharedCheck_1373_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1373_ == 0)
{
lean_object* v_unused_1374_; 
v_unused_1374_ = lean_ctor_get(v___x_1347_, 11);
lean_dec(v_unused_1374_);
v___x_1367_ = v___x_1347_;
v_isShared_1368_ = v_isSharedCheck_1373_;
goto v_resetjp_1366_;
}
else
{
lean_inc(v_failedRapps_1365_);
lean_inc(v_unsafeQueue_1364_);
lean_inc(v_addedInIteration_1362_);
lean_inc(v_forwardRuleMatches_1360_);
lean_inc(v_forwardState_1359_);
lean_inc(v_mvars_1358_);
lean_inc(v_normalizationState_1357_);
lean_inc(v_preNormGoal_1356_);
lean_inc(v_depth_1352_);
lean_inc(v_origin_1351_);
lean_inc(v_children_1350_);
lean_inc(v_parent_1349_);
lean_inc(v_id_1348_);
lean_dec(v___x_1347_);
v___x_1367_ = lean_box(0);
v_isShared_1368_ = v_isSharedCheck_1373_;
goto v_resetjp_1366_;
}
v_resetjp_1366_:
{
lean_object* v___x_1370_; 
if (v_isShared_1368_ == 0)
{
lean_ctor_set(v___x_1367_, 11, v_lastExpandedInIteration_1342_);
v___x_1370_ = v___x_1367_;
goto v_reusejp_1369_;
}
else
{
lean_object* v_reuseFailAlloc_1372_; 
v_reuseFailAlloc_1372_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1372_, 0, v_id_1348_);
lean_ctor_set(v_reuseFailAlloc_1372_, 1, v_parent_1349_);
lean_ctor_set(v_reuseFailAlloc_1372_, 2, v_children_1350_);
lean_ctor_set(v_reuseFailAlloc_1372_, 3, v_origin_1351_);
lean_ctor_set(v_reuseFailAlloc_1372_, 4, v_depth_1352_);
lean_ctor_set(v_reuseFailAlloc_1372_, 5, v_preNormGoal_1356_);
lean_ctor_set(v_reuseFailAlloc_1372_, 6, v_normalizationState_1357_);
lean_ctor_set(v_reuseFailAlloc_1372_, 7, v_mvars_1358_);
lean_ctor_set(v_reuseFailAlloc_1372_, 8, v_forwardState_1359_);
lean_ctor_set(v_reuseFailAlloc_1372_, 9, v_forwardRuleMatches_1360_);
lean_ctor_set(v_reuseFailAlloc_1372_, 10, v_addedInIteration_1362_);
lean_ctor_set(v_reuseFailAlloc_1372_, 11, v_lastExpandedInIteration_1342_);
lean_ctor_set(v_reuseFailAlloc_1372_, 12, v_unsafeQueue_1364_);
lean_ctor_set(v_reuseFailAlloc_1372_, 13, v_failedRapps_1365_);
lean_ctor_set_uint8(v_reuseFailAlloc_1372_, sizeof(void*)*14 + 8, v_state_1353_);
lean_ctor_set_uint8(v_reuseFailAlloc_1372_, sizeof(void*)*14 + 9, v_isIrrelevant_1354_);
lean_ctor_set_uint8(v_reuseFailAlloc_1372_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1355_);
lean_ctor_set_float(v_reuseFailAlloc_1372_, sizeof(void*)*14, v_successProbability_1361_);
lean_ctor_set_uint8(v_reuseFailAlloc_1372_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1363_);
v___x_1370_ = v_reuseFailAlloc_1372_;
goto v_reusejp_1369_;
}
v_reusejp_1369_:
{
lean_object* v___x_1371_; 
lean_inc(v_introGoal_1345_);
v___x_1371_ = lean_apply_1(v_introGoal_1345_, v___x_1370_);
return v___x_1371_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected(uint8_t v_unsafeRulesSelected_1375_, lean_object* v_g_1376_){
_start:
{
lean_object* v___x_1377_; lean_object* v_introGoal_1378_; lean_object* v_elimGoal_1379_; lean_object* v___x_1380_; lean_object* v_id_1381_; lean_object* v_parent_1382_; lean_object* v_children_1383_; lean_object* v_origin_1384_; lean_object* v_depth_1385_; uint8_t v_state_1386_; uint8_t v_isIrrelevant_1387_; uint8_t v_isForcedUnprovable_1388_; lean_object* v_preNormGoal_1389_; lean_object* v_normalizationState_1390_; lean_object* v_mvars_1391_; lean_object* v_forwardState_1392_; lean_object* v_forwardRuleMatches_1393_; double v_successProbability_1394_; lean_object* v_addedInIteration_1395_; lean_object* v_lastExpandedInIteration_1396_; lean_object* v_unsafeQueue_1397_; lean_object* v_failedRapps_1398_; lean_object* v___x_1400_; uint8_t v_isShared_1401_; uint8_t v_isSharedCheck_1406_; 
v___x_1377_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1378_ = lean_ctor_get(v___x_1377_, 0);
v_elimGoal_1379_ = lean_ctor_get(v___x_1377_, 1);
lean_inc_ref(v_elimGoal_1379_);
v___x_1380_ = lean_apply_1(v_elimGoal_1379_, v_g_1376_);
v_id_1381_ = lean_ctor_get(v___x_1380_, 0);
v_parent_1382_ = lean_ctor_get(v___x_1380_, 1);
v_children_1383_ = lean_ctor_get(v___x_1380_, 2);
v_origin_1384_ = lean_ctor_get(v___x_1380_, 3);
v_depth_1385_ = lean_ctor_get(v___x_1380_, 4);
v_state_1386_ = lean_ctor_get_uint8(v___x_1380_, sizeof(void*)*14 + 8);
v_isIrrelevant_1387_ = lean_ctor_get_uint8(v___x_1380_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1388_ = lean_ctor_get_uint8(v___x_1380_, sizeof(void*)*14 + 10);
v_preNormGoal_1389_ = lean_ctor_get(v___x_1380_, 5);
v_normalizationState_1390_ = lean_ctor_get(v___x_1380_, 6);
v_mvars_1391_ = lean_ctor_get(v___x_1380_, 7);
v_forwardState_1392_ = lean_ctor_get(v___x_1380_, 8);
v_forwardRuleMatches_1393_ = lean_ctor_get(v___x_1380_, 9);
v_successProbability_1394_ = lean_ctor_get_float(v___x_1380_, sizeof(void*)*14);
v_addedInIteration_1395_ = lean_ctor_get(v___x_1380_, 10);
v_lastExpandedInIteration_1396_ = lean_ctor_get(v___x_1380_, 11);
v_unsafeQueue_1397_ = lean_ctor_get(v___x_1380_, 12);
v_failedRapps_1398_ = lean_ctor_get(v___x_1380_, 13);
v_isSharedCheck_1406_ = !lean_is_exclusive(v___x_1380_);
if (v_isSharedCheck_1406_ == 0)
{
v___x_1400_ = v___x_1380_;
v_isShared_1401_ = v_isSharedCheck_1406_;
goto v_resetjp_1399_;
}
else
{
lean_inc(v_failedRapps_1398_);
lean_inc(v_unsafeQueue_1397_);
lean_inc(v_lastExpandedInIteration_1396_);
lean_inc(v_addedInIteration_1395_);
lean_inc(v_forwardRuleMatches_1393_);
lean_inc(v_forwardState_1392_);
lean_inc(v_mvars_1391_);
lean_inc(v_normalizationState_1390_);
lean_inc(v_preNormGoal_1389_);
lean_inc(v_depth_1385_);
lean_inc(v_origin_1384_);
lean_inc(v_children_1383_);
lean_inc(v_parent_1382_);
lean_inc(v_id_1381_);
lean_dec(v___x_1380_);
v___x_1400_ = lean_box(0);
v_isShared_1401_ = v_isSharedCheck_1406_;
goto v_resetjp_1399_;
}
v_resetjp_1399_:
{
lean_object* v___x_1403_; 
if (v_isShared_1401_ == 0)
{
v___x_1403_ = v___x_1400_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1405_; 
v_reuseFailAlloc_1405_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1405_, 0, v_id_1381_);
lean_ctor_set(v_reuseFailAlloc_1405_, 1, v_parent_1382_);
lean_ctor_set(v_reuseFailAlloc_1405_, 2, v_children_1383_);
lean_ctor_set(v_reuseFailAlloc_1405_, 3, v_origin_1384_);
lean_ctor_set(v_reuseFailAlloc_1405_, 4, v_depth_1385_);
lean_ctor_set(v_reuseFailAlloc_1405_, 5, v_preNormGoal_1389_);
lean_ctor_set(v_reuseFailAlloc_1405_, 6, v_normalizationState_1390_);
lean_ctor_set(v_reuseFailAlloc_1405_, 7, v_mvars_1391_);
lean_ctor_set(v_reuseFailAlloc_1405_, 8, v_forwardState_1392_);
lean_ctor_set(v_reuseFailAlloc_1405_, 9, v_forwardRuleMatches_1393_);
lean_ctor_set(v_reuseFailAlloc_1405_, 10, v_addedInIteration_1395_);
lean_ctor_set(v_reuseFailAlloc_1405_, 11, v_lastExpandedInIteration_1396_);
lean_ctor_set(v_reuseFailAlloc_1405_, 12, v_unsafeQueue_1397_);
lean_ctor_set(v_reuseFailAlloc_1405_, 13, v_failedRapps_1398_);
lean_ctor_set_uint8(v_reuseFailAlloc_1405_, sizeof(void*)*14 + 8, v_state_1386_);
lean_ctor_set_uint8(v_reuseFailAlloc_1405_, sizeof(void*)*14 + 9, v_isIrrelevant_1387_);
lean_ctor_set_uint8(v_reuseFailAlloc_1405_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1388_);
lean_ctor_set_float(v_reuseFailAlloc_1405_, sizeof(void*)*14, v_successProbability_1394_);
v___x_1403_ = v_reuseFailAlloc_1405_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
lean_object* v___x_1404_; 
lean_ctor_set_uint8(v___x_1403_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1375_);
lean_inc(v_introGoal_1378_);
v___x_1404_ = lean_apply_1(v_introGoal_1378_, v___x_1403_);
return v___x_1404_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeRulesSelected___boxed(lean_object* v_unsafeRulesSelected_1407_, lean_object* v_g_1408_){
_start:
{
uint8_t v_unsafeRulesSelected_boxed_1409_; lean_object* v_res_1410_; 
v_unsafeRulesSelected_boxed_1409_ = lean_unbox(v_unsafeRulesSelected_1407_);
v_res_1410_ = lp_aesop_Aesop_Goal_setUnsafeRulesSelected(v_unsafeRulesSelected_boxed_1409_, v_g_1408_);
return v_res_1410_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setUnsafeQueue(lean_object* v_unsafeQueue_1411_, lean_object* v_g_1412_){
_start:
{
lean_object* v___x_1413_; lean_object* v_introGoal_1414_; lean_object* v_elimGoal_1415_; lean_object* v___x_1416_; lean_object* v_id_1417_; lean_object* v_parent_1418_; lean_object* v_children_1419_; lean_object* v_origin_1420_; lean_object* v_depth_1421_; uint8_t v_state_1422_; uint8_t v_isIrrelevant_1423_; uint8_t v_isForcedUnprovable_1424_; lean_object* v_preNormGoal_1425_; lean_object* v_normalizationState_1426_; lean_object* v_mvars_1427_; lean_object* v_forwardState_1428_; lean_object* v_forwardRuleMatches_1429_; double v_successProbability_1430_; lean_object* v_addedInIteration_1431_; lean_object* v_lastExpandedInIteration_1432_; uint8_t v_unsafeRulesSelected_1433_; lean_object* v_failedRapps_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1442_; 
v___x_1413_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1414_ = lean_ctor_get(v___x_1413_, 0);
v_elimGoal_1415_ = lean_ctor_get(v___x_1413_, 1);
lean_inc_ref(v_elimGoal_1415_);
v___x_1416_ = lean_apply_1(v_elimGoal_1415_, v_g_1412_);
v_id_1417_ = lean_ctor_get(v___x_1416_, 0);
v_parent_1418_ = lean_ctor_get(v___x_1416_, 1);
v_children_1419_ = lean_ctor_get(v___x_1416_, 2);
v_origin_1420_ = lean_ctor_get(v___x_1416_, 3);
v_depth_1421_ = lean_ctor_get(v___x_1416_, 4);
v_state_1422_ = lean_ctor_get_uint8(v___x_1416_, sizeof(void*)*14 + 8);
v_isIrrelevant_1423_ = lean_ctor_get_uint8(v___x_1416_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1424_ = lean_ctor_get_uint8(v___x_1416_, sizeof(void*)*14 + 10);
v_preNormGoal_1425_ = lean_ctor_get(v___x_1416_, 5);
v_normalizationState_1426_ = lean_ctor_get(v___x_1416_, 6);
v_mvars_1427_ = lean_ctor_get(v___x_1416_, 7);
v_forwardState_1428_ = lean_ctor_get(v___x_1416_, 8);
v_forwardRuleMatches_1429_ = lean_ctor_get(v___x_1416_, 9);
v_successProbability_1430_ = lean_ctor_get_float(v___x_1416_, sizeof(void*)*14);
v_addedInIteration_1431_ = lean_ctor_get(v___x_1416_, 10);
v_lastExpandedInIteration_1432_ = lean_ctor_get(v___x_1416_, 11);
v_unsafeRulesSelected_1433_ = lean_ctor_get_uint8(v___x_1416_, sizeof(void*)*14 + 11);
v_failedRapps_1434_ = lean_ctor_get(v___x_1416_, 13);
v_isSharedCheck_1442_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1442_ == 0)
{
lean_object* v_unused_1443_; 
v_unused_1443_ = lean_ctor_get(v___x_1416_, 12);
lean_dec(v_unused_1443_);
v___x_1436_ = v___x_1416_;
v_isShared_1437_ = v_isSharedCheck_1442_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_failedRapps_1434_);
lean_inc(v_lastExpandedInIteration_1432_);
lean_inc(v_addedInIteration_1431_);
lean_inc(v_forwardRuleMatches_1429_);
lean_inc(v_forwardState_1428_);
lean_inc(v_mvars_1427_);
lean_inc(v_normalizationState_1426_);
lean_inc(v_preNormGoal_1425_);
lean_inc(v_depth_1421_);
lean_inc(v_origin_1420_);
lean_inc(v_children_1419_);
lean_inc(v_parent_1418_);
lean_inc(v_id_1417_);
lean_dec(v___x_1416_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1442_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1439_; 
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 12, v_unsafeQueue_1411_);
v___x_1439_ = v___x_1436_;
goto v_reusejp_1438_;
}
else
{
lean_object* v_reuseFailAlloc_1441_; 
v_reuseFailAlloc_1441_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1441_, 0, v_id_1417_);
lean_ctor_set(v_reuseFailAlloc_1441_, 1, v_parent_1418_);
lean_ctor_set(v_reuseFailAlloc_1441_, 2, v_children_1419_);
lean_ctor_set(v_reuseFailAlloc_1441_, 3, v_origin_1420_);
lean_ctor_set(v_reuseFailAlloc_1441_, 4, v_depth_1421_);
lean_ctor_set(v_reuseFailAlloc_1441_, 5, v_preNormGoal_1425_);
lean_ctor_set(v_reuseFailAlloc_1441_, 6, v_normalizationState_1426_);
lean_ctor_set(v_reuseFailAlloc_1441_, 7, v_mvars_1427_);
lean_ctor_set(v_reuseFailAlloc_1441_, 8, v_forwardState_1428_);
lean_ctor_set(v_reuseFailAlloc_1441_, 9, v_forwardRuleMatches_1429_);
lean_ctor_set(v_reuseFailAlloc_1441_, 10, v_addedInIteration_1431_);
lean_ctor_set(v_reuseFailAlloc_1441_, 11, v_lastExpandedInIteration_1432_);
lean_ctor_set(v_reuseFailAlloc_1441_, 12, v_unsafeQueue_1411_);
lean_ctor_set(v_reuseFailAlloc_1441_, 13, v_failedRapps_1434_);
lean_ctor_set_uint8(v_reuseFailAlloc_1441_, sizeof(void*)*14 + 8, v_state_1422_);
lean_ctor_set_uint8(v_reuseFailAlloc_1441_, sizeof(void*)*14 + 9, v_isIrrelevant_1423_);
lean_ctor_set_uint8(v_reuseFailAlloc_1441_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1424_);
lean_ctor_set_float(v_reuseFailAlloc_1441_, sizeof(void*)*14, v_successProbability_1430_);
lean_ctor_set_uint8(v_reuseFailAlloc_1441_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1433_);
v___x_1439_ = v_reuseFailAlloc_1441_;
goto v_reusejp_1438_;
}
v_reusejp_1438_:
{
lean_object* v___x_1440_; 
lean_inc(v_introGoal_1414_);
v___x_1440_ = lean_apply_1(v_introGoal_1414_, v___x_1439_);
return v___x_1440_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState(uint8_t v_state_1444_, lean_object* v_g_1445_){
_start:
{
lean_object* v___x_1446_; lean_object* v_introGoal_1447_; lean_object* v_elimGoal_1448_; lean_object* v___x_1449_; lean_object* v_id_1450_; lean_object* v_parent_1451_; lean_object* v_children_1452_; lean_object* v_origin_1453_; lean_object* v_depth_1454_; uint8_t v_isIrrelevant_1455_; uint8_t v_isForcedUnprovable_1456_; lean_object* v_preNormGoal_1457_; lean_object* v_normalizationState_1458_; lean_object* v_mvars_1459_; lean_object* v_forwardState_1460_; lean_object* v_forwardRuleMatches_1461_; double v_successProbability_1462_; lean_object* v_addedInIteration_1463_; lean_object* v_lastExpandedInIteration_1464_; uint8_t v_unsafeRulesSelected_1465_; lean_object* v_unsafeQueue_1466_; lean_object* v_failedRapps_1467_; lean_object* v___x_1469_; uint8_t v_isShared_1470_; uint8_t v_isSharedCheck_1475_; 
v___x_1446_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1447_ = lean_ctor_get(v___x_1446_, 0);
v_elimGoal_1448_ = lean_ctor_get(v___x_1446_, 1);
lean_inc_ref(v_elimGoal_1448_);
v___x_1449_ = lean_apply_1(v_elimGoal_1448_, v_g_1445_);
v_id_1450_ = lean_ctor_get(v___x_1449_, 0);
v_parent_1451_ = lean_ctor_get(v___x_1449_, 1);
v_children_1452_ = lean_ctor_get(v___x_1449_, 2);
v_origin_1453_ = lean_ctor_get(v___x_1449_, 3);
v_depth_1454_ = lean_ctor_get(v___x_1449_, 4);
v_isIrrelevant_1455_ = lean_ctor_get_uint8(v___x_1449_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1456_ = lean_ctor_get_uint8(v___x_1449_, sizeof(void*)*14 + 10);
v_preNormGoal_1457_ = lean_ctor_get(v___x_1449_, 5);
v_normalizationState_1458_ = lean_ctor_get(v___x_1449_, 6);
v_mvars_1459_ = lean_ctor_get(v___x_1449_, 7);
v_forwardState_1460_ = lean_ctor_get(v___x_1449_, 8);
v_forwardRuleMatches_1461_ = lean_ctor_get(v___x_1449_, 9);
v_successProbability_1462_ = lean_ctor_get_float(v___x_1449_, sizeof(void*)*14);
v_addedInIteration_1463_ = lean_ctor_get(v___x_1449_, 10);
v_lastExpandedInIteration_1464_ = lean_ctor_get(v___x_1449_, 11);
v_unsafeRulesSelected_1465_ = lean_ctor_get_uint8(v___x_1449_, sizeof(void*)*14 + 11);
v_unsafeQueue_1466_ = lean_ctor_get(v___x_1449_, 12);
v_failedRapps_1467_ = lean_ctor_get(v___x_1449_, 13);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1449_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1469_ = v___x_1449_;
v_isShared_1470_ = v_isSharedCheck_1475_;
goto v_resetjp_1468_;
}
else
{
lean_inc(v_failedRapps_1467_);
lean_inc(v_unsafeQueue_1466_);
lean_inc(v_lastExpandedInIteration_1464_);
lean_inc(v_addedInIteration_1463_);
lean_inc(v_forwardRuleMatches_1461_);
lean_inc(v_forwardState_1460_);
lean_inc(v_mvars_1459_);
lean_inc(v_normalizationState_1458_);
lean_inc(v_preNormGoal_1457_);
lean_inc(v_depth_1454_);
lean_inc(v_origin_1453_);
lean_inc(v_children_1452_);
lean_inc(v_parent_1451_);
lean_inc(v_id_1450_);
lean_dec(v___x_1449_);
v___x_1469_ = lean_box(0);
v_isShared_1470_ = v_isSharedCheck_1475_;
goto v_resetjp_1468_;
}
v_resetjp_1468_:
{
lean_object* v___x_1472_; 
if (v_isShared_1470_ == 0)
{
v___x_1472_ = v___x_1469_;
goto v_reusejp_1471_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v_id_1450_);
lean_ctor_set(v_reuseFailAlloc_1474_, 1, v_parent_1451_);
lean_ctor_set(v_reuseFailAlloc_1474_, 2, v_children_1452_);
lean_ctor_set(v_reuseFailAlloc_1474_, 3, v_origin_1453_);
lean_ctor_set(v_reuseFailAlloc_1474_, 4, v_depth_1454_);
lean_ctor_set(v_reuseFailAlloc_1474_, 5, v_preNormGoal_1457_);
lean_ctor_set(v_reuseFailAlloc_1474_, 6, v_normalizationState_1458_);
lean_ctor_set(v_reuseFailAlloc_1474_, 7, v_mvars_1459_);
lean_ctor_set(v_reuseFailAlloc_1474_, 8, v_forwardState_1460_);
lean_ctor_set(v_reuseFailAlloc_1474_, 9, v_forwardRuleMatches_1461_);
lean_ctor_set(v_reuseFailAlloc_1474_, 10, v_addedInIteration_1463_);
lean_ctor_set(v_reuseFailAlloc_1474_, 11, v_lastExpandedInIteration_1464_);
lean_ctor_set(v_reuseFailAlloc_1474_, 12, v_unsafeQueue_1466_);
lean_ctor_set(v_reuseFailAlloc_1474_, 13, v_failedRapps_1467_);
lean_ctor_set_uint8(v_reuseFailAlloc_1474_, sizeof(void*)*14 + 9, v_isIrrelevant_1455_);
lean_ctor_set_uint8(v_reuseFailAlloc_1474_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1456_);
lean_ctor_set_float(v_reuseFailAlloc_1474_, sizeof(void*)*14, v_successProbability_1462_);
lean_ctor_set_uint8(v_reuseFailAlloc_1474_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1465_);
v___x_1472_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1471_;
}
v_reusejp_1471_:
{
lean_object* v___x_1473_; 
lean_ctor_set_uint8(v___x_1472_, sizeof(void*)*14 + 8, v_state_1444_);
lean_inc(v_introGoal_1447_);
v___x_1473_ = lean_apply_1(v_introGoal_1447_, v___x_1472_);
return v___x_1473_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setState___boxed(lean_object* v_state_1476_, lean_object* v_g_1477_){
_start:
{
uint8_t v_state_boxed_1478_; lean_object* v_res_1479_; 
v_state_boxed_1478_ = lean_unbox(v_state_1476_);
v_res_1479_ = lp_aesop_Aesop_Goal_setState(v_state_boxed_1478_, v_g_1477_);
return v_res_1479_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_setFailedRapps(lean_object* v_failedRapps_1480_, lean_object* v_g_1481_){
_start:
{
lean_object* v___x_1482_; lean_object* v_introGoal_1483_; lean_object* v_elimGoal_1484_; lean_object* v___x_1485_; lean_object* v_id_1486_; lean_object* v_parent_1487_; lean_object* v_children_1488_; lean_object* v_origin_1489_; lean_object* v_depth_1490_; uint8_t v_state_1491_; uint8_t v_isIrrelevant_1492_; uint8_t v_isForcedUnprovable_1493_; lean_object* v_preNormGoal_1494_; lean_object* v_normalizationState_1495_; lean_object* v_mvars_1496_; lean_object* v_forwardState_1497_; lean_object* v_forwardRuleMatches_1498_; double v_successProbability_1499_; lean_object* v_addedInIteration_1500_; lean_object* v_lastExpandedInIteration_1501_; uint8_t v_unsafeRulesSelected_1502_; lean_object* v_unsafeQueue_1503_; lean_object* v___x_1505_; uint8_t v_isShared_1506_; uint8_t v_isSharedCheck_1511_; 
v___x_1482_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introGoal_1483_ = lean_ctor_get(v___x_1482_, 0);
v_elimGoal_1484_ = lean_ctor_get(v___x_1482_, 1);
lean_inc_ref(v_elimGoal_1484_);
v___x_1485_ = lean_apply_1(v_elimGoal_1484_, v_g_1481_);
v_id_1486_ = lean_ctor_get(v___x_1485_, 0);
v_parent_1487_ = lean_ctor_get(v___x_1485_, 1);
v_children_1488_ = lean_ctor_get(v___x_1485_, 2);
v_origin_1489_ = lean_ctor_get(v___x_1485_, 3);
v_depth_1490_ = lean_ctor_get(v___x_1485_, 4);
v_state_1491_ = lean_ctor_get_uint8(v___x_1485_, sizeof(void*)*14 + 8);
v_isIrrelevant_1492_ = lean_ctor_get_uint8(v___x_1485_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1493_ = lean_ctor_get_uint8(v___x_1485_, sizeof(void*)*14 + 10);
v_preNormGoal_1494_ = lean_ctor_get(v___x_1485_, 5);
v_normalizationState_1495_ = lean_ctor_get(v___x_1485_, 6);
v_mvars_1496_ = lean_ctor_get(v___x_1485_, 7);
v_forwardState_1497_ = lean_ctor_get(v___x_1485_, 8);
v_forwardRuleMatches_1498_ = lean_ctor_get(v___x_1485_, 9);
v_successProbability_1499_ = lean_ctor_get_float(v___x_1485_, sizeof(void*)*14);
v_addedInIteration_1500_ = lean_ctor_get(v___x_1485_, 10);
v_lastExpandedInIteration_1501_ = lean_ctor_get(v___x_1485_, 11);
v_unsafeRulesSelected_1502_ = lean_ctor_get_uint8(v___x_1485_, sizeof(void*)*14 + 11);
v_unsafeQueue_1503_ = lean_ctor_get(v___x_1485_, 12);
v_isSharedCheck_1511_ = !lean_is_exclusive(v___x_1485_);
if (v_isSharedCheck_1511_ == 0)
{
lean_object* v_unused_1512_; 
v_unused_1512_ = lean_ctor_get(v___x_1485_, 13);
lean_dec(v_unused_1512_);
v___x_1505_ = v___x_1485_;
v_isShared_1506_ = v_isSharedCheck_1511_;
goto v_resetjp_1504_;
}
else
{
lean_inc(v_unsafeQueue_1503_);
lean_inc(v_lastExpandedInIteration_1501_);
lean_inc(v_addedInIteration_1500_);
lean_inc(v_forwardRuleMatches_1498_);
lean_inc(v_forwardState_1497_);
lean_inc(v_mvars_1496_);
lean_inc(v_normalizationState_1495_);
lean_inc(v_preNormGoal_1494_);
lean_inc(v_depth_1490_);
lean_inc(v_origin_1489_);
lean_inc(v_children_1488_);
lean_inc(v_parent_1487_);
lean_inc(v_id_1486_);
lean_dec(v___x_1485_);
v___x_1505_ = lean_box(0);
v_isShared_1506_ = v_isSharedCheck_1511_;
goto v_resetjp_1504_;
}
v_resetjp_1504_:
{
lean_object* v___x_1508_; 
if (v_isShared_1506_ == 0)
{
lean_ctor_set(v___x_1505_, 13, v_failedRapps_1480_);
v___x_1508_ = v___x_1505_;
goto v_reusejp_1507_;
}
else
{
lean_object* v_reuseFailAlloc_1510_; 
v_reuseFailAlloc_1510_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1510_, 0, v_id_1486_);
lean_ctor_set(v_reuseFailAlloc_1510_, 1, v_parent_1487_);
lean_ctor_set(v_reuseFailAlloc_1510_, 2, v_children_1488_);
lean_ctor_set(v_reuseFailAlloc_1510_, 3, v_origin_1489_);
lean_ctor_set(v_reuseFailAlloc_1510_, 4, v_depth_1490_);
lean_ctor_set(v_reuseFailAlloc_1510_, 5, v_preNormGoal_1494_);
lean_ctor_set(v_reuseFailAlloc_1510_, 6, v_normalizationState_1495_);
lean_ctor_set(v_reuseFailAlloc_1510_, 7, v_mvars_1496_);
lean_ctor_set(v_reuseFailAlloc_1510_, 8, v_forwardState_1497_);
lean_ctor_set(v_reuseFailAlloc_1510_, 9, v_forwardRuleMatches_1498_);
lean_ctor_set(v_reuseFailAlloc_1510_, 10, v_addedInIteration_1500_);
lean_ctor_set(v_reuseFailAlloc_1510_, 11, v_lastExpandedInIteration_1501_);
lean_ctor_set(v_reuseFailAlloc_1510_, 12, v_unsafeQueue_1503_);
lean_ctor_set(v_reuseFailAlloc_1510_, 13, v_failedRapps_1480_);
lean_ctor_set_uint8(v_reuseFailAlloc_1510_, sizeof(void*)*14 + 8, v_state_1491_);
lean_ctor_set_uint8(v_reuseFailAlloc_1510_, sizeof(void*)*14 + 9, v_isIrrelevant_1492_);
lean_ctor_set_uint8(v_reuseFailAlloc_1510_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1493_);
lean_ctor_set_float(v_reuseFailAlloc_1510_, sizeof(void*)*14, v_successProbability_1499_);
lean_ctor_set_uint8(v_reuseFailAlloc_1510_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1502_);
v___x_1508_ = v_reuseFailAlloc_1510_;
goto v_reusejp_1507_;
}
v_reusejp_1507_:
{
lean_object* v___x_1509_; 
lean_inc(v_introGoal_1483_);
v___x_1509_ = lean_apply_1(v_introGoal_1483_, v___x_1508_);
return v___x_1509_;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_instBEq___lam__0(lean_object* v_g_u2081_1513_, lean_object* v_g_u2082_1514_){
_start:
{
lean_object* v___x_1515_; lean_object* v_elimGoal_1516_; lean_object* v___x_1517_; lean_object* v_id_1518_; lean_object* v___x_1519_; lean_object* v_id_1520_; uint8_t v___x_1521_; 
v___x_1515_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_1516_ = lean_ctor_get(v___x_1515_, 1);
lean_inc_ref_n(v_elimGoal_1516_, 2);
v___x_1517_ = lean_apply_1(v_elimGoal_1516_, v_g_u2081_1513_);
v_id_1518_ = lean_ctor_get(v___x_1517_, 0);
lean_inc(v_id_1518_);
lean_dec_ref(v___x_1517_);
v___x_1519_ = lean_apply_1(v_elimGoal_1516_, v_g_u2082_1514_);
v_id_1520_ = lean_ctor_get(v___x_1519_, 0);
lean_inc(v_id_1520_);
lean_dec_ref(v___x_1519_);
v___x_1521_ = lean_nat_dec_eq(v_id_1518_, v_id_1520_);
lean_dec(v_id_1520_);
lean_dec(v_id_1518_);
return v___x_1521_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instBEq___lam__0___boxed(lean_object* v_g_u2081_1522_, lean_object* v_g_u2082_1523_){
_start:
{
uint8_t v_res_1524_; lean_object* v_r_1525_; 
v_res_1524_ = lp_aesop_Aesop_Goal_instBEq___lam__0(v_g_u2081_1522_, v_g_u2082_1523_);
v_r_1525_ = lean_box(v_res_1524_);
return v_r_1525_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Goal_instHashable___lam__0(lean_object* v_g_1528_){
_start:
{
lean_object* v___x_1529_; lean_object* v_elimGoal_1530_; lean_object* v___x_1531_; lean_object* v_id_1532_; uint64_t v___x_1533_; 
v___x_1529_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_1530_ = lean_ctor_get(v___x_1529_, 1);
lean_inc_ref(v_elimGoal_1530_);
v___x_1531_ = lean_apply_1(v_elimGoal_1530_, v_g_1528_);
v_id_1532_ = lean_ctor_get(v___x_1531_, 0);
lean_inc(v_id_1532_);
lean_dec_ref(v___x_1531_);
v___x_1533_ = lean_uint64_of_nat(v_id_1532_);
lean_dec(v_id_1532_);
return v___x_1533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_instHashable___lam__0___boxed(lean_object* v_g_1534_){
_start:
{
uint64_t v_res_1535_; lean_object* v_r_1536_; 
v_res_1535_ = lp_aesop_Aesop_Goal_instHashable___lam__0(v_g_1534_);
v_r_1536_ = lean_box_uint64(v_res_1535_);
return v_r_1536_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_mk(lean_object* v_a_1539_){
_start:
{
lean_object* v___x_1540_; lean_object* v_introRapp_1541_; lean_object* v___x_1542_; 
v___x_1540_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1541_ = lean_ctor_get(v___x_1540_, 2);
lean_inc(v_introRapp_1541_);
v___x_1542_ = lean_apply_1(v_introRapp_1541_, v_a_1539_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_elim(lean_object* v_a_1543_){
_start:
{
lean_object* v___x_1544_; lean_object* v_elimRapp_1545_; lean_object* v___x_1546_; 
v___x_1544_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1545_ = lean_ctor_get(v___x_1544_, 3);
lean_inc_ref(v_elimRapp_1545_);
v___x_1546_ = lean_apply_1(v_elimRapp_1545_, v_a_1543_);
return v___x_1546_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_modify(lean_object* v_f_1547_, lean_object* v_r_1548_){
_start:
{
lean_object* v___x_1549_; lean_object* v_introRapp_1550_; lean_object* v_elimRapp_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; 
v___x_1549_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1550_ = lean_ctor_get(v___x_1549_, 2);
v_elimRapp_1551_ = lean_ctor_get(v___x_1549_, 3);
lean_inc_ref(v_elimRapp_1551_);
v___x_1552_ = lean_apply_1(v_elimRapp_1551_, v_r_1548_);
v___x_1553_ = lean_apply_1(v_f_1547_, v___x_1552_);
lean_inc(v_introRapp_1550_);
v___x_1554_ = lean_apply_1(v_introRapp_1550_, v___x_1553_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_id(lean_object* v_r_1555_){
_start:
{
lean_object* v___x_1556_; lean_object* v_elimRapp_1557_; lean_object* v___x_1558_; lean_object* v_id_1559_; 
v___x_1556_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1557_ = lean_ctor_get(v___x_1556_, 3);
lean_inc_ref(v_elimRapp_1557_);
v___x_1558_ = lean_apply_1(v_elimRapp_1557_, v_r_1555_);
v_id_1559_ = lean_ctor_get(v___x_1558_, 0);
lean_inc(v_id_1559_);
lean_dec_ref(v___x_1558_);
return v_id_1559_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parent(lean_object* v_r_1560_){
_start:
{
lean_object* v___x_1561_; lean_object* v_elimRapp_1562_; lean_object* v___x_1563_; lean_object* v_parent_1564_; 
v___x_1561_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1562_ = lean_ctor_get(v___x_1561_, 3);
lean_inc_ref(v_elimRapp_1562_);
v___x_1563_ = lean_apply_1(v_elimRapp_1562_, v_r_1560_);
v_parent_1564_ = lean_ctor_get(v___x_1563_, 1);
lean_inc(v_parent_1564_);
lean_dec_ref(v___x_1563_);
return v_parent_1564_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_children(lean_object* v_r_1565_){
_start:
{
lean_object* v___x_1566_; lean_object* v_elimRapp_1567_; lean_object* v___x_1568_; lean_object* v_children_1569_; 
v___x_1566_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1567_ = lean_ctor_get(v___x_1566_, 3);
lean_inc_ref(v_elimRapp_1567_);
v___x_1568_ = lean_apply_1(v_elimRapp_1567_, v_r_1565_);
v_children_1569_ = lean_ctor_get(v___x_1568_, 2);
lean_inc_ref(v_children_1569_);
lean_dec_ref(v___x_1568_);
return v_children_1569_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_state(lean_object* v_r_1570_){
_start:
{
lean_object* v___x_1571_; lean_object* v_elimRapp_1572_; lean_object* v___x_1573_; uint8_t v_state_1574_; 
v___x_1571_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1572_ = lean_ctor_get(v___x_1571_, 3);
lean_inc_ref(v_elimRapp_1572_);
v___x_1573_ = lean_apply_1(v_elimRapp_1572_, v_r_1570_);
v_state_1574_ = lean_ctor_get_uint8(v___x_1573_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_1573_);
return v_state_1574_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_state___boxed(lean_object* v_r_1575_){
_start:
{
uint8_t v_res_1576_; lean_object* v_r_1577_; 
v_res_1576_ = lp_aesop_Aesop_Rapp_state(v_r_1575_);
v_r_1577_ = lean_box(v_res_1576_);
return v_r_1577_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevant(lean_object* v_r_1578_){
_start:
{
lean_object* v___x_1579_; lean_object* v_elimRapp_1580_; lean_object* v___x_1581_; uint8_t v_isIrrelevant_1582_; 
v___x_1579_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1580_ = lean_ctor_get(v___x_1579_, 3);
lean_inc_ref(v_elimRapp_1580_);
v___x_1581_ = lean_apply_1(v_elimRapp_1580_, v_r_1578_);
v_isIrrelevant_1582_ = lean_ctor_get_uint8(v___x_1581_, sizeof(void*)*9 + 9);
lean_dec_ref(v___x_1581_);
return v_isIrrelevant_1582_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevant___boxed(lean_object* v_r_1583_){
_start:
{
uint8_t v_res_1584_; lean_object* v_r_1585_; 
v_res_1584_ = lp_aesop_Aesop_Rapp_isIrrelevant(v_r_1583_);
v_r_1585_ = lean_box(v_res_1584_);
return v_r_1585_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_appliedRule(lean_object* v_r_1586_){
_start:
{
lean_object* v___x_1587_; lean_object* v_elimRapp_1588_; lean_object* v___x_1589_; lean_object* v_appliedRule_1590_; 
v___x_1587_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1588_ = lean_ctor_get(v___x_1587_, 3);
lean_inc_ref(v_elimRapp_1588_);
v___x_1589_ = lean_apply_1(v_elimRapp_1588_, v_r_1586_);
v_appliedRule_1590_ = lean_ctor_get(v___x_1589_, 3);
lean_inc_ref(v_appliedRule_1590_);
lean_dec_ref(v___x_1589_);
return v_appliedRule_1590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_scriptSteps_x3f(lean_object* v_r_1591_){
_start:
{
lean_object* v___x_1592_; lean_object* v_elimRapp_1593_; lean_object* v___x_1594_; lean_object* v_scriptSteps_x3f_1595_; 
v___x_1592_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1593_ = lean_ctor_get(v___x_1592_, 3);
lean_inc_ref(v_elimRapp_1593_);
v___x_1594_ = lean_apply_1(v_elimRapp_1593_, v_r_1591_);
v_scriptSteps_x3f_1595_ = lean_ctor_get(v___x_1594_, 4);
lean_inc(v_scriptSteps_x3f_1595_);
lean_dec_ref(v___x_1594_);
return v_scriptSteps_x3f_1595_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_originalSubgoals(lean_object* v_r_1596_){
_start:
{
lean_object* v___x_1597_; lean_object* v_elimRapp_1598_; lean_object* v___x_1599_; lean_object* v_originalSubgoals_1600_; 
v___x_1597_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1598_ = lean_ctor_get(v___x_1597_, 3);
lean_inc_ref(v_elimRapp_1598_);
v___x_1599_ = lean_apply_1(v_elimRapp_1598_, v_r_1596_);
v_originalSubgoals_1600_ = lean_ctor_get(v___x_1599_, 5);
lean_inc_ref(v_originalSubgoals_1600_);
lean_dec_ref(v___x_1599_);
return v_originalSubgoals_1600_;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Rapp_successProbability(lean_object* v_r_1601_){
_start:
{
lean_object* v___x_1602_; lean_object* v_elimRapp_1603_; lean_object* v___x_1604_; double v_successProbability_1605_; 
v___x_1602_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1603_ = lean_ctor_get(v___x_1602_, 3);
lean_inc_ref(v_elimRapp_1603_);
v___x_1604_ = lean_apply_1(v_elimRapp_1603_, v_r_1601_);
v_successProbability_1605_ = lean_ctor_get_float(v___x_1604_, sizeof(void*)*9);
lean_dec_ref(v___x_1604_);
return v_successProbability_1605_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_successProbability___boxed(lean_object* v_r_1606_){
_start:
{
double v_res_1607_; lean_object* v_r_1608_; 
v_res_1607_ = lp_aesop_Aesop_Rapp_successProbability(v_r_1606_);
v_r_1608_ = lean_box_float(v_res_1607_);
return v_r_1608_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_metaState(lean_object* v_r_1609_){
_start:
{
lean_object* v___x_1610_; lean_object* v_elimRapp_1611_; lean_object* v___x_1612_; lean_object* v_metaState_1613_; 
v___x_1610_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1611_ = lean_ctor_get(v___x_1610_, 3);
lean_inc_ref(v_elimRapp_1611_);
v___x_1612_ = lean_apply_1(v_elimRapp_1611_, v_r_1609_);
v_metaState_1613_ = lean_ctor_get(v___x_1612_, 6);
lean_inc_ref(v_metaState_1613_);
lean_dec_ref(v___x_1612_);
return v_metaState_1613_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducedMVars(lean_object* v_r_1614_){
_start:
{
lean_object* v___x_1615_; lean_object* v_elimRapp_1616_; lean_object* v___x_1617_; lean_object* v_introducedMVars_1618_; 
v___x_1615_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1616_ = lean_ctor_get(v___x_1615_, 3);
lean_inc_ref(v_elimRapp_1616_);
v___x_1617_ = lean_apply_1(v_elimRapp_1616_, v_r_1614_);
v_introducedMVars_1618_ = lean_ctor_get(v___x_1617_, 7);
lean_inc_ref(v_introducedMVars_1618_);
lean_dec_ref(v___x_1617_);
return v_introducedMVars_1618_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_assignedMVars(lean_object* v_r_1619_){
_start:
{
lean_object* v___x_1620_; lean_object* v_elimRapp_1621_; lean_object* v___x_1622_; lean_object* v_assignedMVars_1623_; 
v___x_1620_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1621_ = lean_ctor_get(v___x_1620_, 3);
lean_inc_ref(v_elimRapp_1621_);
v___x_1622_ = lean_apply_1(v_elimRapp_1621_, v_r_1619_);
v_assignedMVars_1623_ = lean_ctor_get(v___x_1622_, 8);
lean_inc_ref(v_assignedMVars_1623_);
lean_dec_ref(v___x_1622_);
return v_assignedMVars_1623_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setId(lean_object* v_id_1624_, lean_object* v_r_1625_){
_start:
{
lean_object* v___x_1626_; lean_object* v_introRapp_1627_; lean_object* v_elimRapp_1628_; lean_object* v___x_1629_; lean_object* v_parent_1630_; lean_object* v_children_1631_; uint8_t v_state_1632_; uint8_t v_isIrrelevant_1633_; lean_object* v_appliedRule_1634_; lean_object* v_scriptSteps_x3f_1635_; lean_object* v_originalSubgoals_1636_; double v_successProbability_1637_; lean_object* v_metaState_1638_; lean_object* v_introducedMVars_1639_; lean_object* v_assignedMVars_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_1648_; 
v___x_1626_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1627_ = lean_ctor_get(v___x_1626_, 2);
v_elimRapp_1628_ = lean_ctor_get(v___x_1626_, 3);
lean_inc_ref(v_elimRapp_1628_);
v___x_1629_ = lean_apply_1(v_elimRapp_1628_, v_r_1625_);
v_parent_1630_ = lean_ctor_get(v___x_1629_, 1);
v_children_1631_ = lean_ctor_get(v___x_1629_, 2);
v_state_1632_ = lean_ctor_get_uint8(v___x_1629_, sizeof(void*)*9 + 8);
v_isIrrelevant_1633_ = lean_ctor_get_uint8(v___x_1629_, sizeof(void*)*9 + 9);
v_appliedRule_1634_ = lean_ctor_get(v___x_1629_, 3);
v_scriptSteps_x3f_1635_ = lean_ctor_get(v___x_1629_, 4);
v_originalSubgoals_1636_ = lean_ctor_get(v___x_1629_, 5);
v_successProbability_1637_ = lean_ctor_get_float(v___x_1629_, sizeof(void*)*9);
v_metaState_1638_ = lean_ctor_get(v___x_1629_, 6);
v_introducedMVars_1639_ = lean_ctor_get(v___x_1629_, 7);
v_assignedMVars_1640_ = lean_ctor_get(v___x_1629_, 8);
v_isSharedCheck_1648_ = !lean_is_exclusive(v___x_1629_);
if (v_isSharedCheck_1648_ == 0)
{
lean_object* v_unused_1649_; 
v_unused_1649_ = lean_ctor_get(v___x_1629_, 0);
lean_dec(v_unused_1649_);
v___x_1642_ = v___x_1629_;
v_isShared_1643_ = v_isSharedCheck_1648_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_assignedMVars_1640_);
lean_inc(v_introducedMVars_1639_);
lean_inc(v_metaState_1638_);
lean_inc(v_originalSubgoals_1636_);
lean_inc(v_scriptSteps_x3f_1635_);
lean_inc(v_appliedRule_1634_);
lean_inc(v_children_1631_);
lean_inc(v_parent_1630_);
lean_dec(v___x_1629_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_1648_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
lean_object* v___x_1645_; 
if (v_isShared_1643_ == 0)
{
lean_ctor_set(v___x_1642_, 0, v_id_1624_);
v___x_1645_ = v___x_1642_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_id_1624_);
lean_ctor_set(v_reuseFailAlloc_1647_, 1, v_parent_1630_);
lean_ctor_set(v_reuseFailAlloc_1647_, 2, v_children_1631_);
lean_ctor_set(v_reuseFailAlloc_1647_, 3, v_appliedRule_1634_);
lean_ctor_set(v_reuseFailAlloc_1647_, 4, v_scriptSteps_x3f_1635_);
lean_ctor_set(v_reuseFailAlloc_1647_, 5, v_originalSubgoals_1636_);
lean_ctor_set(v_reuseFailAlloc_1647_, 6, v_metaState_1638_);
lean_ctor_set(v_reuseFailAlloc_1647_, 7, v_introducedMVars_1639_);
lean_ctor_set(v_reuseFailAlloc_1647_, 8, v_assignedMVars_1640_);
lean_ctor_set_uint8(v_reuseFailAlloc_1647_, sizeof(void*)*9 + 8, v_state_1632_);
lean_ctor_set_uint8(v_reuseFailAlloc_1647_, sizeof(void*)*9 + 9, v_isIrrelevant_1633_);
lean_ctor_set_float(v_reuseFailAlloc_1647_, sizeof(void*)*9, v_successProbability_1637_);
v___x_1645_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1644_;
}
v_reusejp_1644_:
{
lean_object* v___x_1646_; 
lean_inc(v_introRapp_1627_);
v___x_1646_ = lean_apply_1(v_introRapp_1627_, v___x_1645_);
return v___x_1646_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setParent(lean_object* v_parent_1650_, lean_object* v_r_1651_){
_start:
{
lean_object* v___x_1652_; lean_object* v_introRapp_1653_; lean_object* v_elimRapp_1654_; lean_object* v___x_1655_; lean_object* v_id_1656_; lean_object* v_children_1657_; uint8_t v_state_1658_; uint8_t v_isIrrelevant_1659_; lean_object* v_appliedRule_1660_; lean_object* v_scriptSteps_x3f_1661_; lean_object* v_originalSubgoals_1662_; double v_successProbability_1663_; lean_object* v_metaState_1664_; lean_object* v_introducedMVars_1665_; lean_object* v_assignedMVars_1666_; lean_object* v___x_1668_; uint8_t v_isShared_1669_; uint8_t v_isSharedCheck_1674_; 
v___x_1652_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1653_ = lean_ctor_get(v___x_1652_, 2);
v_elimRapp_1654_ = lean_ctor_get(v___x_1652_, 3);
lean_inc_ref(v_elimRapp_1654_);
v___x_1655_ = lean_apply_1(v_elimRapp_1654_, v_r_1651_);
v_id_1656_ = lean_ctor_get(v___x_1655_, 0);
v_children_1657_ = lean_ctor_get(v___x_1655_, 2);
v_state_1658_ = lean_ctor_get_uint8(v___x_1655_, sizeof(void*)*9 + 8);
v_isIrrelevant_1659_ = lean_ctor_get_uint8(v___x_1655_, sizeof(void*)*9 + 9);
v_appliedRule_1660_ = lean_ctor_get(v___x_1655_, 3);
v_scriptSteps_x3f_1661_ = lean_ctor_get(v___x_1655_, 4);
v_originalSubgoals_1662_ = lean_ctor_get(v___x_1655_, 5);
v_successProbability_1663_ = lean_ctor_get_float(v___x_1655_, sizeof(void*)*9);
v_metaState_1664_ = lean_ctor_get(v___x_1655_, 6);
v_introducedMVars_1665_ = lean_ctor_get(v___x_1655_, 7);
v_assignedMVars_1666_ = lean_ctor_get(v___x_1655_, 8);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1655_);
if (v_isSharedCheck_1674_ == 0)
{
lean_object* v_unused_1675_; 
v_unused_1675_ = lean_ctor_get(v___x_1655_, 1);
lean_dec(v_unused_1675_);
v___x_1668_ = v___x_1655_;
v_isShared_1669_ = v_isSharedCheck_1674_;
goto v_resetjp_1667_;
}
else
{
lean_inc(v_assignedMVars_1666_);
lean_inc(v_introducedMVars_1665_);
lean_inc(v_metaState_1664_);
lean_inc(v_originalSubgoals_1662_);
lean_inc(v_scriptSteps_x3f_1661_);
lean_inc(v_appliedRule_1660_);
lean_inc(v_children_1657_);
lean_inc(v_id_1656_);
lean_dec(v___x_1655_);
v___x_1668_ = lean_box(0);
v_isShared_1669_ = v_isSharedCheck_1674_;
goto v_resetjp_1667_;
}
v_resetjp_1667_:
{
lean_object* v___x_1671_; 
if (v_isShared_1669_ == 0)
{
lean_ctor_set(v___x_1668_, 1, v_parent_1650_);
v___x_1671_ = v___x_1668_;
goto v_reusejp_1670_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_id_1656_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v_parent_1650_);
lean_ctor_set(v_reuseFailAlloc_1673_, 2, v_children_1657_);
lean_ctor_set(v_reuseFailAlloc_1673_, 3, v_appliedRule_1660_);
lean_ctor_set(v_reuseFailAlloc_1673_, 4, v_scriptSteps_x3f_1661_);
lean_ctor_set(v_reuseFailAlloc_1673_, 5, v_originalSubgoals_1662_);
lean_ctor_set(v_reuseFailAlloc_1673_, 6, v_metaState_1664_);
lean_ctor_set(v_reuseFailAlloc_1673_, 7, v_introducedMVars_1665_);
lean_ctor_set(v_reuseFailAlloc_1673_, 8, v_assignedMVars_1666_);
lean_ctor_set_uint8(v_reuseFailAlloc_1673_, sizeof(void*)*9 + 8, v_state_1658_);
lean_ctor_set_uint8(v_reuseFailAlloc_1673_, sizeof(void*)*9 + 9, v_isIrrelevant_1659_);
lean_ctor_set_float(v_reuseFailAlloc_1673_, sizeof(void*)*9, v_successProbability_1663_);
v___x_1671_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1670_;
}
v_reusejp_1670_:
{
lean_object* v___x_1672_; 
lean_inc(v_introRapp_1653_);
v___x_1672_ = lean_apply_1(v_introRapp_1653_, v___x_1671_);
return v___x_1672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setChildren(lean_object* v_children_1676_, lean_object* v_r_1677_){
_start:
{
lean_object* v___x_1678_; lean_object* v_introRapp_1679_; lean_object* v_elimRapp_1680_; lean_object* v___x_1681_; lean_object* v_id_1682_; lean_object* v_parent_1683_; uint8_t v_state_1684_; uint8_t v_isIrrelevant_1685_; lean_object* v_appliedRule_1686_; lean_object* v_scriptSteps_x3f_1687_; lean_object* v_originalSubgoals_1688_; double v_successProbability_1689_; lean_object* v_metaState_1690_; lean_object* v_introducedMVars_1691_; lean_object* v_assignedMVars_1692_; lean_object* v___x_1694_; uint8_t v_isShared_1695_; uint8_t v_isSharedCheck_1700_; 
v___x_1678_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1679_ = lean_ctor_get(v___x_1678_, 2);
v_elimRapp_1680_ = lean_ctor_get(v___x_1678_, 3);
lean_inc_ref(v_elimRapp_1680_);
v___x_1681_ = lean_apply_1(v_elimRapp_1680_, v_r_1677_);
v_id_1682_ = lean_ctor_get(v___x_1681_, 0);
v_parent_1683_ = lean_ctor_get(v___x_1681_, 1);
v_state_1684_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*9 + 8);
v_isIrrelevant_1685_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*9 + 9);
v_appliedRule_1686_ = lean_ctor_get(v___x_1681_, 3);
v_scriptSteps_x3f_1687_ = lean_ctor_get(v___x_1681_, 4);
v_originalSubgoals_1688_ = lean_ctor_get(v___x_1681_, 5);
v_successProbability_1689_ = lean_ctor_get_float(v___x_1681_, sizeof(void*)*9);
v_metaState_1690_ = lean_ctor_get(v___x_1681_, 6);
v_introducedMVars_1691_ = lean_ctor_get(v___x_1681_, 7);
v_assignedMVars_1692_ = lean_ctor_get(v___x_1681_, 8);
v_isSharedCheck_1700_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1700_ == 0)
{
lean_object* v_unused_1701_; 
v_unused_1701_ = lean_ctor_get(v___x_1681_, 2);
lean_dec(v_unused_1701_);
v___x_1694_ = v___x_1681_;
v_isShared_1695_ = v_isSharedCheck_1700_;
goto v_resetjp_1693_;
}
else
{
lean_inc(v_assignedMVars_1692_);
lean_inc(v_introducedMVars_1691_);
lean_inc(v_metaState_1690_);
lean_inc(v_originalSubgoals_1688_);
lean_inc(v_scriptSteps_x3f_1687_);
lean_inc(v_appliedRule_1686_);
lean_inc(v_parent_1683_);
lean_inc(v_id_1682_);
lean_dec(v___x_1681_);
v___x_1694_ = lean_box(0);
v_isShared_1695_ = v_isSharedCheck_1700_;
goto v_resetjp_1693_;
}
v_resetjp_1693_:
{
lean_object* v___x_1697_; 
if (v_isShared_1695_ == 0)
{
lean_ctor_set(v___x_1694_, 2, v_children_1676_);
v___x_1697_ = v___x_1694_;
goto v_reusejp_1696_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v_id_1682_);
lean_ctor_set(v_reuseFailAlloc_1699_, 1, v_parent_1683_);
lean_ctor_set(v_reuseFailAlloc_1699_, 2, v_children_1676_);
lean_ctor_set(v_reuseFailAlloc_1699_, 3, v_appliedRule_1686_);
lean_ctor_set(v_reuseFailAlloc_1699_, 4, v_scriptSteps_x3f_1687_);
lean_ctor_set(v_reuseFailAlloc_1699_, 5, v_originalSubgoals_1688_);
lean_ctor_set(v_reuseFailAlloc_1699_, 6, v_metaState_1690_);
lean_ctor_set(v_reuseFailAlloc_1699_, 7, v_introducedMVars_1691_);
lean_ctor_set(v_reuseFailAlloc_1699_, 8, v_assignedMVars_1692_);
lean_ctor_set_uint8(v_reuseFailAlloc_1699_, sizeof(void*)*9 + 8, v_state_1684_);
lean_ctor_set_uint8(v_reuseFailAlloc_1699_, sizeof(void*)*9 + 9, v_isIrrelevant_1685_);
lean_ctor_set_float(v_reuseFailAlloc_1699_, sizeof(void*)*9, v_successProbability_1689_);
v___x_1697_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1696_;
}
v_reusejp_1696_:
{
lean_object* v___x_1698_; 
lean_inc(v_introRapp_1679_);
v___x_1698_ = lean_apply_1(v_introRapp_1679_, v___x_1697_);
return v___x_1698_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState(uint8_t v_state_1702_, lean_object* v_r_1703_){
_start:
{
lean_object* v___x_1704_; lean_object* v_introRapp_1705_; lean_object* v_elimRapp_1706_; lean_object* v___x_1707_; lean_object* v_id_1708_; lean_object* v_parent_1709_; lean_object* v_children_1710_; uint8_t v_isIrrelevant_1711_; lean_object* v_appliedRule_1712_; lean_object* v_scriptSteps_x3f_1713_; lean_object* v_originalSubgoals_1714_; double v_successProbability_1715_; lean_object* v_metaState_1716_; lean_object* v_introducedMVars_1717_; lean_object* v_assignedMVars_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1726_; 
v___x_1704_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1705_ = lean_ctor_get(v___x_1704_, 2);
v_elimRapp_1706_ = lean_ctor_get(v___x_1704_, 3);
lean_inc_ref(v_elimRapp_1706_);
v___x_1707_ = lean_apply_1(v_elimRapp_1706_, v_r_1703_);
v_id_1708_ = lean_ctor_get(v___x_1707_, 0);
v_parent_1709_ = lean_ctor_get(v___x_1707_, 1);
v_children_1710_ = lean_ctor_get(v___x_1707_, 2);
v_isIrrelevant_1711_ = lean_ctor_get_uint8(v___x_1707_, sizeof(void*)*9 + 9);
v_appliedRule_1712_ = lean_ctor_get(v___x_1707_, 3);
v_scriptSteps_x3f_1713_ = lean_ctor_get(v___x_1707_, 4);
v_originalSubgoals_1714_ = lean_ctor_get(v___x_1707_, 5);
v_successProbability_1715_ = lean_ctor_get_float(v___x_1707_, sizeof(void*)*9);
v_metaState_1716_ = lean_ctor_get(v___x_1707_, 6);
v_introducedMVars_1717_ = lean_ctor_get(v___x_1707_, 7);
v_assignedMVars_1718_ = lean_ctor_get(v___x_1707_, 8);
v_isSharedCheck_1726_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1726_ == 0)
{
v___x_1720_ = v___x_1707_;
v_isShared_1721_ = v_isSharedCheck_1726_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_assignedMVars_1718_);
lean_inc(v_introducedMVars_1717_);
lean_inc(v_metaState_1716_);
lean_inc(v_originalSubgoals_1714_);
lean_inc(v_scriptSteps_x3f_1713_);
lean_inc(v_appliedRule_1712_);
lean_inc(v_children_1710_);
lean_inc(v_parent_1709_);
lean_inc(v_id_1708_);
lean_dec(v___x_1707_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1726_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
lean_object* v___x_1723_; 
if (v_isShared_1721_ == 0)
{
v___x_1723_ = v___x_1720_;
goto v_reusejp_1722_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v_id_1708_);
lean_ctor_set(v_reuseFailAlloc_1725_, 1, v_parent_1709_);
lean_ctor_set(v_reuseFailAlloc_1725_, 2, v_children_1710_);
lean_ctor_set(v_reuseFailAlloc_1725_, 3, v_appliedRule_1712_);
lean_ctor_set(v_reuseFailAlloc_1725_, 4, v_scriptSteps_x3f_1713_);
lean_ctor_set(v_reuseFailAlloc_1725_, 5, v_originalSubgoals_1714_);
lean_ctor_set(v_reuseFailAlloc_1725_, 6, v_metaState_1716_);
lean_ctor_set(v_reuseFailAlloc_1725_, 7, v_introducedMVars_1717_);
lean_ctor_set(v_reuseFailAlloc_1725_, 8, v_assignedMVars_1718_);
lean_ctor_set_uint8(v_reuseFailAlloc_1725_, sizeof(void*)*9 + 9, v_isIrrelevant_1711_);
lean_ctor_set_float(v_reuseFailAlloc_1725_, sizeof(void*)*9, v_successProbability_1715_);
v___x_1723_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1722_;
}
v_reusejp_1722_:
{
lean_object* v___x_1724_; 
lean_ctor_set_uint8(v___x_1723_, sizeof(void*)*9 + 8, v_state_1702_);
lean_inc(v_introRapp_1705_);
v___x_1724_ = lean_apply_1(v_introRapp_1705_, v___x_1723_);
return v___x_1724_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setState___boxed(lean_object* v_state_1727_, lean_object* v_r_1728_){
_start:
{
uint8_t v_state_boxed_1729_; lean_object* v_res_1730_; 
v_state_boxed_1729_ = lean_unbox(v_state_1727_);
v_res_1730_ = lp_aesop_Aesop_Rapp_setState(v_state_boxed_1729_, v_r_1728_);
return v_res_1730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant(uint8_t v_isIrrelevant_1731_, lean_object* v_r_1732_){
_start:
{
lean_object* v___x_1733_; lean_object* v_introRapp_1734_; lean_object* v_elimRapp_1735_; lean_object* v___x_1736_; lean_object* v_id_1737_; lean_object* v_parent_1738_; lean_object* v_children_1739_; uint8_t v_state_1740_; lean_object* v_appliedRule_1741_; lean_object* v_scriptSteps_x3f_1742_; lean_object* v_originalSubgoals_1743_; double v_successProbability_1744_; lean_object* v_metaState_1745_; lean_object* v_introducedMVars_1746_; lean_object* v_assignedMVars_1747_; lean_object* v___x_1749_; uint8_t v_isShared_1750_; uint8_t v_isSharedCheck_1755_; 
v___x_1733_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1734_ = lean_ctor_get(v___x_1733_, 2);
v_elimRapp_1735_ = lean_ctor_get(v___x_1733_, 3);
lean_inc_ref(v_elimRapp_1735_);
v___x_1736_ = lean_apply_1(v_elimRapp_1735_, v_r_1732_);
v_id_1737_ = lean_ctor_get(v___x_1736_, 0);
v_parent_1738_ = lean_ctor_get(v___x_1736_, 1);
v_children_1739_ = lean_ctor_get(v___x_1736_, 2);
v_state_1740_ = lean_ctor_get_uint8(v___x_1736_, sizeof(void*)*9 + 8);
v_appliedRule_1741_ = lean_ctor_get(v___x_1736_, 3);
v_scriptSteps_x3f_1742_ = lean_ctor_get(v___x_1736_, 4);
v_originalSubgoals_1743_ = lean_ctor_get(v___x_1736_, 5);
v_successProbability_1744_ = lean_ctor_get_float(v___x_1736_, sizeof(void*)*9);
v_metaState_1745_ = lean_ctor_get(v___x_1736_, 6);
v_introducedMVars_1746_ = lean_ctor_get(v___x_1736_, 7);
v_assignedMVars_1747_ = lean_ctor_get(v___x_1736_, 8);
v_isSharedCheck_1755_ = !lean_is_exclusive(v___x_1736_);
if (v_isSharedCheck_1755_ == 0)
{
v___x_1749_ = v___x_1736_;
v_isShared_1750_ = v_isSharedCheck_1755_;
goto v_resetjp_1748_;
}
else
{
lean_inc(v_assignedMVars_1747_);
lean_inc(v_introducedMVars_1746_);
lean_inc(v_metaState_1745_);
lean_inc(v_originalSubgoals_1743_);
lean_inc(v_scriptSteps_x3f_1742_);
lean_inc(v_appliedRule_1741_);
lean_inc(v_children_1739_);
lean_inc(v_parent_1738_);
lean_inc(v_id_1737_);
lean_dec(v___x_1736_);
v___x_1749_ = lean_box(0);
v_isShared_1750_ = v_isSharedCheck_1755_;
goto v_resetjp_1748_;
}
v_resetjp_1748_:
{
lean_object* v___x_1752_; 
if (v_isShared_1750_ == 0)
{
v___x_1752_ = v___x_1749_;
goto v_reusejp_1751_;
}
else
{
lean_object* v_reuseFailAlloc_1754_; 
v_reuseFailAlloc_1754_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1754_, 0, v_id_1737_);
lean_ctor_set(v_reuseFailAlloc_1754_, 1, v_parent_1738_);
lean_ctor_set(v_reuseFailAlloc_1754_, 2, v_children_1739_);
lean_ctor_set(v_reuseFailAlloc_1754_, 3, v_appliedRule_1741_);
lean_ctor_set(v_reuseFailAlloc_1754_, 4, v_scriptSteps_x3f_1742_);
lean_ctor_set(v_reuseFailAlloc_1754_, 5, v_originalSubgoals_1743_);
lean_ctor_set(v_reuseFailAlloc_1754_, 6, v_metaState_1745_);
lean_ctor_set(v_reuseFailAlloc_1754_, 7, v_introducedMVars_1746_);
lean_ctor_set(v_reuseFailAlloc_1754_, 8, v_assignedMVars_1747_);
lean_ctor_set_uint8(v_reuseFailAlloc_1754_, sizeof(void*)*9 + 8, v_state_1740_);
lean_ctor_set_float(v_reuseFailAlloc_1754_, sizeof(void*)*9, v_successProbability_1744_);
v___x_1752_ = v_reuseFailAlloc_1754_;
goto v_reusejp_1751_;
}
v_reusejp_1751_:
{
lean_object* v___x_1753_; 
lean_ctor_set_uint8(v___x_1752_, sizeof(void*)*9 + 9, v_isIrrelevant_1731_);
lean_inc(v_introRapp_1734_);
v___x_1753_ = lean_apply_1(v_introRapp_1734_, v___x_1752_);
return v___x_1753_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIsIrrelevant___boxed(lean_object* v_isIrrelevant_1756_, lean_object* v_r_1757_){
_start:
{
uint8_t v_isIrrelevant_boxed_1758_; lean_object* v_res_1759_; 
v_isIrrelevant_boxed_1758_ = lean_unbox(v_isIrrelevant_1756_);
v_res_1759_ = lp_aesop_Aesop_Rapp_setIsIrrelevant(v_isIrrelevant_boxed_1758_, v_r_1757_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAppliedRule(lean_object* v_appliedRule_1760_, lean_object* v_r_1761_){
_start:
{
lean_object* v___x_1762_; lean_object* v_introRapp_1763_; lean_object* v_elimRapp_1764_; lean_object* v___x_1765_; lean_object* v_id_1766_; lean_object* v_parent_1767_; lean_object* v_children_1768_; uint8_t v_state_1769_; uint8_t v_isIrrelevant_1770_; lean_object* v_scriptSteps_x3f_1771_; lean_object* v_originalSubgoals_1772_; double v_successProbability_1773_; lean_object* v_metaState_1774_; lean_object* v_introducedMVars_1775_; lean_object* v_assignedMVars_1776_; lean_object* v___x_1778_; uint8_t v_isShared_1779_; uint8_t v_isSharedCheck_1784_; 
v___x_1762_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1763_ = lean_ctor_get(v___x_1762_, 2);
v_elimRapp_1764_ = lean_ctor_get(v___x_1762_, 3);
lean_inc_ref(v_elimRapp_1764_);
v___x_1765_ = lean_apply_1(v_elimRapp_1764_, v_r_1761_);
v_id_1766_ = lean_ctor_get(v___x_1765_, 0);
v_parent_1767_ = lean_ctor_get(v___x_1765_, 1);
v_children_1768_ = lean_ctor_get(v___x_1765_, 2);
v_state_1769_ = lean_ctor_get_uint8(v___x_1765_, sizeof(void*)*9 + 8);
v_isIrrelevant_1770_ = lean_ctor_get_uint8(v___x_1765_, sizeof(void*)*9 + 9);
v_scriptSteps_x3f_1771_ = lean_ctor_get(v___x_1765_, 4);
v_originalSubgoals_1772_ = lean_ctor_get(v___x_1765_, 5);
v_successProbability_1773_ = lean_ctor_get_float(v___x_1765_, sizeof(void*)*9);
v_metaState_1774_ = lean_ctor_get(v___x_1765_, 6);
v_introducedMVars_1775_ = lean_ctor_get(v___x_1765_, 7);
v_assignedMVars_1776_ = lean_ctor_get(v___x_1765_, 8);
v_isSharedCheck_1784_ = !lean_is_exclusive(v___x_1765_);
if (v_isSharedCheck_1784_ == 0)
{
lean_object* v_unused_1785_; 
v_unused_1785_ = lean_ctor_get(v___x_1765_, 3);
lean_dec(v_unused_1785_);
v___x_1778_ = v___x_1765_;
v_isShared_1779_ = v_isSharedCheck_1784_;
goto v_resetjp_1777_;
}
else
{
lean_inc(v_assignedMVars_1776_);
lean_inc(v_introducedMVars_1775_);
lean_inc(v_metaState_1774_);
lean_inc(v_originalSubgoals_1772_);
lean_inc(v_scriptSteps_x3f_1771_);
lean_inc(v_children_1768_);
lean_inc(v_parent_1767_);
lean_inc(v_id_1766_);
lean_dec(v___x_1765_);
v___x_1778_ = lean_box(0);
v_isShared_1779_ = v_isSharedCheck_1784_;
goto v_resetjp_1777_;
}
v_resetjp_1777_:
{
lean_object* v___x_1781_; 
if (v_isShared_1779_ == 0)
{
lean_ctor_set(v___x_1778_, 3, v_appliedRule_1760_);
v___x_1781_ = v___x_1778_;
goto v_reusejp_1780_;
}
else
{
lean_object* v_reuseFailAlloc_1783_; 
v_reuseFailAlloc_1783_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1783_, 0, v_id_1766_);
lean_ctor_set(v_reuseFailAlloc_1783_, 1, v_parent_1767_);
lean_ctor_set(v_reuseFailAlloc_1783_, 2, v_children_1768_);
lean_ctor_set(v_reuseFailAlloc_1783_, 3, v_appliedRule_1760_);
lean_ctor_set(v_reuseFailAlloc_1783_, 4, v_scriptSteps_x3f_1771_);
lean_ctor_set(v_reuseFailAlloc_1783_, 5, v_originalSubgoals_1772_);
lean_ctor_set(v_reuseFailAlloc_1783_, 6, v_metaState_1774_);
lean_ctor_set(v_reuseFailAlloc_1783_, 7, v_introducedMVars_1775_);
lean_ctor_set(v_reuseFailAlloc_1783_, 8, v_assignedMVars_1776_);
lean_ctor_set_uint8(v_reuseFailAlloc_1783_, sizeof(void*)*9 + 8, v_state_1769_);
lean_ctor_set_uint8(v_reuseFailAlloc_1783_, sizeof(void*)*9 + 9, v_isIrrelevant_1770_);
lean_ctor_set_float(v_reuseFailAlloc_1783_, sizeof(void*)*9, v_successProbability_1773_);
v___x_1781_ = v_reuseFailAlloc_1783_;
goto v_reusejp_1780_;
}
v_reusejp_1780_:
{
lean_object* v___x_1782_; 
lean_inc(v_introRapp_1763_);
v___x_1782_ = lean_apply_1(v_introRapp_1763_, v___x_1781_);
return v___x_1782_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setScriptSteps_x3f(lean_object* v_scriptSteps_x3f_1786_, lean_object* v_r_1787_){
_start:
{
lean_object* v___x_1788_; lean_object* v_introRapp_1789_; lean_object* v_elimRapp_1790_; lean_object* v___x_1791_; lean_object* v_id_1792_; lean_object* v_parent_1793_; lean_object* v_children_1794_; uint8_t v_state_1795_; uint8_t v_isIrrelevant_1796_; lean_object* v_appliedRule_1797_; lean_object* v_originalSubgoals_1798_; double v_successProbability_1799_; lean_object* v_metaState_1800_; lean_object* v_introducedMVars_1801_; lean_object* v_assignedMVars_1802_; lean_object* v___x_1804_; uint8_t v_isShared_1805_; uint8_t v_isSharedCheck_1810_; 
v___x_1788_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1789_ = lean_ctor_get(v___x_1788_, 2);
v_elimRapp_1790_ = lean_ctor_get(v___x_1788_, 3);
lean_inc_ref(v_elimRapp_1790_);
v___x_1791_ = lean_apply_1(v_elimRapp_1790_, v_r_1787_);
v_id_1792_ = lean_ctor_get(v___x_1791_, 0);
v_parent_1793_ = lean_ctor_get(v___x_1791_, 1);
v_children_1794_ = lean_ctor_get(v___x_1791_, 2);
v_state_1795_ = lean_ctor_get_uint8(v___x_1791_, sizeof(void*)*9 + 8);
v_isIrrelevant_1796_ = lean_ctor_get_uint8(v___x_1791_, sizeof(void*)*9 + 9);
v_appliedRule_1797_ = lean_ctor_get(v___x_1791_, 3);
v_originalSubgoals_1798_ = lean_ctor_get(v___x_1791_, 5);
v_successProbability_1799_ = lean_ctor_get_float(v___x_1791_, sizeof(void*)*9);
v_metaState_1800_ = lean_ctor_get(v___x_1791_, 6);
v_introducedMVars_1801_ = lean_ctor_get(v___x_1791_, 7);
v_assignedMVars_1802_ = lean_ctor_get(v___x_1791_, 8);
v_isSharedCheck_1810_ = !lean_is_exclusive(v___x_1791_);
if (v_isSharedCheck_1810_ == 0)
{
lean_object* v_unused_1811_; 
v_unused_1811_ = lean_ctor_get(v___x_1791_, 4);
lean_dec(v_unused_1811_);
v___x_1804_ = v___x_1791_;
v_isShared_1805_ = v_isSharedCheck_1810_;
goto v_resetjp_1803_;
}
else
{
lean_inc(v_assignedMVars_1802_);
lean_inc(v_introducedMVars_1801_);
lean_inc(v_metaState_1800_);
lean_inc(v_originalSubgoals_1798_);
lean_inc(v_appliedRule_1797_);
lean_inc(v_children_1794_);
lean_inc(v_parent_1793_);
lean_inc(v_id_1792_);
lean_dec(v___x_1791_);
v___x_1804_ = lean_box(0);
v_isShared_1805_ = v_isSharedCheck_1810_;
goto v_resetjp_1803_;
}
v_resetjp_1803_:
{
lean_object* v___x_1807_; 
if (v_isShared_1805_ == 0)
{
lean_ctor_set(v___x_1804_, 4, v_scriptSteps_x3f_1786_);
v___x_1807_ = v___x_1804_;
goto v_reusejp_1806_;
}
else
{
lean_object* v_reuseFailAlloc_1809_; 
v_reuseFailAlloc_1809_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1809_, 0, v_id_1792_);
lean_ctor_set(v_reuseFailAlloc_1809_, 1, v_parent_1793_);
lean_ctor_set(v_reuseFailAlloc_1809_, 2, v_children_1794_);
lean_ctor_set(v_reuseFailAlloc_1809_, 3, v_appliedRule_1797_);
lean_ctor_set(v_reuseFailAlloc_1809_, 4, v_scriptSteps_x3f_1786_);
lean_ctor_set(v_reuseFailAlloc_1809_, 5, v_originalSubgoals_1798_);
lean_ctor_set(v_reuseFailAlloc_1809_, 6, v_metaState_1800_);
lean_ctor_set(v_reuseFailAlloc_1809_, 7, v_introducedMVars_1801_);
lean_ctor_set(v_reuseFailAlloc_1809_, 8, v_assignedMVars_1802_);
lean_ctor_set_uint8(v_reuseFailAlloc_1809_, sizeof(void*)*9 + 8, v_state_1795_);
lean_ctor_set_uint8(v_reuseFailAlloc_1809_, sizeof(void*)*9 + 9, v_isIrrelevant_1796_);
lean_ctor_set_float(v_reuseFailAlloc_1809_, sizeof(void*)*9, v_successProbability_1799_);
v___x_1807_ = v_reuseFailAlloc_1809_;
goto v_reusejp_1806_;
}
v_reusejp_1806_:
{
lean_object* v___x_1808_; 
lean_inc(v_introRapp_1789_);
v___x_1808_ = lean_apply_1(v_introRapp_1789_, v___x_1807_);
return v___x_1808_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setOriginalSubgoals(lean_object* v_originalSubgoals_1812_, lean_object* v_r_1813_){
_start:
{
lean_object* v___x_1814_; lean_object* v_introRapp_1815_; lean_object* v_elimRapp_1816_; lean_object* v___x_1817_; lean_object* v_id_1818_; lean_object* v_parent_1819_; lean_object* v_children_1820_; uint8_t v_state_1821_; uint8_t v_isIrrelevant_1822_; lean_object* v_appliedRule_1823_; lean_object* v_scriptSteps_x3f_1824_; double v_successProbability_1825_; lean_object* v_metaState_1826_; lean_object* v_introducedMVars_1827_; lean_object* v_assignedMVars_1828_; lean_object* v___x_1830_; uint8_t v_isShared_1831_; uint8_t v_isSharedCheck_1836_; 
v___x_1814_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1815_ = lean_ctor_get(v___x_1814_, 2);
v_elimRapp_1816_ = lean_ctor_get(v___x_1814_, 3);
lean_inc_ref(v_elimRapp_1816_);
v___x_1817_ = lean_apply_1(v_elimRapp_1816_, v_r_1813_);
v_id_1818_ = lean_ctor_get(v___x_1817_, 0);
v_parent_1819_ = lean_ctor_get(v___x_1817_, 1);
v_children_1820_ = lean_ctor_get(v___x_1817_, 2);
v_state_1821_ = lean_ctor_get_uint8(v___x_1817_, sizeof(void*)*9 + 8);
v_isIrrelevant_1822_ = lean_ctor_get_uint8(v___x_1817_, sizeof(void*)*9 + 9);
v_appliedRule_1823_ = lean_ctor_get(v___x_1817_, 3);
v_scriptSteps_x3f_1824_ = lean_ctor_get(v___x_1817_, 4);
v_successProbability_1825_ = lean_ctor_get_float(v___x_1817_, sizeof(void*)*9);
v_metaState_1826_ = lean_ctor_get(v___x_1817_, 6);
v_introducedMVars_1827_ = lean_ctor_get(v___x_1817_, 7);
v_assignedMVars_1828_ = lean_ctor_get(v___x_1817_, 8);
v_isSharedCheck_1836_ = !lean_is_exclusive(v___x_1817_);
if (v_isSharedCheck_1836_ == 0)
{
lean_object* v_unused_1837_; 
v_unused_1837_ = lean_ctor_get(v___x_1817_, 5);
lean_dec(v_unused_1837_);
v___x_1830_ = v___x_1817_;
v_isShared_1831_ = v_isSharedCheck_1836_;
goto v_resetjp_1829_;
}
else
{
lean_inc(v_assignedMVars_1828_);
lean_inc(v_introducedMVars_1827_);
lean_inc(v_metaState_1826_);
lean_inc(v_scriptSteps_x3f_1824_);
lean_inc(v_appliedRule_1823_);
lean_inc(v_children_1820_);
lean_inc(v_parent_1819_);
lean_inc(v_id_1818_);
lean_dec(v___x_1817_);
v___x_1830_ = lean_box(0);
v_isShared_1831_ = v_isSharedCheck_1836_;
goto v_resetjp_1829_;
}
v_resetjp_1829_:
{
lean_object* v___x_1833_; 
if (v_isShared_1831_ == 0)
{
lean_ctor_set(v___x_1830_, 5, v_originalSubgoals_1812_);
v___x_1833_ = v___x_1830_;
goto v_reusejp_1832_;
}
else
{
lean_object* v_reuseFailAlloc_1835_; 
v_reuseFailAlloc_1835_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1835_, 0, v_id_1818_);
lean_ctor_set(v_reuseFailAlloc_1835_, 1, v_parent_1819_);
lean_ctor_set(v_reuseFailAlloc_1835_, 2, v_children_1820_);
lean_ctor_set(v_reuseFailAlloc_1835_, 3, v_appliedRule_1823_);
lean_ctor_set(v_reuseFailAlloc_1835_, 4, v_scriptSteps_x3f_1824_);
lean_ctor_set(v_reuseFailAlloc_1835_, 5, v_originalSubgoals_1812_);
lean_ctor_set(v_reuseFailAlloc_1835_, 6, v_metaState_1826_);
lean_ctor_set(v_reuseFailAlloc_1835_, 7, v_introducedMVars_1827_);
lean_ctor_set(v_reuseFailAlloc_1835_, 8, v_assignedMVars_1828_);
lean_ctor_set_uint8(v_reuseFailAlloc_1835_, sizeof(void*)*9 + 8, v_state_1821_);
lean_ctor_set_uint8(v_reuseFailAlloc_1835_, sizeof(void*)*9 + 9, v_isIrrelevant_1822_);
lean_ctor_set_float(v_reuseFailAlloc_1835_, sizeof(void*)*9, v_successProbability_1825_);
v___x_1833_ = v_reuseFailAlloc_1835_;
goto v_reusejp_1832_;
}
v_reusejp_1832_:
{
lean_object* v___x_1834_; 
lean_inc(v_introRapp_1815_);
v___x_1834_ = lean_apply_1(v_introRapp_1815_, v___x_1833_);
return v___x_1834_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability(double v_successProbability_1838_, lean_object* v_r_1839_){
_start:
{
lean_object* v___x_1840_; lean_object* v_introRapp_1841_; lean_object* v_elimRapp_1842_; lean_object* v___x_1843_; lean_object* v_id_1844_; lean_object* v_parent_1845_; lean_object* v_children_1846_; uint8_t v_state_1847_; uint8_t v_isIrrelevant_1848_; lean_object* v_appliedRule_1849_; lean_object* v_scriptSteps_x3f_1850_; lean_object* v_originalSubgoals_1851_; lean_object* v_metaState_1852_; lean_object* v_introducedMVars_1853_; lean_object* v_assignedMVars_1854_; lean_object* v___x_1856_; uint8_t v_isShared_1857_; uint8_t v_isSharedCheck_1862_; 
v___x_1840_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1841_ = lean_ctor_get(v___x_1840_, 2);
v_elimRapp_1842_ = lean_ctor_get(v___x_1840_, 3);
lean_inc_ref(v_elimRapp_1842_);
v___x_1843_ = lean_apply_1(v_elimRapp_1842_, v_r_1839_);
v_id_1844_ = lean_ctor_get(v___x_1843_, 0);
v_parent_1845_ = lean_ctor_get(v___x_1843_, 1);
v_children_1846_ = lean_ctor_get(v___x_1843_, 2);
v_state_1847_ = lean_ctor_get_uint8(v___x_1843_, sizeof(void*)*9 + 8);
v_isIrrelevant_1848_ = lean_ctor_get_uint8(v___x_1843_, sizeof(void*)*9 + 9);
v_appliedRule_1849_ = lean_ctor_get(v___x_1843_, 3);
v_scriptSteps_x3f_1850_ = lean_ctor_get(v___x_1843_, 4);
v_originalSubgoals_1851_ = lean_ctor_get(v___x_1843_, 5);
v_metaState_1852_ = lean_ctor_get(v___x_1843_, 6);
v_introducedMVars_1853_ = lean_ctor_get(v___x_1843_, 7);
v_assignedMVars_1854_ = lean_ctor_get(v___x_1843_, 8);
v_isSharedCheck_1862_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1862_ == 0)
{
v___x_1856_ = v___x_1843_;
v_isShared_1857_ = v_isSharedCheck_1862_;
goto v_resetjp_1855_;
}
else
{
lean_inc(v_assignedMVars_1854_);
lean_inc(v_introducedMVars_1853_);
lean_inc(v_metaState_1852_);
lean_inc(v_originalSubgoals_1851_);
lean_inc(v_scriptSteps_x3f_1850_);
lean_inc(v_appliedRule_1849_);
lean_inc(v_children_1846_);
lean_inc(v_parent_1845_);
lean_inc(v_id_1844_);
lean_dec(v___x_1843_);
v___x_1856_ = lean_box(0);
v_isShared_1857_ = v_isSharedCheck_1862_;
goto v_resetjp_1855_;
}
v_resetjp_1855_:
{
lean_object* v___x_1859_; 
if (v_isShared_1857_ == 0)
{
v___x_1859_ = v___x_1856_;
goto v_reusejp_1858_;
}
else
{
lean_object* v_reuseFailAlloc_1861_; 
v_reuseFailAlloc_1861_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1861_, 0, v_id_1844_);
lean_ctor_set(v_reuseFailAlloc_1861_, 1, v_parent_1845_);
lean_ctor_set(v_reuseFailAlloc_1861_, 2, v_children_1846_);
lean_ctor_set(v_reuseFailAlloc_1861_, 3, v_appliedRule_1849_);
lean_ctor_set(v_reuseFailAlloc_1861_, 4, v_scriptSteps_x3f_1850_);
lean_ctor_set(v_reuseFailAlloc_1861_, 5, v_originalSubgoals_1851_);
lean_ctor_set(v_reuseFailAlloc_1861_, 6, v_metaState_1852_);
lean_ctor_set(v_reuseFailAlloc_1861_, 7, v_introducedMVars_1853_);
lean_ctor_set(v_reuseFailAlloc_1861_, 8, v_assignedMVars_1854_);
lean_ctor_set_uint8(v_reuseFailAlloc_1861_, sizeof(void*)*9 + 8, v_state_1847_);
lean_ctor_set_uint8(v_reuseFailAlloc_1861_, sizeof(void*)*9 + 9, v_isIrrelevant_1848_);
v___x_1859_ = v_reuseFailAlloc_1861_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
lean_object* v___x_1860_; 
lean_ctor_set_float(v___x_1859_, sizeof(void*)*9, v_successProbability_1838_);
lean_inc(v_introRapp_1841_);
v___x_1860_ = lean_apply_1(v_introRapp_1841_, v___x_1859_);
return v___x_1860_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setSuccessProbability___boxed(lean_object* v_successProbability_1863_, lean_object* v_r_1864_){
_start:
{
double v_successProbability_boxed_1865_; lean_object* v_res_1866_; 
v_successProbability_boxed_1865_ = lean_unbox_float(v_successProbability_1863_);
lean_dec_ref(v_successProbability_1863_);
v_res_1866_ = lp_aesop_Aesop_Rapp_setSuccessProbability(v_successProbability_boxed_1865_, v_r_1864_);
return v_res_1866_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setMetaState(lean_object* v_metaState_1867_, lean_object* v_r_1868_){
_start:
{
lean_object* v___x_1869_; lean_object* v_introRapp_1870_; lean_object* v_elimRapp_1871_; lean_object* v___x_1872_; lean_object* v_id_1873_; lean_object* v_parent_1874_; lean_object* v_children_1875_; uint8_t v_state_1876_; uint8_t v_isIrrelevant_1877_; lean_object* v_appliedRule_1878_; lean_object* v_scriptSteps_x3f_1879_; lean_object* v_originalSubgoals_1880_; double v_successProbability_1881_; lean_object* v_introducedMVars_1882_; lean_object* v_assignedMVars_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1891_; 
v___x_1869_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1870_ = lean_ctor_get(v___x_1869_, 2);
v_elimRapp_1871_ = lean_ctor_get(v___x_1869_, 3);
lean_inc_ref(v_elimRapp_1871_);
v___x_1872_ = lean_apply_1(v_elimRapp_1871_, v_r_1868_);
v_id_1873_ = lean_ctor_get(v___x_1872_, 0);
v_parent_1874_ = lean_ctor_get(v___x_1872_, 1);
v_children_1875_ = lean_ctor_get(v___x_1872_, 2);
v_state_1876_ = lean_ctor_get_uint8(v___x_1872_, sizeof(void*)*9 + 8);
v_isIrrelevant_1877_ = lean_ctor_get_uint8(v___x_1872_, sizeof(void*)*9 + 9);
v_appliedRule_1878_ = lean_ctor_get(v___x_1872_, 3);
v_scriptSteps_x3f_1879_ = lean_ctor_get(v___x_1872_, 4);
v_originalSubgoals_1880_ = lean_ctor_get(v___x_1872_, 5);
v_successProbability_1881_ = lean_ctor_get_float(v___x_1872_, sizeof(void*)*9);
v_introducedMVars_1882_ = lean_ctor_get(v___x_1872_, 7);
v_assignedMVars_1883_ = lean_ctor_get(v___x_1872_, 8);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1872_);
if (v_isSharedCheck_1891_ == 0)
{
lean_object* v_unused_1892_; 
v_unused_1892_ = lean_ctor_get(v___x_1872_, 6);
lean_dec(v_unused_1892_);
v___x_1885_ = v___x_1872_;
v_isShared_1886_ = v_isSharedCheck_1891_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_assignedMVars_1883_);
lean_inc(v_introducedMVars_1882_);
lean_inc(v_originalSubgoals_1880_);
lean_inc(v_scriptSteps_x3f_1879_);
lean_inc(v_appliedRule_1878_);
lean_inc(v_children_1875_);
lean_inc(v_parent_1874_);
lean_inc(v_id_1873_);
lean_dec(v___x_1872_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1891_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
lean_object* v___x_1888_; 
if (v_isShared_1886_ == 0)
{
lean_ctor_set(v___x_1885_, 6, v_metaState_1867_);
v___x_1888_ = v___x_1885_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1890_, 0, v_id_1873_);
lean_ctor_set(v_reuseFailAlloc_1890_, 1, v_parent_1874_);
lean_ctor_set(v_reuseFailAlloc_1890_, 2, v_children_1875_);
lean_ctor_set(v_reuseFailAlloc_1890_, 3, v_appliedRule_1878_);
lean_ctor_set(v_reuseFailAlloc_1890_, 4, v_scriptSteps_x3f_1879_);
lean_ctor_set(v_reuseFailAlloc_1890_, 5, v_originalSubgoals_1880_);
lean_ctor_set(v_reuseFailAlloc_1890_, 6, v_metaState_1867_);
lean_ctor_set(v_reuseFailAlloc_1890_, 7, v_introducedMVars_1882_);
lean_ctor_set(v_reuseFailAlloc_1890_, 8, v_assignedMVars_1883_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, sizeof(void*)*9 + 8, v_state_1876_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, sizeof(void*)*9 + 9, v_isIrrelevant_1877_);
lean_ctor_set_float(v_reuseFailAlloc_1890_, sizeof(void*)*9, v_successProbability_1881_);
v___x_1888_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1887_;
}
v_reusejp_1887_:
{
lean_object* v___x_1889_; 
lean_inc(v_introRapp_1870_);
v___x_1889_ = lean_apply_1(v_introRapp_1870_, v___x_1888_);
return v___x_1889_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setIntroducedMVars(lean_object* v_introducedMVars_1893_, lean_object* v_r_1894_){
_start:
{
lean_object* v___x_1895_; lean_object* v_introRapp_1896_; lean_object* v_elimRapp_1897_; lean_object* v___x_1898_; lean_object* v_id_1899_; lean_object* v_parent_1900_; lean_object* v_children_1901_; uint8_t v_state_1902_; uint8_t v_isIrrelevant_1903_; lean_object* v_appliedRule_1904_; lean_object* v_scriptSteps_x3f_1905_; lean_object* v_originalSubgoals_1906_; double v_successProbability_1907_; lean_object* v_metaState_1908_; lean_object* v_assignedMVars_1909_; lean_object* v___x_1911_; uint8_t v_isShared_1912_; uint8_t v_isSharedCheck_1917_; 
v___x_1895_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1896_ = lean_ctor_get(v___x_1895_, 2);
v_elimRapp_1897_ = lean_ctor_get(v___x_1895_, 3);
lean_inc_ref(v_elimRapp_1897_);
v___x_1898_ = lean_apply_1(v_elimRapp_1897_, v_r_1894_);
v_id_1899_ = lean_ctor_get(v___x_1898_, 0);
v_parent_1900_ = lean_ctor_get(v___x_1898_, 1);
v_children_1901_ = lean_ctor_get(v___x_1898_, 2);
v_state_1902_ = lean_ctor_get_uint8(v___x_1898_, sizeof(void*)*9 + 8);
v_isIrrelevant_1903_ = lean_ctor_get_uint8(v___x_1898_, sizeof(void*)*9 + 9);
v_appliedRule_1904_ = lean_ctor_get(v___x_1898_, 3);
v_scriptSteps_x3f_1905_ = lean_ctor_get(v___x_1898_, 4);
v_originalSubgoals_1906_ = lean_ctor_get(v___x_1898_, 5);
v_successProbability_1907_ = lean_ctor_get_float(v___x_1898_, sizeof(void*)*9);
v_metaState_1908_ = lean_ctor_get(v___x_1898_, 6);
v_assignedMVars_1909_ = lean_ctor_get(v___x_1898_, 8);
v_isSharedCheck_1917_ = !lean_is_exclusive(v___x_1898_);
if (v_isSharedCheck_1917_ == 0)
{
lean_object* v_unused_1918_; 
v_unused_1918_ = lean_ctor_get(v___x_1898_, 7);
lean_dec(v_unused_1918_);
v___x_1911_ = v___x_1898_;
v_isShared_1912_ = v_isSharedCheck_1917_;
goto v_resetjp_1910_;
}
else
{
lean_inc(v_assignedMVars_1909_);
lean_inc(v_metaState_1908_);
lean_inc(v_originalSubgoals_1906_);
lean_inc(v_scriptSteps_x3f_1905_);
lean_inc(v_appliedRule_1904_);
lean_inc(v_children_1901_);
lean_inc(v_parent_1900_);
lean_inc(v_id_1899_);
lean_dec(v___x_1898_);
v___x_1911_ = lean_box(0);
v_isShared_1912_ = v_isSharedCheck_1917_;
goto v_resetjp_1910_;
}
v_resetjp_1910_:
{
lean_object* v___x_1914_; 
if (v_isShared_1912_ == 0)
{
lean_ctor_set(v___x_1911_, 7, v_introducedMVars_1893_);
v___x_1914_ = v___x_1911_;
goto v_reusejp_1913_;
}
else
{
lean_object* v_reuseFailAlloc_1916_; 
v_reuseFailAlloc_1916_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1916_, 0, v_id_1899_);
lean_ctor_set(v_reuseFailAlloc_1916_, 1, v_parent_1900_);
lean_ctor_set(v_reuseFailAlloc_1916_, 2, v_children_1901_);
lean_ctor_set(v_reuseFailAlloc_1916_, 3, v_appliedRule_1904_);
lean_ctor_set(v_reuseFailAlloc_1916_, 4, v_scriptSteps_x3f_1905_);
lean_ctor_set(v_reuseFailAlloc_1916_, 5, v_originalSubgoals_1906_);
lean_ctor_set(v_reuseFailAlloc_1916_, 6, v_metaState_1908_);
lean_ctor_set(v_reuseFailAlloc_1916_, 7, v_introducedMVars_1893_);
lean_ctor_set(v_reuseFailAlloc_1916_, 8, v_assignedMVars_1909_);
lean_ctor_set_uint8(v_reuseFailAlloc_1916_, sizeof(void*)*9 + 8, v_state_1902_);
lean_ctor_set_uint8(v_reuseFailAlloc_1916_, sizeof(void*)*9 + 9, v_isIrrelevant_1903_);
lean_ctor_set_float(v_reuseFailAlloc_1916_, sizeof(void*)*9, v_successProbability_1907_);
v___x_1914_ = v_reuseFailAlloc_1916_;
goto v_reusejp_1913_;
}
v_reusejp_1913_:
{
lean_object* v___x_1915_; 
lean_inc(v_introRapp_1896_);
v___x_1915_ = lean_apply_1(v_introRapp_1896_, v___x_1914_);
return v___x_1915_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_setAssignedMVars(lean_object* v_assignedMVars_1919_, lean_object* v_r_1920_){
_start:
{
lean_object* v___x_1921_; lean_object* v_introRapp_1922_; lean_object* v_elimRapp_1923_; lean_object* v___x_1924_; lean_object* v_id_1925_; lean_object* v_parent_1926_; lean_object* v_children_1927_; uint8_t v_state_1928_; uint8_t v_isIrrelevant_1929_; lean_object* v_appliedRule_1930_; lean_object* v_scriptSteps_x3f_1931_; lean_object* v_originalSubgoals_1932_; double v_successProbability_1933_; lean_object* v_metaState_1934_; lean_object* v_introducedMVars_1935_; lean_object* v___x_1937_; uint8_t v_isShared_1938_; uint8_t v_isSharedCheck_1943_; 
v___x_1921_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_1922_ = lean_ctor_get(v___x_1921_, 2);
v_elimRapp_1923_ = lean_ctor_get(v___x_1921_, 3);
lean_inc_ref(v_elimRapp_1923_);
v___x_1924_ = lean_apply_1(v_elimRapp_1923_, v_r_1920_);
v_id_1925_ = lean_ctor_get(v___x_1924_, 0);
v_parent_1926_ = lean_ctor_get(v___x_1924_, 1);
v_children_1927_ = lean_ctor_get(v___x_1924_, 2);
v_state_1928_ = lean_ctor_get_uint8(v___x_1924_, sizeof(void*)*9 + 8);
v_isIrrelevant_1929_ = lean_ctor_get_uint8(v___x_1924_, sizeof(void*)*9 + 9);
v_appliedRule_1930_ = lean_ctor_get(v___x_1924_, 3);
v_scriptSteps_x3f_1931_ = lean_ctor_get(v___x_1924_, 4);
v_originalSubgoals_1932_ = lean_ctor_get(v___x_1924_, 5);
v_successProbability_1933_ = lean_ctor_get_float(v___x_1924_, sizeof(void*)*9);
v_metaState_1934_ = lean_ctor_get(v___x_1924_, 6);
v_introducedMVars_1935_ = lean_ctor_get(v___x_1924_, 7);
v_isSharedCheck_1943_ = !lean_is_exclusive(v___x_1924_);
if (v_isSharedCheck_1943_ == 0)
{
lean_object* v_unused_1944_; 
v_unused_1944_ = lean_ctor_get(v___x_1924_, 8);
lean_dec(v_unused_1944_);
v___x_1937_ = v___x_1924_;
v_isShared_1938_ = v_isSharedCheck_1943_;
goto v_resetjp_1936_;
}
else
{
lean_inc(v_introducedMVars_1935_);
lean_inc(v_metaState_1934_);
lean_inc(v_originalSubgoals_1932_);
lean_inc(v_scriptSteps_x3f_1931_);
lean_inc(v_appliedRule_1930_);
lean_inc(v_children_1927_);
lean_inc(v_parent_1926_);
lean_inc(v_id_1925_);
lean_dec(v___x_1924_);
v___x_1937_ = lean_box(0);
v_isShared_1938_ = v_isSharedCheck_1943_;
goto v_resetjp_1936_;
}
v_resetjp_1936_:
{
lean_object* v___x_1940_; 
if (v_isShared_1938_ == 0)
{
lean_ctor_set(v___x_1937_, 8, v_assignedMVars_1919_);
v___x_1940_ = v___x_1937_;
goto v_reusejp_1939_;
}
else
{
lean_object* v_reuseFailAlloc_1942_; 
v_reuseFailAlloc_1942_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1942_, 0, v_id_1925_);
lean_ctor_set(v_reuseFailAlloc_1942_, 1, v_parent_1926_);
lean_ctor_set(v_reuseFailAlloc_1942_, 2, v_children_1927_);
lean_ctor_set(v_reuseFailAlloc_1942_, 3, v_appliedRule_1930_);
lean_ctor_set(v_reuseFailAlloc_1942_, 4, v_scriptSteps_x3f_1931_);
lean_ctor_set(v_reuseFailAlloc_1942_, 5, v_originalSubgoals_1932_);
lean_ctor_set(v_reuseFailAlloc_1942_, 6, v_metaState_1934_);
lean_ctor_set(v_reuseFailAlloc_1942_, 7, v_introducedMVars_1935_);
lean_ctor_set(v_reuseFailAlloc_1942_, 8, v_assignedMVars_1919_);
lean_ctor_set_uint8(v_reuseFailAlloc_1942_, sizeof(void*)*9 + 8, v_state_1928_);
lean_ctor_set_uint8(v_reuseFailAlloc_1942_, sizeof(void*)*9 + 9, v_isIrrelevant_1929_);
lean_ctor_set_float(v_reuseFailAlloc_1942_, sizeof(void*)*9, v_successProbability_1933_);
v___x_1940_ = v_reuseFailAlloc_1942_;
goto v_reusejp_1939_;
}
v_reusejp_1939_:
{
lean_object* v___x_1941_; 
lean_inc(v_introRapp_1922_);
v___x_1941_ = lean_apply_1(v_introRapp_1922_, v___x_1940_);
return v___x_1941_;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_instBEq___lam__0(lean_object* v_r_u2081_1945_, lean_object* v_r_u2082_1946_){
_start:
{
lean_object* v___x_1947_; lean_object* v_elimRapp_1948_; lean_object* v___x_1949_; lean_object* v_id_1950_; lean_object* v___x_1951_; lean_object* v_id_1952_; uint8_t v___x_1953_; 
v___x_1947_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1948_ = lean_ctor_get(v___x_1947_, 3);
lean_inc_ref_n(v_elimRapp_1948_, 2);
v___x_1949_ = lean_apply_1(v_elimRapp_1948_, v_r_u2081_1945_);
v_id_1950_ = lean_ctor_get(v___x_1949_, 0);
lean_inc(v_id_1950_);
lean_dec_ref(v___x_1949_);
v___x_1951_ = lean_apply_1(v_elimRapp_1948_, v_r_u2082_1946_);
v_id_1952_ = lean_ctor_get(v___x_1951_, 0);
lean_inc(v_id_1952_);
lean_dec_ref(v___x_1951_);
v___x_1953_ = lean_nat_dec_eq(v_id_1950_, v_id_1952_);
lean_dec(v_id_1952_);
lean_dec(v_id_1950_);
return v___x_1953_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instBEq___lam__0___boxed(lean_object* v_r_u2081_1954_, lean_object* v_r_u2082_1955_){
_start:
{
uint8_t v_res_1956_; lean_object* v_r_1957_; 
v_res_1956_ = lp_aesop_Aesop_Rapp_instBEq___lam__0(v_r_u2081_1954_, v_r_u2082_1955_);
v_r_1957_ = lean_box(v_res_1956_);
return v_r_1957_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rapp_instHashable___lam__0(lean_object* v_r_1960_){
_start:
{
lean_object* v___x_1961_; lean_object* v_elimRapp_1962_; lean_object* v___x_1963_; lean_object* v_id_1964_; uint64_t v___x_1965_; 
v___x_1961_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1962_ = lean_ctor_get(v___x_1961_, 3);
lean_inc_ref(v_elimRapp_1962_);
v___x_1963_ = lean_apply_1(v_elimRapp_1962_, v_r_1960_);
v_id_1964_ = lean_ctor_get(v___x_1963_, 0);
lean_inc(v_id_1964_);
lean_dec_ref(v___x_1963_);
v___x_1965_ = lean_uint64_of_nat(v_id_1964_);
lean_dec(v_id_1964_);
return v___x_1965_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_instHashable___lam__0___boxed(lean_object* v_r_1966_){
_start:
{
uint64_t v_res_1967_; lean_object* v_r_1968_; 
v_res_1967_ = lp_aesop_Aesop_Rapp_instHashable___lam__0(v_r_1966_);
v_r_1968_ = lean_box_uint64(v_res_1967_);
return v_r_1968_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(lean_object* v_s_1971_){
_start:
{
lean_object* v___x_1972_; lean_object* v___x_1973_; uint8_t v___x_1974_; 
v___x_1972_ = lean_array_get_size(v_s_1971_);
v___x_1973_ = lean_unsigned_to_nat(0u);
v___x_1974_ = lean_nat_dec_eq(v___x_1972_, v___x_1973_);
return v___x_1974_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0___boxed(lean_object* v_s_1975_){
_start:
{
uint8_t v_res_1976_; lean_object* v_r_1977_; 
v_res_1976_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(v_s_1975_);
lean_dec_ref(v_s_1975_);
v_r_1977_ = lean_box(v_res_1976_);
return v_r_1977_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isSafe(lean_object* v_r_1978_){
_start:
{
lean_object* v___x_1979_; lean_object* v_elimRapp_1980_; lean_object* v___x_1981_; lean_object* v_appliedRule_1982_; lean_object* v_assignedMVars_1983_; uint8_t v___x_1984_; 
v___x_1979_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_1980_ = lean_ctor_get(v___x_1979_, 3);
lean_inc_ref(v_elimRapp_1980_);
v___x_1981_ = lean_apply_1(v_elimRapp_1980_, v_r_1978_);
v_appliedRule_1982_ = lean_ctor_get(v___x_1981_, 3);
lean_inc_ref(v_appliedRule_1982_);
v_assignedMVars_1983_ = lean_ctor_get(v___x_1981_, 8);
lean_inc_ref(v_assignedMVars_1983_);
lean_dec_ref(v___x_1981_);
v___x_1984_ = lp_aesop_Aesop_RegularRule_isSafe(v_appliedRule_1982_);
lean_dec_ref(v_appliedRule_1982_);
if (v___x_1984_ == 0)
{
lean_dec_ref(v_assignedMVars_1983_);
return v___x_1984_;
}
else
{
uint8_t v___x_1985_; 
v___x_1985_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(v_assignedMVars_1983_);
lean_dec_ref(v_assignedMVars_1983_);
return v___x_1985_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isSafe___boxed(lean_object* v_r_1986_){
_start:
{
uint8_t v_res_1987_; lean_object* v_r_1988_; 
v_res_1987_ = lp_aesop_Aesop_Rapp_isSafe(v_r_1986_);
v_r_1988_ = lean_box(v_res_1987_);
return v_r_1988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoalAndMetaState_x3f(lean_object* v_g_1989_){
_start:
{
lean_object* v___x_1990_; lean_object* v_elimGoal_1991_; lean_object* v___x_1992_; lean_object* v_normalizationState_1993_; 
v___x_1990_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_1991_ = lean_ctor_get(v___x_1990_, 1);
lean_inc_ref(v_elimGoal_1991_);
v___x_1992_ = lean_apply_1(v_elimGoal_1991_, v_g_1989_);
v_normalizationState_1993_ = lean_ctor_get(v___x_1992_, 6);
lean_inc(v_normalizationState_1993_);
lean_dec_ref(v___x_1992_);
if (lean_obj_tag(v_normalizationState_1993_) == 1)
{
lean_object* v_postGoal_1994_; lean_object* v_postState_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; 
v_postGoal_1994_ = lean_ctor_get(v_normalizationState_1993_, 0);
lean_inc(v_postGoal_1994_);
v_postState_1995_ = lean_ctor_get(v_normalizationState_1993_, 1);
lean_inc_ref(v_postState_1995_);
lean_dec_ref_known(v_normalizationState_1993_, 3);
v___x_1996_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1996_, 0, v_postGoal_1994_);
lean_ctor_set(v___x_1996_, 1, v_postState_1995_);
v___x_1997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1997_, 0, v___x_1996_);
return v___x_1997_;
}
else
{
lean_object* v___x_1998_; 
lean_dec(v_normalizationState_1993_);
v___x_1998_ = lean_box(0);
return v___x_1998_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_postNormGoal_x3f(lean_object* v_g_1999_){
_start:
{
lean_object* v___x_2000_; lean_object* v_elimGoal_2001_; lean_object* v___x_2002_; lean_object* v_normalizationState_2003_; 
v___x_2000_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2001_ = lean_ctor_get(v___x_2000_, 1);
lean_inc_ref(v_elimGoal_2001_);
v___x_2002_ = lean_apply_1(v_elimGoal_2001_, v_g_1999_);
v_normalizationState_2003_ = lean_ctor_get(v___x_2002_, 6);
lean_inc(v_normalizationState_2003_);
lean_dec_ref(v___x_2002_);
if (lean_obj_tag(v_normalizationState_2003_) == 1)
{
lean_object* v_postGoal_2004_; lean_object* v___x_2005_; 
v_postGoal_2004_ = lean_ctor_get(v_normalizationState_2003_, 0);
lean_inc(v_postGoal_2004_);
lean_dec_ref_known(v_normalizationState_2003_, 3);
v___x_2005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2005_, 0, v_postGoal_2004_);
return v___x_2005_;
}
else
{
lean_object* v___x_2006_; 
lean_dec(v_normalizationState_2003_);
v___x_2006_ = lean_box(0);
return v___x_2006_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoal(lean_object* v_g_2007_){
_start:
{
lean_object* v___x_2008_; 
lean_inc(v_g_2007_);
v___x_2008_ = lp_aesop_Aesop_Goal_postNormGoal_x3f(v_g_2007_);
if (lean_obj_tag(v___x_2008_) == 0)
{
lean_object* v___x_2009_; lean_object* v_elimGoal_2010_; lean_object* v___x_2011_; lean_object* v_preNormGoal_2012_; 
v___x_2009_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2010_ = lean_ctor_get(v___x_2009_, 1);
lean_inc_ref(v_elimGoal_2010_);
v___x_2011_ = lean_apply_1(v_elimGoal_2010_, v_g_2007_);
v_preNormGoal_2012_ = lean_ctor_get(v___x_2011_, 5);
lean_inc(v_preNormGoal_2012_);
lean_dec_ref(v___x_2011_);
return v_preNormGoal_2012_;
}
else
{
lean_object* v_val_2013_; 
lean_dec(v_g_2007_);
v_val_2013_ = lean_ctor_get(v___x_2008_, 0);
lean_inc(v_val_2013_);
lean_dec_ref_known(v___x_2008_, 1);
return v_val_2013_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object* v_g_2014_){
_start:
{
lean_object* v___x_2016_; lean_object* v_elimGoal_2017_; lean_object* v_elimMVarCluster_2018_; lean_object* v___x_2019_; lean_object* v_parent_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v_parent_x3f_2023_; 
v___x_2016_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2017_ = lean_ctor_get(v___x_2016_, 1);
v_elimMVarCluster_2018_ = lean_ctor_get(v___x_2016_, 5);
lean_inc_ref(v_elimGoal_2017_);
v___x_2019_ = lean_apply_1(v_elimGoal_2017_, v_g_2014_);
v_parent_2020_ = lean_ctor_get(v___x_2019_, 1);
lean_inc(v_parent_2020_);
lean_dec_ref(v___x_2019_);
v___x_2021_ = lean_st_ref_get(v_parent_2020_);
lean_dec(v_parent_2020_);
lean_inc_ref(v_elimMVarCluster_2018_);
v___x_2022_ = lean_apply_1(v_elimMVarCluster_2018_, v___x_2021_);
v_parent_x3f_2023_ = lean_ctor_get(v___x_2022_, 0);
lean_inc(v_parent_x3f_2023_);
lean_dec_ref(v___x_2022_);
return v_parent_x3f_2023_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f___boxed(lean_object* v_g_2024_, lean_object* v_a_2025_){
_start:
{
lean_object* v_res_2026_; 
v_res_2026_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_2024_);
return v_res_2026_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState(lean_object* v_g_2027_, lean_object* v_rootMetaState_2028_){
_start:
{
lean_object* v___x_2030_; 
v___x_2030_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_2027_);
if (lean_obj_tag(v___x_2030_) == 0)
{
lean_inc_ref(v_rootMetaState_2028_);
return v_rootMetaState_2028_;
}
else
{
lean_object* v_val_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v_elimRapp_2034_; lean_object* v___x_2035_; lean_object* v_metaState_2036_; 
v_val_2031_ = lean_ctor_get(v___x_2030_, 0);
lean_inc(v_val_2031_);
lean_dec_ref_known(v___x_2030_, 1);
v___x_2032_ = lean_st_ref_get(v_val_2031_);
lean_dec(v_val_2031_);
v___x_2033_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2034_ = lean_ctor_get(v___x_2033_, 3);
lean_inc_ref(v_elimRapp_2034_);
v___x_2035_ = lean_apply_1(v_elimRapp_2034_, v___x_2032_);
v_metaState_2036_ = lean_ctor_get(v___x_2035_, 6);
lean_inc_ref(v_metaState_2036_);
lean_dec_ref(v___x_2035_);
return v_metaState_2036_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_parentMetaState___boxed(lean_object* v_g_2037_, lean_object* v_rootMetaState_2038_, lean_object* v_a_2039_){
_start:
{
lean_object* v_res_2040_; 
v_res_2040_ = lp_aesop_Aesop_Goal_parentMetaState(v_g_2037_, v_rootMetaState_2038_);
lean_dec_ref(v_rootMetaState_2038_);
return v_res_2040_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object* v_g_2041_, lean_object* v_rootMetaState_2042_){
_start:
{
lean_object* v___x_2044_; lean_object* v_elimGoal_2045_; lean_object* v___x_2046_; lean_object* v_normalizationState_2047_; 
v___x_2044_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2045_ = lean_ctor_get(v___x_2044_, 1);
lean_inc_ref(v_elimGoal_2045_);
lean_inc(v_g_2041_);
v___x_2046_ = lean_apply_1(v_elimGoal_2045_, v_g_2041_);
v_normalizationState_2047_ = lean_ctor_get(v___x_2046_, 6);
lean_inc(v_normalizationState_2047_);
if (lean_obj_tag(v_normalizationState_2047_) == 1)
{
lean_object* v_postGoal_2048_; lean_object* v_postState_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; 
lean_dec_ref(v___x_2046_);
lean_dec(v_g_2041_);
v_postGoal_2048_ = lean_ctor_get(v_normalizationState_2047_, 0);
lean_inc(v_postGoal_2048_);
v_postState_2049_ = lean_ctor_get(v_normalizationState_2047_, 1);
lean_inc_ref(v_postState_2049_);
lean_dec_ref_known(v_normalizationState_2047_, 3);
v___x_2050_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2050_, 0, v_postGoal_2048_);
lean_ctor_set(v___x_2050_, 1, v_postState_2049_);
v___x_2051_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2051_, 0, v___x_2050_);
return v___x_2051_;
}
else
{
lean_object* v_preNormGoal_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; 
lean_dec(v_normalizationState_2047_);
v_preNormGoal_2052_ = lean_ctor_get(v___x_2046_, 5);
lean_inc(v_preNormGoal_2052_);
lean_dec_ref(v___x_2046_);
v___x_2053_ = lp_aesop_Aesop_Goal_parentMetaState(v_g_2041_, v_rootMetaState_2042_);
v___x_2054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2054_, 0, v_preNormGoal_2052_);
lean_ctor_set(v___x_2054_, 1, v___x_2053_);
v___x_2055_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2055_, 0, v___x_2054_);
return v___x_2055_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg___boxed(lean_object* v_g_2056_, lean_object* v_rootMetaState_2057_, lean_object* v_a_2058_){
_start:
{
lean_object* v_res_2059_; 
v_res_2059_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(v_g_2056_, v_rootMetaState_2057_);
lean_dec_ref(v_rootMetaState_2057_);
return v_res_2059_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState(lean_object* v_g_2060_, lean_object* v_rootMetaState_2061_, lean_object* v_a_2062_, lean_object* v_a_2063_, lean_object* v_a_2064_, lean_object* v_a_2065_){
_start:
{
lean_object* v___x_2067_; 
v___x_2067_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(v_g_2060_, v_rootMetaState_2061_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___boxed(lean_object* v_g_2068_, lean_object* v_rootMetaState_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_, lean_object* v_a_2073_, lean_object* v_a_2074_){
_start:
{
lean_object* v_res_2075_; 
v_res_2075_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState(v_g_2068_, v_rootMetaState_2069_, v_a_2070_, v_a_2071_, v_a_2072_, v_a_2073_);
lean_dec(v_a_2073_);
lean_dec_ref(v_a_2072_);
lean_dec(v_a_2071_);
lean_dec_ref(v_a_2070_);
lean_dec_ref(v_rootMetaState_2069_);
return v_res_2075_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(lean_object* v_as_2076_, size_t v_i_2077_, size_t v_stop_2078_, lean_object* v_b_2079_){
_start:
{
uint8_t v___x_2081_; 
v___x_2081_ = lean_usize_dec_eq(v_i_2077_, v_stop_2078_);
if (v___x_2081_ == 0)
{
lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v_val_2085_; uint8_t v___x_2089_; 
v___x_2082_ = lean_array_uget_borrowed(v_as_2076_, v_i_2077_);
v___x_2083_ = lean_st_ref_get(v___x_2082_);
v___x_2089_ = lp_aesop_Aesop_Rapp_isSafe(v___x_2083_);
if (v___x_2089_ == 0)
{
v_val_2085_ = v_b_2079_;
goto v___jp_2084_;
}
else
{
lean_object* v___x_2090_; 
lean_inc(v___x_2082_);
v___x_2090_ = lean_array_push(v_b_2079_, v___x_2082_);
v_val_2085_ = v___x_2090_;
goto v___jp_2084_;
}
v___jp_2084_:
{
size_t v___x_2086_; size_t v___x_2087_; 
v___x_2086_ = ((size_t)1ULL);
v___x_2087_ = lean_usize_add(v_i_2077_, v___x_2086_);
v_i_2077_ = v___x_2087_;
v_b_2079_ = v_val_2085_;
goto _start;
}
}
else
{
return v_b_2079_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0___boxed(lean_object* v_as_2091_, lean_object* v_i_2092_, lean_object* v_stop_2093_, lean_object* v_b_2094_, lean_object* v___y_2095_){
_start:
{
size_t v_i_boxed_2096_; size_t v_stop_boxed_2097_; lean_object* v_res_2098_; 
v_i_boxed_2096_ = lean_unbox_usize(v_i_2092_);
lean_dec(v_i_2092_);
v_stop_boxed_2097_ = lean_unbox_usize(v_stop_2093_);
lean_dec(v_stop_2093_);
v_res_2098_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(v_as_2091_, v_i_boxed_2096_, v_stop_boxed_2097_, v_b_2094_);
lean_dec_ref(v_as_2091_);
return v_res_2098_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object* v_g_2101_){
_start:
{
lean_object* v___x_2103_; lean_object* v_elimGoal_2104_; lean_object* v___x_2105_; lean_object* v_children_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; uint8_t v___x_2110_; 
v___x_2103_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2104_ = lean_ctor_get(v___x_2103_, 1);
lean_inc_ref(v_elimGoal_2104_);
v___x_2105_ = lean_apply_1(v_elimGoal_2104_, v_g_2101_);
v_children_2106_ = lean_ctor_get(v___x_2105_, 2);
lean_inc_ref(v_children_2106_);
lean_dec_ref(v___x_2105_);
v___x_2107_ = lean_unsigned_to_nat(0u);
v___x_2108_ = lean_array_get_size(v_children_2106_);
v___x_2109_ = ((lean_object*)(lp_aesop_Aesop_Goal_safeRapps___closed__0));
v___x_2110_ = lean_nat_dec_lt(v___x_2107_, v___x_2108_);
if (v___x_2110_ == 0)
{
lean_dec_ref(v_children_2106_);
return v___x_2109_;
}
else
{
uint8_t v___x_2111_; 
v___x_2111_ = lean_nat_dec_le(v___x_2108_, v___x_2108_);
if (v___x_2111_ == 0)
{
if (v___x_2110_ == 0)
{
lean_dec_ref(v_children_2106_);
return v___x_2109_;
}
else
{
size_t v___x_2112_; size_t v___x_2113_; lean_object* v___x_2114_; 
v___x_2112_ = ((size_t)0ULL);
v___x_2113_ = lean_usize_of_nat(v___x_2108_);
v___x_2114_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(v_children_2106_, v___x_2112_, v___x_2113_, v___x_2109_);
lean_dec_ref(v_children_2106_);
return v___x_2114_;
}
}
else
{
size_t v___x_2115_; size_t v___x_2116_; lean_object* v___x_2117_; 
v___x_2115_ = ((size_t)0ULL);
v___x_2116_ = lean_usize_of_nat(v___x_2108_);
v___x_2117_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Goal_safeRapps_spec__0(v_children_2106_, v___x_2115_, v___x_2116_, v___x_2109_);
lean_dec_ref(v_children_2106_);
return v___x_2117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_safeRapps___boxed(lean_object* v_g_2118_, lean_object* v_a_2119_){
_start:
{
lean_object* v_res_2120_; 
v_res_2120_ = lp_aesop_Aesop_Goal_safeRapps(v_g_2118_);
return v_res_2120_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(lean_object* v_as_2121_, size_t v_i_2122_, size_t v_stop_2123_){
_start:
{
uint8_t v___x_2125_; 
v___x_2125_ = lean_usize_dec_eq(v_i_2122_, v_stop_2123_);
if (v___x_2125_ == 0)
{
lean_object* v___x_2126_; lean_object* v___x_2127_; uint8_t v___x_2128_; 
v___x_2126_ = lean_array_uget_borrowed(v_as_2121_, v_i_2122_);
v___x_2127_ = lean_st_ref_get(v___x_2126_);
v___x_2128_ = lp_aesop_Aesop_Rapp_isSafe(v___x_2127_);
if (v___x_2128_ == 0)
{
size_t v___x_2129_; size_t v___x_2130_; 
v___x_2129_ = ((size_t)1ULL);
v___x_2130_ = lean_usize_add(v_i_2122_, v___x_2129_);
v_i_2122_ = v___x_2130_;
goto _start;
}
else
{
return v___x_2128_;
}
}
else
{
uint8_t v___x_2132_; 
v___x_2132_ = 0;
return v___x_2132_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0___boxed(lean_object* v_as_2133_, lean_object* v_i_2134_, lean_object* v_stop_2135_, lean_object* v___y_2136_){
_start:
{
size_t v_i_boxed_2137_; size_t v_stop_boxed_2138_; uint8_t v_res_2139_; lean_object* v_r_2140_; 
v_i_boxed_2137_ = lean_unbox_usize(v_i_2134_);
lean_dec(v_i_2134_);
v_stop_boxed_2138_ = lean_unbox_usize(v_stop_2135_);
lean_dec(v_stop_2135_);
v_res_2139_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(v_as_2133_, v_i_boxed_2137_, v_stop_boxed_2138_);
lean_dec_ref(v_as_2133_);
v_r_2140_ = lean_box(v_res_2139_);
return v_r_2140_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasSafeRapp(lean_object* v_g_2141_){
_start:
{
lean_object* v___x_2143_; lean_object* v_elimGoal_2144_; lean_object* v___x_2145_; lean_object* v_children_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; uint8_t v___x_2149_; 
v___x_2143_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2144_ = lean_ctor_get(v___x_2143_, 1);
lean_inc_ref(v_elimGoal_2144_);
v___x_2145_ = lean_apply_1(v_elimGoal_2144_, v_g_2141_);
v_children_2146_ = lean_ctor_get(v___x_2145_, 2);
lean_inc_ref(v_children_2146_);
lean_dec_ref(v___x_2145_);
v___x_2147_ = lean_unsigned_to_nat(0u);
v___x_2148_ = lean_array_get_size(v_children_2146_);
v___x_2149_ = lean_nat_dec_lt(v___x_2147_, v___x_2148_);
if (v___x_2149_ == 0)
{
lean_dec_ref(v_children_2146_);
return v___x_2149_;
}
else
{
if (v___x_2149_ == 0)
{
lean_dec_ref(v_children_2146_);
return v___x_2149_;
}
else
{
size_t v___x_2150_; size_t v___x_2151_; uint8_t v___x_2152_; 
v___x_2150_ = ((size_t)0ULL);
v___x_2151_ = lean_usize_of_nat(v___x_2148_);
v___x_2152_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasSafeRapp_spec__0(v_children_2146_, v___x_2150_, v___x_2151_);
lean_dec_ref(v_children_2146_);
return v___x_2152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasSafeRapp___boxed(lean_object* v_g_2153_, lean_object* v_a_2154_){
_start:
{
uint8_t v_res_2155_; lean_object* v_r_2156_; 
v_res_2155_ = lp_aesop_Aesop_Goal_hasSafeRapp(v_g_2153_);
v_r_2156_ = lean_box(v_res_2155_);
return v_r_2156_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnsafeExhausted(lean_object* v_g_2157_){
_start:
{
lean_object* v___x_2158_; lean_object* v_elimGoal_2159_; lean_object* v___x_2160_; uint8_t v_unsafeRulesSelected_2161_; 
v___x_2158_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2159_ = lean_ctor_get(v___x_2158_, 1);
lean_inc_ref(v_elimGoal_2159_);
v___x_2160_ = lean_apply_1(v_elimGoal_2159_, v_g_2157_);
v_unsafeRulesSelected_2161_ = lean_ctor_get_uint8(v___x_2160_, sizeof(void*)*14 + 11);
if (v_unsafeRulesSelected_2161_ == 0)
{
lean_dec_ref(v___x_2160_);
return v_unsafeRulesSelected_2161_;
}
else
{
lean_object* v_unsafeQueue_2162_; lean_object* v_start_2163_; lean_object* v_stop_2164_; uint8_t v___x_2165_; 
v_unsafeQueue_2162_ = lean_ctor_get(v___x_2160_, 12);
lean_inc_ref(v_unsafeQueue_2162_);
lean_dec_ref(v___x_2160_);
v_start_2163_ = lean_ctor_get(v_unsafeQueue_2162_, 1);
lean_inc(v_start_2163_);
v_stop_2164_ = lean_ctor_get(v_unsafeQueue_2162_, 2);
lean_inc(v_stop_2164_);
lean_dec_ref(v_unsafeQueue_2162_);
v___x_2165_ = lean_nat_dec_eq(v_start_2163_, v_stop_2164_);
lean_dec(v_stop_2164_);
lean_dec(v_start_2163_);
return v___x_2165_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnsafeExhausted___boxed(lean_object* v_g_2166_){
_start:
{
uint8_t v_res_2167_; lean_object* v_r_2168_; 
v_res_2167_ = lp_aesop_Aesop_Goal_isUnsafeExhausted(v_g_2166_);
v_r_2168_ = lean_box(v_res_2167_);
return v_r_2168_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isExhausted(lean_object* v_g_2169_){
_start:
{
uint8_t v___x_2171_; 
lean_inc(v_g_2169_);
v___x_2171_ = lp_aesop_Aesop_Goal_isUnsafeExhausted(v_g_2169_);
if (v___x_2171_ == 0)
{
uint8_t v___x_2172_; 
v___x_2172_ = lp_aesop_Aesop_Goal_hasSafeRapp(v_g_2169_);
return v___x_2172_;
}
else
{
lean_dec(v_g_2169_);
return v___x_2171_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isExhausted___boxed(lean_object* v_g_2173_, lean_object* v_a_2174_){
_start:
{
uint8_t v_res_2175_; lean_object* v_r_2176_; 
v_res_2175_ = lp_aesop_Aesop_Goal_isExhausted(v_g_2173_);
v_r_2176_ = lean_box(v_res_2175_);
return v_r_2176_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isActive(lean_object* v_g_2177_){
_start:
{
lean_object* v___x_2181_; lean_object* v_elimGoal_2182_; lean_object* v___x_2183_; uint8_t v_isIrrelevant_2184_; 
v___x_2181_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2182_ = lean_ctor_get(v___x_2181_, 1);
lean_inc_ref(v_elimGoal_2182_);
lean_inc(v_g_2177_);
v___x_2183_ = lean_apply_1(v_elimGoal_2182_, v_g_2177_);
v_isIrrelevant_2184_ = lean_ctor_get_uint8(v___x_2183_, sizeof(void*)*14 + 9);
lean_dec_ref(v___x_2183_);
if (v_isIrrelevant_2184_ == 0)
{
uint8_t v___x_2185_; 
v___x_2185_ = lp_aesop_Aesop_Goal_isExhausted(v_g_2177_);
if (v___x_2185_ == 0)
{
uint8_t v___x_2186_; 
v___x_2186_ = 1;
return v___x_2186_;
}
else
{
goto v___jp_2179_;
}
}
else
{
lean_dec(v_g_2177_);
goto v___jp_2179_;
}
v___jp_2179_:
{
uint8_t v___x_2180_; 
v___x_2180_ = 0;
return v___x_2180_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isActive___boxed(lean_object* v_g_2187_, lean_object* v_a_2188_){
_start:
{
uint8_t v_res_2189_; lean_object* v_r_2190_; 
v_res_2189_ = lp_aesop_Aesop_Goal_isActive(v_g_2187_);
v_r_2190_ = lean_box(v_res_2189_);
return v_r_2190_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(lean_object* v_as_2191_, size_t v_i_2192_, size_t v_stop_2193_){
_start:
{
uint8_t v___x_2195_; 
v___x_2195_ = lean_usize_dec_eq(v_i_2192_, v_stop_2193_);
if (v___x_2195_ == 0)
{
lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v_elimRapp_2199_; lean_object* v___x_2200_; uint8_t v_state_2201_; uint8_t v___x_2202_; uint8_t v___x_2203_; 
v___x_2196_ = lean_array_uget_borrowed(v_as_2191_, v_i_2192_);
v___x_2197_ = lean_st_ref_get(v___x_2196_);
v___x_2198_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2199_ = lean_ctor_get(v___x_2198_, 3);
lean_inc_ref(v_elimRapp_2199_);
v___x_2200_ = lean_apply_1(v_elimRapp_2199_, v___x_2197_);
v_state_2201_ = lean_ctor_get_uint8(v___x_2200_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_2200_);
v___x_2202_ = 1;
v___x_2203_ = lp_aesop_Aesop_NodeState_isUnprovable(v_state_2201_);
if (v___x_2203_ == 0)
{
return v___x_2202_;
}
else
{
if (v___x_2195_ == 0)
{
size_t v___x_2204_; size_t v___x_2205_; 
v___x_2204_ = ((size_t)1ULL);
v___x_2205_ = lean_usize_add(v_i_2192_, v___x_2204_);
v_i_2192_ = v___x_2205_;
goto _start;
}
else
{
return v___x_2202_;
}
}
}
else
{
uint8_t v___x_2207_; 
v___x_2207_ = 0;
return v___x_2207_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0___boxed(lean_object* v_as_2208_, lean_object* v_i_2209_, lean_object* v_stop_2210_, lean_object* v___y_2211_){
_start:
{
size_t v_i_boxed_2212_; size_t v_stop_boxed_2213_; uint8_t v_res_2214_; lean_object* v_r_2215_; 
v_i_boxed_2212_ = lean_unbox_usize(v_i_2209_);
lean_dec(v_i_2209_);
v_stop_boxed_2213_ = lean_unbox_usize(v_stop_2210_);
lean_dec(v_stop_2210_);
v_res_2214_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(v_as_2208_, v_i_boxed_2212_, v_stop_boxed_2213_);
lean_dec_ref(v_as_2208_);
v_r_2215_ = lean_box(v_res_2214_);
return v_r_2215_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasProvableRapp(lean_object* v_g_2216_){
_start:
{
lean_object* v___x_2218_; lean_object* v_elimGoal_2219_; lean_object* v___x_2220_; lean_object* v_children_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; uint8_t v___x_2224_; 
v___x_2218_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2219_ = lean_ctor_get(v___x_2218_, 1);
lean_inc_ref(v_elimGoal_2219_);
v___x_2220_ = lean_apply_1(v_elimGoal_2219_, v_g_2216_);
v_children_2221_ = lean_ctor_get(v___x_2220_, 2);
lean_inc_ref(v_children_2221_);
lean_dec_ref(v___x_2220_);
v___x_2222_ = lean_unsigned_to_nat(0u);
v___x_2223_ = lean_array_get_size(v_children_2221_);
v___x_2224_ = lean_nat_dec_lt(v___x_2222_, v___x_2223_);
if (v___x_2224_ == 0)
{
lean_dec_ref(v_children_2221_);
return v___x_2224_;
}
else
{
if (v___x_2224_ == 0)
{
lean_dec_ref(v_children_2221_);
return v___x_2224_;
}
else
{
size_t v___x_2225_; size_t v___x_2226_; uint8_t v___x_2227_; 
v___x_2225_ = ((size_t)0ULL);
v___x_2226_ = lean_usize_of_nat(v___x_2223_);
v___x_2227_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_hasProvableRapp_spec__0(v_children_2221_, v___x_2225_, v___x_2226_);
lean_dec_ref(v_children_2221_);
return v___x_2227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasProvableRapp___boxed(lean_object* v_g_2228_, lean_object* v_a_2229_){
_start:
{
uint8_t v_res_2230_; lean_object* v_r_2231_; 
v_res_2230_ = lp_aesop_Aesop_Goal_hasProvableRapp(v_g_2228_);
v_r_2231_ = lean_box(v_res_2230_);
return v_r_2231_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(lean_object* v_as_2235_, size_t v_sz_2236_, size_t v_i_2237_, lean_object* v_b_2238_){
_start:
{
uint8_t v___x_2240_; 
v___x_2240_ = lean_usize_dec_lt(v_i_2237_, v_sz_2236_);
if (v___x_2240_ == 0)
{
lean_inc_ref(v_b_2238_);
return v_b_2238_;
}
else
{
lean_object* v_a_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v_elimRapp_2244_; lean_object* v___x_2245_; uint8_t v_state_2246_; lean_object* v___x_2247_; uint8_t v___x_2248_; 
v_a_2241_ = lean_array_uget_borrowed(v_as_2235_, v_i_2237_);
v___x_2242_ = lean_st_ref_get(v_a_2241_);
v___x_2243_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2244_ = lean_ctor_get(v___x_2243_, 3);
lean_inc_ref(v_elimRapp_2244_);
v___x_2245_ = lean_apply_1(v_elimRapp_2244_, v___x_2242_);
v_state_2246_ = lean_ctor_get_uint8(v___x_2245_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_2245_);
v___x_2247_ = lean_box(0);
v___x_2248_ = lp_aesop_Aesop_NodeState_isProven(v_state_2246_);
if (v___x_2248_ == 0)
{
lean_object* v___x_2249_; size_t v___x_2250_; size_t v___x_2251_; 
v___x_2249_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0));
v___x_2250_ = ((size_t)1ULL);
v___x_2251_ = lean_usize_add(v_i_2237_, v___x_2250_);
v_i_2237_ = v___x_2251_;
v_b_2238_ = v___x_2249_;
goto _start;
}
else
{
lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; 
lean_inc(v_a_2241_);
v___x_2253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2253_, 0, v_a_2241_);
v___x_2254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2254_, 0, v___x_2253_);
v___x_2255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2255_, 0, v___x_2254_);
lean_ctor_set(v___x_2255_, 1, v___x_2247_);
return v___x_2255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___boxed(lean_object* v_as_2256_, lean_object* v_sz_2257_, lean_object* v_i_2258_, lean_object* v_b_2259_, lean_object* v___y_2260_){
_start:
{
size_t v_sz_boxed_2261_; size_t v_i_boxed_2262_; lean_object* v_res_2263_; 
v_sz_boxed_2261_ = lean_unbox_usize(v_sz_2257_);
lean_dec(v_sz_2257_);
v_i_boxed_2262_ = lean_unbox_usize(v_i_2258_);
lean_dec(v_i_2258_);
v_res_2263_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(v_as_2256_, v_sz_boxed_2261_, v_i_boxed_2262_, v_b_2259_);
lean_dec_ref(v_b_2259_);
lean_dec_ref(v_as_2256_);
return v_res_2263_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f(lean_object* v_g_2264_){
_start:
{
lean_object* v___x_2266_; lean_object* v_elimGoal_2267_; lean_object* v___x_2268_; lean_object* v_children_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; size_t v_sz_2272_; size_t v___x_2273_; lean_object* v___x_2274_; lean_object* v_fst_2275_; 
v___x_2266_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2267_ = lean_ctor_get(v___x_2266_, 1);
lean_inc_ref(v_elimGoal_2267_);
v___x_2268_ = lean_apply_1(v_elimGoal_2267_, v_g_2264_);
v_children_2269_ = lean_ctor_get(v___x_2268_, 2);
lean_inc_ref(v_children_2269_);
lean_dec_ref(v___x_2268_);
v___x_2270_ = lean_box(0);
v___x_2271_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0));
v_sz_2272_ = lean_array_size(v_children_2269_);
v___x_2273_ = ((size_t)0ULL);
v___x_2274_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0(v_children_2269_, v_sz_2272_, v___x_2273_, v___x_2271_);
lean_dec_ref(v_children_2269_);
v_fst_2275_ = lean_ctor_get(v___x_2274_, 0);
lean_inc(v_fst_2275_);
lean_dec_ref(v___x_2274_);
if (lean_obj_tag(v_fst_2275_) == 0)
{
return v___x_2270_;
}
else
{
lean_object* v_val_2276_; 
v_val_2276_ = lean_ctor_get(v_fst_2275_, 0);
lean_inc(v_val_2276_);
lean_dec_ref_known(v_fst_2275_, 1);
return v_val_2276_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f___boxed(lean_object* v_g_2277_, lean_object* v_a_2278_){
_start:
{
lean_object* v_res_2279_; 
v_res_2279_ = lp_aesop_Aesop_Goal_firstProvenRapp_x3f(v_g_2277_);
return v_res_2279_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_hasMVar(lean_object* v_g_2280_){
_start:
{
lean_object* v___x_2281_; lean_object* v_elimGoal_2282_; lean_object* v___x_2283_; lean_object* v_mvars_2284_; uint8_t v___x_2285_; 
v___x_2281_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2282_ = lean_ctor_get(v___x_2281_, 1);
lean_inc_ref(v_elimGoal_2282_);
v___x_2283_ = lean_apply_1(v_elimGoal_2282_, v_g_2280_);
v_mvars_2284_ = lean_ctor_get(v___x_2283_, 7);
lean_inc_ref(v_mvars_2284_);
lean_dec_ref(v___x_2283_);
v___x_2285_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(v_mvars_2284_);
lean_dec_ref(v_mvars_2284_);
if (v___x_2285_ == 0)
{
uint8_t v___x_2286_; 
v___x_2286_ = 1;
return v___x_2286_;
}
else
{
uint8_t v___x_2287_; 
v___x_2287_ = 0;
return v___x_2287_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_hasMVar___boxed(lean_object* v_g_2288_){
_start:
{
uint8_t v_res_2289_; lean_object* v_r_2290_; 
v_res_2289_ = lp_aesop_Aesop_Goal_hasMVar(v_g_2288_);
v_r_2290_ = lean_box(v_res_2289_);
return v_r_2290_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(lean_object* v_s_2291_){
_start:
{
lean_object* v___x_2292_; 
v___x_2292_ = lean_array_get_size(v_s_2291_);
return v___x_2292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0___boxed(lean_object* v_s_2293_){
_start:
{
lean_object* v_res_2294_; 
v_res_2294_ = lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_Goal_priority_spec__0(v_s_2293_);
lean_dec_ref(v_s_2293_);
return v_res_2294_;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Goal_priority(lean_object* v_g_2295_){
_start:
{
lean_object* v___x_2296_; lean_object* v_elimGoal_2297_; lean_object* v___x_2298_; lean_object* v_mvars_2299_; double v_successProbability_2300_; double v_toFloat_2301_; lean_object* v___x_2302_; double v___x_2303_; double v___x_2304_; double v___x_2305_; 
v___x_2296_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2297_ = lean_ctor_get(v___x_2296_, 1);
lean_inc_ref(v_elimGoal_2297_);
v___x_2298_ = lean_apply_1(v_elimGoal_2297_, v_g_2295_);
v_mvars_2299_ = lean_ctor_get(v___x_2298_, 7);
lean_inc_ref(v_mvars_2299_);
v_successProbability_2300_ = lean_ctor_get_float(v___x_2298_, sizeof(void*)*14);
lean_dec_ref(v___x_2298_);
v_toFloat_2301_ = lp_aesop_Aesop_unificationGoalPenalty;
v___x_2302_ = lean_array_get_size(v_mvars_2299_);
lean_dec_ref(v_mvars_2299_);
v___x_2303_ = lean_float_of_nat(v___x_2302_);
v___x_2304_ = pow(v_toFloat_2301_, v___x_2303_);
v___x_2305_ = lean_float_mul(v_successProbability_2300_, v___x_2304_);
return v___x_2305_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_priority___boxed(lean_object* v_g_2306_){
_start:
{
double v_res_2307_; lean_object* v_r_2308_; 
v_res_2307_ = lp_aesop_Aesop_Goal_priority(v_g_2306_);
v_r_2308_ = lean_box_float(v_res_2307_);
return v_r_2308_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object* v_g_2309_){
_start:
{
lean_object* v___x_2310_; lean_object* v_elimGoal_2311_; lean_object* v___x_2312_; lean_object* v_normalizationState_2313_; uint8_t v___x_2314_; 
v___x_2310_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2311_ = lean_ctor_get(v___x_2310_, 1);
lean_inc_ref(v_elimGoal_2311_);
v___x_2312_ = lean_apply_1(v_elimGoal_2311_, v_g_2309_);
v_normalizationState_2313_ = lean_ctor_get(v___x_2312_, 6);
lean_inc(v_normalizationState_2313_);
lean_dec_ref(v___x_2312_);
v___x_2314_ = lp_aesop_Aesop_NormalizationState_isNormal(v_normalizationState_2313_);
lean_dec(v_normalizationState_2313_);
return v___x_2314_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isNormal___boxed(lean_object* v_g_2315_){
_start:
{
uint8_t v_res_2316_; lean_object* v_r_2317_; 
v_res_2316_ = lp_aesop_Aesop_Goal_isNormal(v_g_2315_);
v_r_2317_ = lean_box(v_res_2316_);
return v_r_2317_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_originalGoalId(lean_object* v_g_2318_){
_start:
{
lean_object* v___x_2319_; lean_object* v_elimGoal_2320_; lean_object* v___x_2321_; lean_object* v_id_2322_; lean_object* v_origin_2323_; lean_object* v___x_2324_; 
v___x_2319_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2320_ = lean_ctor_get(v___x_2319_, 1);
lean_inc_ref(v_elimGoal_2320_);
v___x_2321_ = lean_apply_1(v_elimGoal_2320_, v_g_2318_);
v_id_2322_ = lean_ctor_get(v___x_2321_, 0);
lean_inc(v_id_2322_);
v_origin_2323_ = lean_ctor_get(v___x_2321_, 3);
lean_inc(v_origin_2323_);
lean_dec_ref(v___x_2321_);
v___x_2324_ = lp_aesop_Aesop_GoalOrigin_originalGoalId_x3f(v_origin_2323_);
lean_dec(v_origin_2323_);
if (lean_obj_tag(v___x_2324_) == 0)
{
return v_id_2322_;
}
else
{
lean_object* v_val_2325_; 
lean_dec(v_id_2322_);
v_val_2325_ = lean_ctor_get(v___x_2324_, 0);
lean_inc(v_val_2325_);
lean_dec_ref_known(v___x_2324_, 1);
return v_val_2325_;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object* v_g_2326_){
_start:
{
lean_object* v___x_2328_; 
v___x_2328_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_2326_);
if (lean_obj_tag(v___x_2328_) == 0)
{
uint8_t v___x_2329_; 
v___x_2329_ = 1;
return v___x_2329_;
}
else
{
uint8_t v___x_2330_; 
lean_dec_ref_known(v___x_2328_, 1);
v___x_2330_ = 0;
return v___x_2330_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isRoot___boxed(lean_object* v_g_2331_, lean_object* v_a_2332_){
_start:
{
uint8_t v_res_2333_; lean_object* v_r_2334_; 
v_res_2333_ = lp_aesop_Aesop_Goal_isRoot(v_g_2331_);
v_r_2334_ = lean_box(v_res_2333_);
return v_r_2334_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_introducesMVar(lean_object* v_r_2335_){
_start:
{
lean_object* v___x_2336_; lean_object* v_elimRapp_2337_; lean_object* v___x_2338_; lean_object* v_introducedMVars_2339_; uint8_t v___x_2340_; 
v___x_2336_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2337_ = lean_ctor_get(v___x_2336_, 3);
lean_inc_ref(v_elimRapp_2337_);
v___x_2338_ = lean_apply_1(v_elimRapp_2337_, v_r_2335_);
v_introducedMVars_2339_ = lean_ctor_get(v___x_2338_, 7);
lean_inc_ref(v_introducedMVars_2339_);
lean_dec_ref(v___x_2338_);
v___x_2340_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_Rapp_isSafe_spec__0(v_introducedMVars_2339_);
lean_dec_ref(v_introducedMVars_2339_);
if (v___x_2340_ == 0)
{
uint8_t v___x_2341_; 
v___x_2341_ = 1;
return v___x_2341_;
}
else
{
uint8_t v___x_2342_; 
v___x_2342_ = 0;
return v___x_2342_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_introducesMVar___boxed(lean_object* v_r_2343_){
_start:
{
uint8_t v_res_2344_; lean_object* v_r_2345_; 
v_res_2344_ = lp_aesop_Aesop_Rapp_introducesMVar(v_r_2343_);
v_r_2345_ = lean_box(v_res_2344_);
return v_r_2345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState(lean_object* v_r_2346_, lean_object* v_rootMetaState_2347_){
_start:
{
lean_object* v___x_2349_; lean_object* v_elimRapp_2350_; lean_object* v___x_2351_; lean_object* v_parent_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; 
v___x_2349_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2350_ = lean_ctor_get(v___x_2349_, 3);
lean_inc_ref(v_elimRapp_2350_);
v___x_2351_ = lean_apply_1(v_elimRapp_2350_, v_r_2346_);
v_parent_2352_ = lean_ctor_get(v___x_2351_, 1);
lean_inc(v_parent_2352_);
lean_dec_ref(v___x_2351_);
v___x_2353_ = lean_st_ref_get(v_parent_2352_);
lean_dec(v_parent_2352_);
v___x_2354_ = lp_aesop_Aesop_Goal_parentMetaState(v___x_2353_, v_rootMetaState_2347_);
return v___x_2354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_parentPostNormMetaState___boxed(lean_object* v_r_2355_, lean_object* v_rootMetaState_2356_, lean_object* v_a_2357_){
_start:
{
lean_object* v_res_2358_; 
v_res_2358_ = lp_aesop_Aesop_Rapp_parentPostNormMetaState(v_r_2355_, v_rootMetaState_2356_);
lean_dec_ref(v_rootMetaState_2356_);
return v_res_2358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0(lean_object* v_toApplicative_2359_, lean_object* v_s_2360_, lean_object* v_inst_2361_, lean_object* v_f_2362_, lean_object* v_____do__lift_2363_){
_start:
{
lean_object* v___x_2364_; lean_object* v_elimMVarCluster_2365_; lean_object* v___x_2366_; lean_object* v_goals_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; uint8_t v___x_2370_; 
v___x_2364_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_2365_ = lean_ctor_get(v___x_2364_, 5);
lean_inc_ref(v_elimMVarCluster_2365_);
v___x_2366_ = lean_apply_1(v_elimMVarCluster_2365_, v_____do__lift_2363_);
v_goals_2367_ = lean_ctor_get(v___x_2366_, 1);
lean_inc_ref(v_goals_2367_);
lean_dec_ref(v___x_2366_);
v___x_2368_ = lean_unsigned_to_nat(0u);
v___x_2369_ = lean_array_get_size(v_goals_2367_);
v___x_2370_ = lean_nat_dec_lt(v___x_2368_, v___x_2369_);
if (v___x_2370_ == 0)
{
lean_object* v_toPure_2371_; lean_object* v___x_2372_; 
lean_dec_ref(v_goals_2367_);
lean_dec(v_f_2362_);
lean_dec_ref(v_inst_2361_);
v_toPure_2371_ = lean_ctor_get(v_toApplicative_2359_, 1);
lean_inc(v_toPure_2371_);
lean_dec_ref(v_toApplicative_2359_);
v___x_2372_ = lean_apply_2(v_toPure_2371_, lean_box(0), v_s_2360_);
return v___x_2372_;
}
else
{
uint8_t v___x_2373_; 
v___x_2373_ = lean_nat_dec_le(v___x_2369_, v___x_2369_);
if (v___x_2373_ == 0)
{
if (v___x_2370_ == 0)
{
lean_object* v_toPure_2374_; lean_object* v___x_2375_; 
lean_dec_ref(v_goals_2367_);
lean_dec(v_f_2362_);
lean_dec_ref(v_inst_2361_);
v_toPure_2374_ = lean_ctor_get(v_toApplicative_2359_, 1);
lean_inc(v_toPure_2374_);
lean_dec_ref(v_toApplicative_2359_);
v___x_2375_ = lean_apply_2(v_toPure_2374_, lean_box(0), v_s_2360_);
return v___x_2375_;
}
else
{
size_t v___x_2376_; size_t v___x_2377_; lean_object* v___x_2378_; 
lean_dec_ref(v_toApplicative_2359_);
v___x_2376_ = ((size_t)0ULL);
v___x_2377_ = lean_usize_of_nat(v___x_2369_);
v___x_2378_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2361_, v_f_2362_, v_goals_2367_, v___x_2376_, v___x_2377_, v_s_2360_);
return v___x_2378_;
}
}
else
{
size_t v___x_2379_; size_t v___x_2380_; lean_object* v___x_2381_; 
lean_dec_ref(v_toApplicative_2359_);
v___x_2379_ = ((size_t)0ULL);
v___x_2380_ = lean_usize_of_nat(v___x_2369_);
v___x_2381_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2361_, v_f_2362_, v_goals_2367_, v___x_2379_, v___x_2380_, v_s_2360_);
return v___x_2381_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1(lean_object* v_toApplicative_2382_, lean_object* v_inst_2383_, lean_object* v_f_2384_, lean_object* v_inst_2385_, lean_object* v_toBind_2386_, lean_object* v_s_2387_, lean_object* v_cref_2388_){
_start:
{
lean_object* v___f_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; 
v___f_2389_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__0), 5, 4);
lean_closure_set(v___f_2389_, 0, v_toApplicative_2382_);
lean_closure_set(v___f_2389_, 1, v_s_2387_);
lean_closure_set(v___f_2389_, 2, v_inst_2383_);
lean_closure_set(v___f_2389_, 3, v_f_2384_);
v___x_2390_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2390_, 0, lean_box(0));
lean_closure_set(v___x_2390_, 1, lean_box(0));
lean_closure_set(v___x_2390_, 2, v_cref_2388_);
v___x_2391_ = lean_apply_2(v_inst_2385_, lean_box(0), v___x_2390_);
v___x_2392_ = lean_apply_4(v_toBind_2386_, lean_box(0), lean_box(0), v___x_2391_, v___f_2389_);
return v___x_2392_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(lean_object* v_inst_2393_, lean_object* v_inst_2394_, lean_object* v_init_2395_, lean_object* v_f_2396_, lean_object* v_r_2397_){
_start:
{
lean_object* v_toApplicative_2398_; lean_object* v_toBind_2399_; lean_object* v___x_2400_; lean_object* v_elimRapp_2401_; lean_object* v___x_2402_; lean_object* v_children_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; uint8_t v___x_2406_; 
v_toApplicative_2398_ = lean_ctor_get(v_inst_2393_, 0);
v_toBind_2399_ = lean_ctor_get(v_inst_2393_, 1);
v___x_2400_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2401_ = lean_ctor_get(v___x_2400_, 3);
lean_inc_ref(v_elimRapp_2401_);
v___x_2402_ = lean_apply_1(v_elimRapp_2401_, v_r_2397_);
v_children_2403_ = lean_ctor_get(v___x_2402_, 2);
lean_inc_ref(v_children_2403_);
lean_dec_ref(v___x_2402_);
v___x_2404_ = lean_unsigned_to_nat(0u);
v___x_2405_ = lean_array_get_size(v_children_2403_);
v___x_2406_ = lean_nat_dec_lt(v___x_2404_, v___x_2405_);
if (v___x_2406_ == 0)
{
lean_object* v_toPure_2407_; lean_object* v___x_2408_; 
lean_inc_ref(v_toApplicative_2398_);
lean_dec_ref(v_children_2403_);
lean_dec(v_f_2396_);
lean_dec(v_inst_2394_);
lean_dec_ref(v_inst_2393_);
v_toPure_2407_ = lean_ctor_get(v_toApplicative_2398_, 1);
lean_inc(v_toPure_2407_);
lean_dec_ref(v_toApplicative_2398_);
v___x_2408_ = lean_apply_2(v_toPure_2407_, lean_box(0), v_init_2395_);
return v___x_2408_;
}
else
{
lean_object* v___f_2409_; uint8_t v___x_2410_; 
lean_inc(v_toBind_2399_);
lean_inc_ref(v_inst_2393_);
lean_inc_ref(v_toApplicative_2398_);
v___f_2409_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg___lam__1), 7, 5);
lean_closure_set(v___f_2409_, 0, v_toApplicative_2398_);
lean_closure_set(v___f_2409_, 1, v_inst_2393_);
lean_closure_set(v___f_2409_, 2, v_f_2396_);
lean_closure_set(v___f_2409_, 3, v_inst_2394_);
lean_closure_set(v___f_2409_, 4, v_toBind_2399_);
v___x_2410_ = lean_nat_dec_le(v___x_2405_, v___x_2405_);
if (v___x_2410_ == 0)
{
if (v___x_2406_ == 0)
{
lean_object* v_toPure_2411_; lean_object* v___x_2412_; 
lean_inc_ref(v_toApplicative_2398_);
lean_dec_ref(v___f_2409_);
lean_dec_ref(v_children_2403_);
lean_dec_ref(v_inst_2393_);
v_toPure_2411_ = lean_ctor_get(v_toApplicative_2398_, 1);
lean_inc(v_toPure_2411_);
lean_dec_ref(v_toApplicative_2398_);
v___x_2412_ = lean_apply_2(v_toPure_2411_, lean_box(0), v_init_2395_);
return v___x_2412_;
}
else
{
size_t v___x_2413_; size_t v___x_2414_; lean_object* v___x_2415_; 
v___x_2413_ = ((size_t)0ULL);
v___x_2414_ = lean_usize_of_nat(v___x_2405_);
v___x_2415_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2393_, v___f_2409_, v_children_2403_, v___x_2413_, v___x_2414_, v_init_2395_);
return v___x_2415_;
}
}
else
{
size_t v___x_2416_; size_t v___x_2417_; lean_object* v___x_2418_; 
v___x_2416_ = ((size_t)0ULL);
v___x_2417_ = lean_usize_of_nat(v___x_2405_);
v___x_2418_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2393_, v___f_2409_, v_children_2403_, v___x_2416_, v___x_2417_, v_init_2395_);
return v___x_2418_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_foldSubgoalsM(lean_object* v_m_2419_, lean_object* v_00_u03c3_2420_, lean_object* v_inst_2421_, lean_object* v_inst_2422_, lean_object* v_init_2423_, lean_object* v_f_2424_, lean_object* v_r_2425_){
_start:
{
lean_object* v___x_2426_; 
v___x_2426_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(v_inst_2421_, v_inst_2422_, v_init_2423_, v_f_2424_, v_r_2425_);
return v___x_2426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0(lean_object* v_f_2427_, lean_object* v_x_2428_, lean_object* v___y_2429_){
_start:
{
lean_object* v___x_2430_; 
v___x_2430_ = lean_apply_1(v_f_2427_, v___y_2429_);
return v___x_2430_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1(lean_object* v_toApplicative_2431_, lean_object* v_inst_2432_, lean_object* v___f_2433_, lean_object* v_____do__lift_2434_){
_start:
{
lean_object* v___x_2435_; lean_object* v_elimMVarCluster_2436_; lean_object* v___x_2437_; lean_object* v_goals_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; uint8_t v___x_2442_; 
v___x_2435_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_2436_ = lean_ctor_get(v___x_2435_, 5);
lean_inc_ref(v_elimMVarCluster_2436_);
v___x_2437_ = lean_apply_1(v_elimMVarCluster_2436_, v_____do__lift_2434_);
v_goals_2438_ = lean_ctor_get(v___x_2437_, 1);
lean_inc_ref(v_goals_2438_);
lean_dec_ref(v___x_2437_);
v___x_2439_ = lean_unsigned_to_nat(0u);
v___x_2440_ = lean_array_get_size(v_goals_2438_);
v___x_2441_ = lean_box(0);
v___x_2442_ = lean_nat_dec_lt(v___x_2439_, v___x_2440_);
if (v___x_2442_ == 0)
{
lean_object* v_toPure_2443_; lean_object* v___x_2444_; 
lean_dec_ref(v_goals_2438_);
lean_dec(v___f_2433_);
lean_dec_ref(v_inst_2432_);
v_toPure_2443_ = lean_ctor_get(v_toApplicative_2431_, 1);
lean_inc(v_toPure_2443_);
lean_dec_ref(v_toApplicative_2431_);
v___x_2444_ = lean_apply_2(v_toPure_2443_, lean_box(0), v___x_2441_);
return v___x_2444_;
}
else
{
uint8_t v___x_2445_; 
v___x_2445_ = lean_nat_dec_le(v___x_2440_, v___x_2440_);
if (v___x_2445_ == 0)
{
if (v___x_2442_ == 0)
{
lean_object* v_toPure_2446_; lean_object* v___x_2447_; 
lean_dec_ref(v_goals_2438_);
lean_dec(v___f_2433_);
lean_dec_ref(v_inst_2432_);
v_toPure_2446_ = lean_ctor_get(v_toApplicative_2431_, 1);
lean_inc(v_toPure_2446_);
lean_dec_ref(v_toApplicative_2431_);
v___x_2447_ = lean_apply_2(v_toPure_2446_, lean_box(0), v___x_2441_);
return v___x_2447_;
}
else
{
size_t v___x_2448_; size_t v___x_2449_; lean_object* v___x_2450_; 
lean_dec_ref(v_toApplicative_2431_);
v___x_2448_ = ((size_t)0ULL);
v___x_2449_ = lean_usize_of_nat(v___x_2440_);
v___x_2450_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2432_, v___f_2433_, v_goals_2438_, v___x_2448_, v___x_2449_, v___x_2441_);
return v___x_2450_;
}
}
else
{
size_t v___x_2451_; size_t v___x_2452_; lean_object* v___x_2453_; 
lean_dec_ref(v_toApplicative_2431_);
v___x_2451_ = ((size_t)0ULL);
v___x_2452_ = lean_usize_of_nat(v___x_2440_);
v___x_2453_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2432_, v___f_2433_, v_goals_2438_, v___x_2451_, v___x_2452_, v___x_2441_);
return v___x_2453_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2(lean_object* v_inst_2454_, lean_object* v_toBind_2455_, lean_object* v___f_2456_, lean_object* v_x_2457_, lean_object* v___y_2458_){
_start:
{
lean_object* v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; 
v___x_2459_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2459_, 0, lean_box(0));
lean_closure_set(v___x_2459_, 1, lean_box(0));
lean_closure_set(v___x_2459_, 2, v___y_2458_);
v___x_2460_ = lean_apply_2(v_inst_2454_, lean_box(0), v___x_2459_);
v___x_2461_ = lean_apply_4(v_toBind_2455_, lean_box(0), lean_box(0), v___x_2460_, v___f_2456_);
return v___x_2461_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object* v_inst_2462_, lean_object* v_inst_2463_, lean_object* v_f_2464_, lean_object* v_r_2465_){
_start:
{
lean_object* v_toApplicative_2466_; lean_object* v_toBind_2467_; lean_object* v___x_2468_; lean_object* v_elimRapp_2469_; lean_object* v___x_2470_; lean_object* v_children_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; uint8_t v___x_2475_; 
v_toApplicative_2466_ = lean_ctor_get(v_inst_2462_, 0);
v_toBind_2467_ = lean_ctor_get(v_inst_2462_, 1);
v___x_2468_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimRapp_2469_ = lean_ctor_get(v___x_2468_, 3);
lean_inc_ref(v_elimRapp_2469_);
v___x_2470_ = lean_apply_1(v_elimRapp_2469_, v_r_2465_);
v_children_2471_ = lean_ctor_get(v___x_2470_, 2);
lean_inc_ref(v_children_2471_);
lean_dec_ref(v___x_2470_);
v___x_2472_ = lean_unsigned_to_nat(0u);
v___x_2473_ = lean_array_get_size(v_children_2471_);
v___x_2474_ = lean_box(0);
v___x_2475_ = lean_nat_dec_lt(v___x_2472_, v___x_2473_);
if (v___x_2475_ == 0)
{
lean_object* v_toPure_2476_; lean_object* v___x_2477_; 
lean_inc_ref(v_toApplicative_2466_);
lean_dec_ref(v_children_2471_);
lean_dec(v_f_2464_);
lean_dec(v_inst_2463_);
lean_dec_ref(v_inst_2462_);
v_toPure_2476_ = lean_ctor_get(v_toApplicative_2466_, 1);
lean_inc(v_toPure_2476_);
lean_dec_ref(v_toApplicative_2466_);
v___x_2477_ = lean_apply_2(v_toPure_2476_, lean_box(0), v___x_2474_);
return v___x_2477_;
}
else
{
lean_object* v___f_2478_; lean_object* v___f_2479_; lean_object* v___f_2480_; uint8_t v___x_2481_; 
v___f_2478_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2478_, 0, v_f_2464_);
lean_inc_ref(v_inst_2462_);
lean_inc_ref(v_toApplicative_2466_);
v___f_2479_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2479_, 0, v_toApplicative_2466_);
lean_closure_set(v___f_2479_, 1, v_inst_2462_);
lean_closure_set(v___f_2479_, 2, v___f_2478_);
lean_inc(v_toBind_2467_);
v___f_2480_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_forSubgoalsM___redArg___lam__2), 5, 3);
lean_closure_set(v___f_2480_, 0, v_inst_2463_);
lean_closure_set(v___f_2480_, 1, v_toBind_2467_);
lean_closure_set(v___f_2480_, 2, v___f_2479_);
v___x_2481_ = lean_nat_dec_le(v___x_2473_, v___x_2473_);
if (v___x_2481_ == 0)
{
if (v___x_2475_ == 0)
{
lean_object* v_toPure_2482_; lean_object* v___x_2483_; 
lean_inc_ref(v_toApplicative_2466_);
lean_dec_ref(v___f_2480_);
lean_dec_ref(v_children_2471_);
lean_dec_ref(v_inst_2462_);
v_toPure_2482_ = lean_ctor_get(v_toApplicative_2466_, 1);
lean_inc(v_toPure_2482_);
lean_dec_ref(v_toApplicative_2466_);
v___x_2483_ = lean_apply_2(v_toPure_2482_, lean_box(0), v___x_2474_);
return v___x_2483_;
}
else
{
size_t v___x_2484_; size_t v___x_2485_; lean_object* v___x_2486_; 
v___x_2484_ = ((size_t)0ULL);
v___x_2485_ = lean_usize_of_nat(v___x_2473_);
v___x_2486_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2462_, v___f_2480_, v_children_2471_, v___x_2484_, v___x_2485_, v___x_2474_);
return v___x_2486_;
}
}
else
{
size_t v___x_2487_; size_t v___x_2488_; lean_object* v___x_2489_; 
v___x_2487_ = ((size_t)0ULL);
v___x_2488_ = lean_usize_of_nat(v___x_2473_);
v___x_2489_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2462_, v___f_2480_, v_children_2471_, v___x_2487_, v___x_2488_, v___x_2474_);
return v___x_2489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM(lean_object* v_m_2490_, lean_object* v_inst_2491_, lean_object* v_inst_2492_, lean_object* v_f_2493_, lean_object* v_r_2494_){
_start:
{
lean_object* v___x_2495_; 
v___x_2495_ = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(v_inst_2491_, v_inst_2492_, v_f_2493_, v_r_2494_);
return v___x_2495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0(lean_object* v_toPure_2496_, lean_object* v_subgoals_2497_, lean_object* v_gref_2498_){
_start:
{
lean_object* v___x_2499_; lean_object* v___x_2500_; 
v___x_2499_ = lean_array_push(v_subgoals_2497_, v_gref_2498_);
v___x_2500_ = lean_apply_2(v_toPure_2496_, lean_box(0), v___x_2499_);
return v___x_2500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals___redArg(lean_object* v_inst_2501_, lean_object* v_inst_2502_, lean_object* v_r_2503_){
_start:
{
lean_object* v_toApplicative_2504_; lean_object* v_toPure_2505_; lean_object* v___x_2506_; lean_object* v___f_2507_; lean_object* v___x_2508_; 
v_toApplicative_2504_ = lean_ctor_get(v_inst_2501_, 0);
v_toPure_2505_ = lean_ctor_get(v_toApplicative_2504_, 1);
v___x_2506_ = ((lean_object*)(lp_aesop_Aesop_Goal_safeRapps___closed__0));
lean_inc(v_toPure_2505_);
v___f_2507_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_subgoals___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2507_, 0, v_toPure_2505_);
v___x_2508_ = lp_aesop_Aesop_Rapp_foldSubgoalsM___redArg(v_inst_2501_, v_inst_2502_, v___x_2506_, v___f_2507_, v_r_2503_);
return v___x_2508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_subgoals(lean_object* v_m_2509_, lean_object* v_inst_2510_, lean_object* v_inst_2511_, lean_object* v_r_2512_){
_start:
{
lean_object* v___x_2513_; 
v___x_2513_ = lp_aesop_Aesop_Rapp_subgoals___redArg(v_inst_2510_, v_inst_2511_, v_r_2512_);
return v___x_2513_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth(lean_object* v_r_2514_){
_start:
{
lean_object* v___x_2516_; lean_object* v_elimGoal_2517_; lean_object* v_elimRapp_2518_; lean_object* v___x_2519_; lean_object* v_parent_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v_depth_2523_; 
v___x_2516_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2517_ = lean_ctor_get(v___x_2516_, 1);
v_elimRapp_2518_ = lean_ctor_get(v___x_2516_, 3);
lean_inc_ref(v_elimRapp_2518_);
v___x_2519_ = lean_apply_1(v_elimRapp_2518_, v_r_2514_);
v_parent_2520_ = lean_ctor_get(v___x_2519_, 1);
lean_inc(v_parent_2520_);
lean_dec_ref(v___x_2519_);
v___x_2521_ = lean_st_ref_get(v_parent_2520_);
lean_dec(v_parent_2520_);
lean_inc_ref(v_elimGoal_2517_);
v___x_2522_ = lean_apply_1(v_elimGoal_2517_, v___x_2521_);
v_depth_2523_ = lean_ctor_get(v___x_2522_, 4);
lean_inc(v_depth_2523_);
lean_dec_ref(v___x_2522_);
return v_depth_2523_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_depth___boxed(lean_object* v_r_2524_, lean_object* v_a_2525_){
_start:
{
lean_object* v_res_2526_; 
v_res_2526_ = lp_aesop_Aesop_Rapp_depth(v_r_2524_);
return v_res_2526_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(lean_object* v_as_2527_, size_t v_sz_2528_, size_t v_i_2529_, lean_object* v_b_2530_){
_start:
{
uint8_t v___x_2532_; 
v___x_2532_ = lean_usize_dec_lt(v_i_2529_, v_sz_2528_);
if (v___x_2532_ == 0)
{
lean_inc_ref(v_b_2530_);
return v_b_2530_;
}
else
{
lean_object* v_a_2533_; lean_object* v___x_2534_; lean_object* v___x_2535_; lean_object* v_elimGoal_2536_; lean_object* v___x_2537_; uint8_t v_state_2538_; lean_object* v___x_2539_; uint8_t v___x_2540_; 
v_a_2533_ = lean_array_uget_borrowed(v_as_2527_, v_i_2529_);
v___x_2534_ = lean_st_ref_get(v_a_2533_);
v___x_2535_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimGoal_2536_ = lean_ctor_get(v___x_2535_, 1);
lean_inc_ref(v_elimGoal_2536_);
v___x_2537_ = lean_apply_1(v_elimGoal_2536_, v___x_2534_);
v_state_2538_ = lean_ctor_get_uint8(v___x_2537_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_2537_);
v___x_2539_ = lean_box(0);
v___x_2540_ = lp_aesop_Aesop_GoalState_isProven(v_state_2538_);
if (v___x_2540_ == 0)
{
lean_object* v___x_2541_; size_t v___x_2542_; size_t v___x_2543_; 
v___x_2541_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0));
v___x_2542_ = ((size_t)1ULL);
v___x_2543_ = lean_usize_add(v_i_2529_, v___x_2542_);
v_i_2529_ = v___x_2543_;
v_b_2530_ = v___x_2541_;
goto _start;
}
else
{
lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; 
lean_inc(v_a_2533_);
v___x_2545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2545_, 0, v_a_2533_);
v___x_2546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2546_, 0, v___x_2545_);
v___x_2547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2547_, 0, v___x_2546_);
lean_ctor_set(v___x_2547_, 1, v___x_2539_);
return v___x_2547_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0___boxed(lean_object* v_as_2548_, lean_object* v_sz_2549_, lean_object* v_i_2550_, lean_object* v_b_2551_, lean_object* v___y_2552_){
_start:
{
size_t v_sz_boxed_2553_; size_t v_i_boxed_2554_; lean_object* v_res_2555_; 
v_sz_boxed_2553_ = lean_unbox_usize(v_sz_2549_);
lean_dec(v_sz_2549_);
v_i_boxed_2554_ = lean_unbox_usize(v_i_2550_);
lean_dec(v_i_2550_);
v_res_2555_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(v_as_2548_, v_sz_boxed_2553_, v_i_boxed_2554_, v_b_2551_);
lean_dec_ref(v_b_2551_);
lean_dec_ref(v_as_2548_);
return v_res_2555_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f(lean_object* v_c_2556_){
_start:
{
lean_object* v___x_2558_; lean_object* v_elimMVarCluster_2559_; lean_object* v___x_2560_; lean_object* v_goals_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; size_t v_sz_2564_; size_t v___x_2565_; lean_object* v___x_2566_; lean_object* v_fst_2567_; 
v___x_2558_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_elimMVarCluster_2559_ = lean_ctor_get(v___x_2558_, 5);
lean_inc_ref(v_elimMVarCluster_2559_);
v___x_2560_ = lean_apply_1(v_elimMVarCluster_2559_, v_c_2556_);
v_goals_2561_ = lean_ctor_get(v___x_2560_, 1);
lean_inc_ref(v_goals_2561_);
lean_dec_ref(v___x_2560_);
v___x_2562_ = lean_box(0);
v___x_2563_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Goal_firstProvenRapp_x3f_spec__0___closed__0));
v_sz_2564_ = lean_array_size(v_goals_2561_);
v___x_2565_ = ((size_t)0ULL);
v___x_2566_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_MVarCluster_provenGoal_x3f_spec__0(v_goals_2561_, v_sz_2564_, v___x_2565_, v___x_2563_);
lean_dec_ref(v_goals_2561_);
v_fst_2567_ = lean_ctor_get(v___x_2566_, 0);
lean_inc(v_fst_2567_);
lean_dec_ref(v___x_2566_);
if (lean_obj_tag(v_fst_2567_) == 0)
{
return v___x_2562_;
}
else
{
lean_object* v_val_2568_; 
v_val_2568_ = lean_ctor_get(v_fst_2567_, 0);
lean_inc(v_val_2568_);
lean_dec_ref_known(v_fst_2567_, 1);
return v_val_2568_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f___boxed(lean_object* v_c_2569_, lean_object* v_a_2570_){
_start:
{
lean_object* v_res_2571_; 
v_res_2571_ = lp_aesop_Aesop_MVarCluster_provenGoal_x3f(v_c_2569_);
return v_res_2571_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(lean_object* v_r_2572_){
_start:
{
lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v_introRapp_2576_; lean_object* v_elimRapp_2577_; lean_object* v___x_2578_; lean_object* v_metaState_2579_; lean_object* v_core_2580_; lean_object* v_toState_2581_; lean_object* v_id_2582_; lean_object* v_parent_2583_; lean_object* v_children_2584_; uint8_t v_state_2585_; uint8_t v_isIrrelevant_2586_; lean_object* v_appliedRule_2587_; lean_object* v_scriptSteps_x3f_2588_; lean_object* v_originalSubgoals_2589_; double v_successProbability_2590_; lean_object* v_introducedMVars_2591_; lean_object* v_assignedMVars_2592_; lean_object* v___x_2594_; uint8_t v_isShared_2595_; uint8_t v_isSharedCheck_2638_; 
v___x_2574_ = lean_st_ref_take(v_r_2572_);
v___x_2575_ = ((lean_object*)(lp_aesop_Aesop_treeImpl));
v_introRapp_2576_ = lean_ctor_get(v___x_2575_, 2);
v_elimRapp_2577_ = lean_ctor_get(v___x_2575_, 3);
lean_inc_ref(v_elimRapp_2577_);
v___x_2578_ = lean_apply_1(v_elimRapp_2577_, v___x_2574_);
v_metaState_2579_ = lean_ctor_get(v___x_2578_, 6);
lean_inc_ref(v_metaState_2579_);
v_core_2580_ = lean_ctor_get(v_metaState_2579_, 0);
lean_inc_ref(v_core_2580_);
v_toState_2581_ = lean_ctor_get(v_core_2580_, 0);
lean_inc_ref(v_toState_2581_);
v_id_2582_ = lean_ctor_get(v___x_2578_, 0);
v_parent_2583_ = lean_ctor_get(v___x_2578_, 1);
v_children_2584_ = lean_ctor_get(v___x_2578_, 2);
v_state_2585_ = lean_ctor_get_uint8(v___x_2578_, sizeof(void*)*9 + 8);
v_isIrrelevant_2586_ = lean_ctor_get_uint8(v___x_2578_, sizeof(void*)*9 + 9);
v_appliedRule_2587_ = lean_ctor_get(v___x_2578_, 3);
v_scriptSteps_x3f_2588_ = lean_ctor_get(v___x_2578_, 4);
v_originalSubgoals_2589_ = lean_ctor_get(v___x_2578_, 5);
v_successProbability_2590_ = lean_ctor_get_float(v___x_2578_, sizeof(void*)*9);
v_introducedMVars_2591_ = lean_ctor_get(v___x_2578_, 7);
v_assignedMVars_2592_ = lean_ctor_get(v___x_2578_, 8);
v_isSharedCheck_2638_ = !lean_is_exclusive(v___x_2578_);
if (v_isSharedCheck_2638_ == 0)
{
lean_object* v_unused_2639_; 
v_unused_2639_ = lean_ctor_get(v___x_2578_, 6);
lean_dec(v_unused_2639_);
v___x_2594_ = v___x_2578_;
v_isShared_2595_ = v_isSharedCheck_2638_;
goto v_resetjp_2593_;
}
else
{
lean_inc(v_assignedMVars_2592_);
lean_inc(v_introducedMVars_2591_);
lean_inc(v_originalSubgoals_2589_);
lean_inc(v_scriptSteps_x3f_2588_);
lean_inc(v_appliedRule_2587_);
lean_inc(v_children_2584_);
lean_inc(v_parent_2583_);
lean_inc(v_id_2582_);
lean_dec(v___x_2578_);
v___x_2594_ = lean_box(0);
v_isShared_2595_ = v_isSharedCheck_2638_;
goto v_resetjp_2593_;
}
v_resetjp_2593_:
{
lean_object* v_meta_2596_; lean_object* v___x_2598_; uint8_t v_isShared_2599_; uint8_t v_isSharedCheck_2636_; 
v_meta_2596_ = lean_ctor_get(v_metaState_2579_, 1);
v_isSharedCheck_2636_ = !lean_is_exclusive(v_metaState_2579_);
if (v_isSharedCheck_2636_ == 0)
{
lean_object* v_unused_2637_; 
v_unused_2637_ = lean_ctor_get(v_metaState_2579_, 0);
lean_dec(v_unused_2637_);
v___x_2598_ = v_metaState_2579_;
v_isShared_2599_ = v_isSharedCheck_2636_;
goto v_resetjp_2597_;
}
else
{
lean_inc(v_meta_2596_);
lean_dec(v_metaState_2579_);
v___x_2598_ = lean_box(0);
v_isShared_2599_ = v_isSharedCheck_2636_;
goto v_resetjp_2597_;
}
v_resetjp_2597_:
{
lean_object* v_passedHeartbeats_2600_; lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2634_; 
v_passedHeartbeats_2600_ = lean_ctor_get(v_core_2580_, 1);
v_isSharedCheck_2634_ = !lean_is_exclusive(v_core_2580_);
if (v_isSharedCheck_2634_ == 0)
{
lean_object* v_unused_2635_; 
v_unused_2635_ = lean_ctor_get(v_core_2580_, 0);
lean_dec(v_unused_2635_);
v___x_2602_ = v_core_2580_;
v_isShared_2603_ = v_isSharedCheck_2634_;
goto v_resetjp_2601_;
}
else
{
lean_inc(v_passedHeartbeats_2600_);
lean_dec(v_core_2580_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2634_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v_env_2604_; lean_object* v_nextMacroScope_2605_; lean_object* v_ngen_2606_; lean_object* v_auxDeclNGen_2607_; lean_object* v_traceState_2608_; lean_object* v_cache_2609_; lean_object* v_messages_2610_; lean_object* v_infoState_2611_; lean_object* v_snapshotTasks_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2633_; 
v_env_2604_ = lean_ctor_get(v_toState_2581_, 0);
v_nextMacroScope_2605_ = lean_ctor_get(v_toState_2581_, 1);
v_ngen_2606_ = lean_ctor_get(v_toState_2581_, 2);
v_auxDeclNGen_2607_ = lean_ctor_get(v_toState_2581_, 3);
v_traceState_2608_ = lean_ctor_get(v_toState_2581_, 4);
v_cache_2609_ = lean_ctor_get(v_toState_2581_, 5);
v_messages_2610_ = lean_ctor_get(v_toState_2581_, 6);
v_infoState_2611_ = lean_ctor_get(v_toState_2581_, 7);
v_snapshotTasks_2612_ = lean_ctor_get(v_toState_2581_, 8);
v_isSharedCheck_2633_ = !lean_is_exclusive(v_toState_2581_);
if (v_isSharedCheck_2633_ == 0)
{
v___x_2614_ = v_toState_2581_;
v_isShared_2615_ = v_isSharedCheck_2633_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_snapshotTasks_2612_);
lean_inc(v_infoState_2611_);
lean_inc(v_messages_2610_);
lean_inc(v_cache_2609_);
lean_inc(v_traceState_2608_);
lean_inc(v_auxDeclNGen_2607_);
lean_inc(v_ngen_2606_);
lean_inc(v_nextMacroScope_2605_);
lean_inc(v_env_2604_);
lean_dec(v_toState_2581_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2633_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2616_; lean_object* v_fst_2617_; lean_object* v_snd_2618_; lean_object* v___x_2620_; 
v___x_2616_ = l_Lean_DeclNameGenerator_mkChild(v_auxDeclNGen_2607_);
v_fst_2617_ = lean_ctor_get(v___x_2616_, 0);
lean_inc(v_fst_2617_);
v_snd_2618_ = lean_ctor_get(v___x_2616_, 1);
lean_inc(v_snd_2618_);
lean_dec_ref(v___x_2616_);
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 3, v_snd_2618_);
v___x_2620_ = v___x_2614_;
goto v_reusejp_2619_;
}
else
{
lean_object* v_reuseFailAlloc_2632_; 
v_reuseFailAlloc_2632_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2632_, 0, v_env_2604_);
lean_ctor_set(v_reuseFailAlloc_2632_, 1, v_nextMacroScope_2605_);
lean_ctor_set(v_reuseFailAlloc_2632_, 2, v_ngen_2606_);
lean_ctor_set(v_reuseFailAlloc_2632_, 3, v_snd_2618_);
lean_ctor_set(v_reuseFailAlloc_2632_, 4, v_traceState_2608_);
lean_ctor_set(v_reuseFailAlloc_2632_, 5, v_cache_2609_);
lean_ctor_set(v_reuseFailAlloc_2632_, 6, v_messages_2610_);
lean_ctor_set(v_reuseFailAlloc_2632_, 7, v_infoState_2611_);
lean_ctor_set(v_reuseFailAlloc_2632_, 8, v_snapshotTasks_2612_);
v___x_2620_ = v_reuseFailAlloc_2632_;
goto v_reusejp_2619_;
}
v_reusejp_2619_:
{
lean_object* v___x_2622_; 
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 0, v___x_2620_);
v___x_2622_ = v___x_2602_;
goto v_reusejp_2621_;
}
else
{
lean_object* v_reuseFailAlloc_2631_; 
v_reuseFailAlloc_2631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2631_, 0, v___x_2620_);
lean_ctor_set(v_reuseFailAlloc_2631_, 1, v_passedHeartbeats_2600_);
v___x_2622_ = v_reuseFailAlloc_2631_;
goto v_reusejp_2621_;
}
v_reusejp_2621_:
{
lean_object* v___x_2624_; 
if (v_isShared_2599_ == 0)
{
lean_ctor_set(v___x_2598_, 0, v___x_2622_);
v___x_2624_ = v___x_2598_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v___x_2622_);
lean_ctor_set(v_reuseFailAlloc_2630_, 1, v_meta_2596_);
v___x_2624_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
lean_object* v___x_2626_; 
if (v_isShared_2595_ == 0)
{
lean_ctor_set(v___x_2594_, 6, v___x_2624_);
v___x_2626_ = v___x_2594_;
goto v_reusejp_2625_;
}
else
{
lean_object* v_reuseFailAlloc_2629_; 
v_reuseFailAlloc_2629_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_2629_, 0, v_id_2582_);
lean_ctor_set(v_reuseFailAlloc_2629_, 1, v_parent_2583_);
lean_ctor_set(v_reuseFailAlloc_2629_, 2, v_children_2584_);
lean_ctor_set(v_reuseFailAlloc_2629_, 3, v_appliedRule_2587_);
lean_ctor_set(v_reuseFailAlloc_2629_, 4, v_scriptSteps_x3f_2588_);
lean_ctor_set(v_reuseFailAlloc_2629_, 5, v_originalSubgoals_2589_);
lean_ctor_set(v_reuseFailAlloc_2629_, 6, v___x_2624_);
lean_ctor_set(v_reuseFailAlloc_2629_, 7, v_introducedMVars_2591_);
lean_ctor_set(v_reuseFailAlloc_2629_, 8, v_assignedMVars_2592_);
lean_ctor_set_uint8(v_reuseFailAlloc_2629_, sizeof(void*)*9 + 8, v_state_2585_);
lean_ctor_set_uint8(v_reuseFailAlloc_2629_, sizeof(void*)*9 + 9, v_isIrrelevant_2586_);
lean_ctor_set_float(v_reuseFailAlloc_2629_, sizeof(void*)*9, v_successProbability_2590_);
v___x_2626_ = v_reuseFailAlloc_2629_;
goto v_reusejp_2625_;
}
v_reusejp_2625_:
{
lean_object* v_r_2627_; lean_object* v___x_2628_; 
lean_inc(v_introRapp_2576_);
v_r_2627_ = lean_apply_1(v_introRapp_2576_, v___x_2626_);
v___x_2628_ = lean_st_ref_set(v_r_2572_, v_r_2627_);
return v_fst_2617_;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator___boxed(lean_object* v_r_2640_, lean_object* v_a_2641_){
_start:
{
lean_object* v_res_2642_; 
v_res_2642_ = lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(v_r_2640_);
lean_dec(v_r_2640_);
return v_res_2642_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Constants(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_Data(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedGoalId_default = _init_lp_aesop_Aesop_instInhabitedGoalId_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalId_default);
lp_aesop_Aesop_instInhabitedGoalId = _init_lp_aesop_Aesop_instInhabitedGoalId();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalId);
lp_aesop_Aesop_GoalId_zero = _init_lp_aesop_Aesop_GoalId_zero();
lean_mark_persistent(lp_aesop_Aesop_GoalId_zero);
lp_aesop_Aesop_GoalId_one = _init_lp_aesop_Aesop_GoalId_one();
lean_mark_persistent(lp_aesop_Aesop_GoalId_one);
lp_aesop_Aesop_GoalId_dummy = _init_lp_aesop_Aesop_GoalId_dummy();
lean_mark_persistent(lp_aesop_Aesop_GoalId_dummy);
lp_aesop_Aesop_GoalId_instLT = _init_lp_aesop_Aesop_GoalId_instLT();
lean_mark_persistent(lp_aesop_Aesop_GoalId_instLT);
lp_aesop_Aesop_instInhabitedRappId_default = _init_lp_aesop_Aesop_instInhabitedRappId_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRappId_default);
lp_aesop_Aesop_instInhabitedRappId = _init_lp_aesop_Aesop_instInhabitedRappId();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRappId);
lp_aesop_Aesop_RappId_zero = _init_lp_aesop_Aesop_RappId_zero();
lean_mark_persistent(lp_aesop_Aesop_RappId_zero);
lp_aesop_Aesop_RappId_one = _init_lp_aesop_Aesop_RappId_one();
lean_mark_persistent(lp_aesop_Aesop_RappId_one);
lp_aesop_Aesop_RappId_dummy = _init_lp_aesop_Aesop_RappId_dummy();
lean_mark_persistent(lp_aesop_Aesop_RappId_dummy);
lp_aesop_Aesop_RappId_instLT = _init_lp_aesop_Aesop_RappId_instLT();
lean_mark_persistent(lp_aesop_Aesop_RappId_instLT);
lp_aesop_Aesop_instInhabitedIteration___aux__1 = _init_lp_aesop_Aesop_instInhabitedIteration___aux__1();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIteration___aux__1);
lp_aesop_Aesop_instInhabitedIteration = _init_lp_aesop_Aesop_instInhabitedIteration();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedIteration);
lp_aesop_Aesop_Iteration_one = _init_lp_aesop_Aesop_Iteration_one();
lean_mark_persistent(lp_aesop_Aesop_Iteration_one);
lp_aesop_Aesop_Iteration_none = _init_lp_aesop_Aesop_Iteration_none();
lean_mark_persistent(lp_aesop_Aesop_Iteration_none);
lp_aesop_Aesop_Iteration_instLT = _init_lp_aesop_Aesop_Iteration_instLT();
lean_mark_persistent(lp_aesop_Aesop_Iteration_instLT);
lp_aesop_Aesop_Iteration_instLE = _init_lp_aesop_Aesop_Iteration_instLE();
lean_mark_persistent(lp_aesop_Aesop_Iteration_instLE);
lp_aesop_Aesop_instInhabitedNodeState_default = _init_lp_aesop_Aesop_instInhabitedNodeState_default();
lp_aesop_Aesop_instInhabitedNodeState = _init_lp_aesop_Aesop_instInhabitedNodeState();
lp_aesop_Aesop_instInhabitedGoalState_default = _init_lp_aesop_Aesop_instInhabitedGoalState_default();
lp_aesop_Aesop_instInhabitedGoalState = _init_lp_aesop_Aesop_instInhabitedGoalState();
lp_aesop_Aesop_instInhabitedNormalizationState_default = _init_lp_aesop_Aesop_instInhabitedNormalizationState_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormalizationState_default);
lp_aesop_Aesop_instInhabitedNormalizationState = _init_lp_aesop_Aesop_instInhabitedNormalizationState();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormalizationState);
lp_aesop_Aesop_instInhabitedGoalOrigin_default = _init_lp_aesop_Aesop_instInhabitedGoalOrigin_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalOrigin_default);
lp_aesop_Aesop_instInhabitedGoalOrigin = _init_lp_aesop_Aesop_instInhabitedGoalOrigin();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGoalOrigin);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_Data(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_UnsafeQueue(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Constants(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data_ForwardRuleMatches(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_UnsafeQueue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_Data(builtin);
}
#ifdef __cplusplus
}
#endif
