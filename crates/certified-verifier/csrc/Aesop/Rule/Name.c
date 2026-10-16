// Lean compiler output
// Module: Aesop.Rule.Name
// Imports: public import Init public meta import Init public import Lean.Meta.Basic
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
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Name_cmp(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
lean_object* l_Lean_Json_mkObj(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedPhaseName_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedPhaseName;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPhaseName_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqPhaseName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqPhaseName_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqPhaseName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqPhaseName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqPhaseName = (const lean_object*)&lp_aesop_Aesop_instBEqPhaseName___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePhaseName_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashablePhaseName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashablePhaseName_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashablePhaseName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashablePhaseName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashablePhaseName = (const lean_object*)&lp_aesop_Aesop_instHashablePhaseName___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__5 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonPhaseName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonPhaseName_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonPhaseName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonPhaseName = (const lean_object*)&lp_aesop_Aesop_instToJsonPhaseName___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_PhaseName_instOrd___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_PhaseName_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_PhaseName_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PhaseName_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_PhaseName_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_PhaseName_instOrd = (const lean_object*)&lp_aesop_Aesop_PhaseName_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_PhaseName_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_PhaseName_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_PhaseName_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_PhaseName_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_PhaseName_instToString = (const lean_object*)&lp_aesop_Aesop_PhaseName_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedScopeName_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedScopeName;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqScopeName_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqScopeName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqScopeName_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqScopeName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqScopeName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqScopeName = (const lean_object*)&lp_aesop_Aesop_instBEqScopeName___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableScopeName_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashableScopeName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashableScopeName_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashableScopeName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashableScopeName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashableScopeName = (const lean_object*)&lp_aesop_Aesop_instHashableScopeName___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonScopeName_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonScopeName_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName_toJson___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonScopeName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonScopeName_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonScopeName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonScopeName = (const lean_object*)&lp_aesop_Aesop_instToJsonScopeName___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ScopeName_instOrd___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ScopeName_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ScopeName_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ScopeName_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScopeName_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ScopeName_instOrd = (const lean_object*)&lp_aesop_Aesop_ScopeName_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ScopeName_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ScopeName_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ScopeName_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScopeName_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ScopeName_instToString = (const lean_object*)&lp_aesop_Aesop_ScopeName_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedBuilderName_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedBuilderName;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqBuilderName_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqBuilderName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqBuilderName_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqBuilderName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqBuilderName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqBuilderName = (const lean_object*)&lp_aesop_Aesop_instBEqBuilderName___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableBuilderName_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashableBuilderName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashableBuilderName_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashableBuilderName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashableBuilderName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashableBuilderName = (const lean_object*)&lp_aesop_Aesop_instHashableBuilderName___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__5 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__5_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__7 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__7_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__9 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__9_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__11 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__11_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__13 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__13_value;
static const lean_string_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__15 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonBuilderName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonBuilderName_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonBuilderName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonBuilderName = (const lean_object*)&lp_aesop_Aesop_instToJsonBuilderName___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_BuilderName_instOrd___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BuilderName_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BuilderName_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BuilderName_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuilderName_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_BuilderName_instOrd = (const lean_object*)&lp_aesop_Aesop_BuilderName_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_BuilderName_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BuilderName_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BuilderName_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuilderName_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_BuilderName_instToString = (const lean_object*)&lp_aesop_Aesop_BuilderName_instToString___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__3;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__4;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__5;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleName_default___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRuleName_default___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleName_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleName;
LEAN_EXPORT uint64_t lp_aesop_Aesop_RuleName_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleName_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleName_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleName_instHashable = (const lean_object*)&lp_aesop_Aesop_RuleName_instHashable___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleName_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleName_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleName_instBEq = (const lean_object*)&lp_aesop_Aesop_RuleName_instBEq___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_compare___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_quickCompare(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_quickCompare___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleName_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_compare___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleName_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleName_instOrd = (const lean_object*)&lp_aesop_Aesop_RuleName_instOrd___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToString___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleName_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_instToString___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleName_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleName_instToString = (const lean_object*)&lp_aesop_Aesop_RuleName_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "builder"};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "phase"};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "scope"};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rendered"};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__4_value;
static const lean_array_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToJson___private__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToJson___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleName_instToJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleName_instToJson___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleName_instToJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleName_instToJson___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleName_instToJson = (const lean_object*)&lp_aesop_Aesop_RuleName_instToJson___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRuleNameForExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRuleNameForExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ruleName_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ruleName_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normSimp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normSimp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normUnfold_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normUnfold_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedDisplayRuleName_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedDisplayRuleName;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqDisplayRuleName_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqDisplayRuleName_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqDisplayRuleName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqDisplayRuleName_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqDisplayRuleName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqDisplayRuleName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqDisplayRuleName = (const lean_object*)&lp_aesop_Aesop_instBEqDisplayRuleName___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdDisplayRuleName_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdDisplayRuleName_ord___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdDisplayRuleName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdDisplayRuleName_ord___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdDisplayRuleName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdDisplayRuleName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdDisplayRuleName = (const lean_object*)&lp_aesop_Aesop_instOrdDisplayRuleName___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableDisplayRuleName_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableDisplayRuleName_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashableDisplayRuleName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashableDisplayRuleName_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashableDisplayRuleName___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashableDisplayRuleName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashableDisplayRuleName = (const lean_object*)&lp_aesop_Aesop_instHashableDisplayRuleName___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___closed__0 = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_DisplayRuleName_instCoeRuleName = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___closed__0_value;
static const lean_string_object lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "<norm simp>"};
static const lean_object* lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "<norm unfold>"};
static const lean_object* lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToString___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_DisplayRuleName_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_DisplayRuleName_instToString___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_DisplayRuleName_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_DisplayRuleName_instToString = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToString___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10_value),LEAN_SCALAR_PTR_LITERAL(195, 61, 75, 186, 44, 210, 52, 194)}};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1;
static lean_once_cell_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2;
static const lean_ctor_object lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14_value),LEAN_SCALAR_PTR_LITERAL(185, 37, 25, 138, 30, 217, 227, 180)}};
static const lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___private__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___private__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_DisplayRuleName_instToJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToJson___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson = (const lean_object*)&lp_aesop_Aesop_DisplayRuleName_instToJson___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorIdx(uint8_t v_x_1_){
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
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
uint8_t v_x_boxed_6_; lean_object* v_res_7_; 
v_x_boxed_6_ = lean_unbox(v_x_5_);
v_res_7_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_x_boxed_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___redArg(lean_object* v_k_8_){
_start:
{
lean_inc(v_k_8_);
return v_k_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___redArg___boxed(lean_object* v_k_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop_Aesop_PhaseName_ctorElim___redArg(v_k_9_);
lean_dec(v_k_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, uint8_t v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_inc(v_k_15_);
return v_k_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
uint8_t v_t_boxed_21_; lean_object* v_res_22_; 
v_t_boxed_21_ = lean_unbox(v_t_18_);
v_res_22_ = lp_aesop_Aesop_PhaseName_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_boxed_21_, v_h_19_, v_k_20_);
lean_dec(v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___redArg(lean_object* v_norm_23_){
_start:
{
lean_inc(v_norm_23_);
return v_norm_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___redArg___boxed(lean_object* v_norm_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_PhaseName_norm_elim___redArg(v_norm_24_);
lean_dec(v_norm_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim(lean_object* v_motive_26_, uint8_t v_t_27_, lean_object* v_h_28_, lean_object* v_norm_29_){
_start:
{
lean_inc(v_norm_29_);
return v_norm_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_norm_elim___boxed(lean_object* v_motive_30_, lean_object* v_t_31_, lean_object* v_h_32_, lean_object* v_norm_33_){
_start:
{
uint8_t v_t_boxed_34_; lean_object* v_res_35_; 
v_t_boxed_34_ = lean_unbox(v_t_31_);
v_res_35_ = lp_aesop_Aesop_PhaseName_norm_elim(v_motive_30_, v_t_boxed_34_, v_h_32_, v_norm_33_);
lean_dec(v_norm_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___redArg(lean_object* v_safe_36_){
_start:
{
lean_inc(v_safe_36_);
return v_safe_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___redArg___boxed(lean_object* v_safe_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Aesop_PhaseName_safe_elim___redArg(v_safe_37_);
lean_dec(v_safe_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim(lean_object* v_motive_39_, uint8_t v_t_40_, lean_object* v_h_41_, lean_object* v_safe_42_){
_start:
{
lean_inc(v_safe_42_);
return v_safe_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_safe_elim___boxed(lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_safe_46_){
_start:
{
uint8_t v_t_boxed_47_; lean_object* v_res_48_; 
v_t_boxed_47_ = lean_unbox(v_t_44_);
v_res_48_ = lp_aesop_Aesop_PhaseName_safe_elim(v_motive_43_, v_t_boxed_47_, v_h_45_, v_safe_46_);
lean_dec(v_safe_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___redArg(lean_object* v_unsafe_49_){
_start:
{
lean_inc(v_unsafe_49_);
return v_unsafe_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___redArg___boxed(lean_object* v_unsafe_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_aesop_Aesop_PhaseName_unsafe_elim___redArg(v_unsafe_50_);
lean_dec(v_unsafe_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim(lean_object* v_motive_52_, uint8_t v_t_53_, lean_object* v_h_54_, lean_object* v_unsafe_55_){
_start:
{
lean_inc(v_unsafe_55_);
return v_unsafe_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_unsafe_elim___boxed(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_unsafe_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_aesop_Aesop_PhaseName_unsafe_elim(v_motive_56_, v_t_boxed_60_, v_h_58_, v_unsafe_59_);
lean_dec(v_unsafe_59_);
return v_res_61_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedPhaseName_default(void){
_start:
{
uint8_t v___x_62_; 
v___x_62_ = 0;
return v___x_62_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedPhaseName(void){
_start:
{
uint8_t v___x_63_; 
v___x_63_ = 0;
return v___x_63_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t v_x_64_, uint8_t v_y_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_66_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_x_64_);
v___x_67_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_y_65_);
v___x_68_ = lean_nat_dec_eq(v___x_66_, v___x_67_);
lean_dec(v___x_67_);
lean_dec(v___x_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqPhaseName_beq___boxed(lean_object* v_x_69_, lean_object* v_y_70_){
_start:
{
uint8_t v_x_17__boxed_71_; uint8_t v_y_18__boxed_72_; uint8_t v_res_73_; lean_object* v_r_74_; 
v_x_17__boxed_71_ = lean_unbox(v_x_69_);
v_y_18__boxed_72_ = lean_unbox(v_y_70_);
v_res_73_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_x_17__boxed_71_, v_y_18__boxed_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t v_x_77_){
_start:
{
switch(v_x_77_)
{
case 0:
{
uint64_t v___x_78_; 
v___x_78_ = 0ULL;
return v___x_78_;
}
case 1:
{
uint64_t v___x_79_; 
v___x_79_ = 1ULL;
return v___x_79_;
}
default: 
{
uint64_t v___x_80_; 
v___x_80_ = 2ULL;
return v___x_80_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashablePhaseName_hash___boxed(lean_object* v_x_81_){
_start:
{
uint8_t v_x_40__boxed_82_; uint64_t v_res_83_; lean_object* v_r_84_; 
v_x_40__boxed_82_ = lean_unbox(v_x_81_);
v_res_83_ = lp_aesop_Aesop_instHashablePhaseName_hash(v_x_40__boxed_82_);
v_r_84_ = lean_box_uint64(v_res_83_);
return v_r_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson(uint8_t v_x_96_){
_start:
{
switch(v_x_96_)
{
case 0:
{
lean_object* v___x_97_; 
v___x_97_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__1));
return v___x_97_;
}
case 1:
{
lean_object* v___x_98_; 
v___x_98_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__3));
return v___x_98_;
}
default: 
{
lean_object* v___x_99_; 
v___x_99_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__5));
return v___x_99_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonPhaseName_toJson___boxed(lean_object* v_x_100_){
_start:
{
uint8_t v_x_67__boxed_101_; lean_object* v_res_102_; 
v_x_67__boxed_101_ = lean_unbox(v_x_100_);
v_res_102_ = lp_aesop_Aesop_instToJsonPhaseName_toJson(v_x_67__boxed_101_);
return v_res_102_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_PhaseName_instOrd___lam__0(uint8_t v_s_u2081_105_, uint8_t v_s_u2082_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_107_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_s_u2081_105_);
v___x_108_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_s_u2082_106_);
v___x_109_ = lean_nat_dec_lt(v___x_107_, v___x_108_);
if (v___x_109_ == 0)
{
uint8_t v___x_110_; 
v___x_110_ = lean_nat_dec_eq(v___x_107_, v___x_108_);
lean_dec(v___x_108_);
lean_dec(v___x_107_);
if (v___x_110_ == 0)
{
uint8_t v___x_111_; 
v___x_111_ = 2;
return v___x_111_;
}
else
{
uint8_t v___x_112_; 
v___x_112_ = 1;
return v___x_112_;
}
}
else
{
uint8_t v___x_113_; 
lean_dec(v___x_108_);
lean_dec(v___x_107_);
v___x_113_ = 0;
return v___x_113_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instOrd___lam__0___boxed(lean_object* v_s_u2081_114_, lean_object* v_s_u2082_115_){
_start:
{
uint8_t v_s_u2081_boxed_116_; uint8_t v_s_u2082_boxed_117_; uint8_t v_res_118_; lean_object* v_r_119_; 
v_s_u2081_boxed_116_ = lean_unbox(v_s_u2081_114_);
v_s_u2082_boxed_117_ = lean_unbox(v_s_u2082_115_);
v_res_118_ = lp_aesop_Aesop_PhaseName_instOrd___lam__0(v_s_u2081_boxed_116_, v_s_u2082_boxed_117_);
v_r_119_ = lean_box(v_res_118_);
return v_r_119_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instToString___lam__0(uint8_t v_x_122_){
_start:
{
switch(v_x_122_)
{
case 0:
{
lean_object* v___x_123_; 
v___x_123_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
return v___x_123_;
}
case 1:
{
lean_object* v___x_124_; 
v___x_124_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
return v___x_124_;
}
default: 
{
lean_object* v___x_125_; 
v___x_125_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
return v___x_125_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseName_instToString___lam__0___boxed(lean_object* v_x_126_){
_start:
{
uint8_t v_x_33__boxed_127_; lean_object* v_res_128_; 
v_x_33__boxed_127_ = lean_unbox(v_x_126_);
v_res_128_ = lp_aesop_Aesop_PhaseName_instToString___lam__0(v_x_33__boxed_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorIdx(uint8_t v_x_131_){
_start:
{
if (v_x_131_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = lean_unsigned_to_nat(0u);
return v___x_132_;
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_unsigned_to_nat(1u);
return v___x_133_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorIdx___boxed(lean_object* v_x_134_){
_start:
{
uint8_t v_x_boxed_135_; lean_object* v_res_136_; 
v_x_boxed_135_ = lean_unbox(v_x_134_);
v_res_136_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_x_boxed_135_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___redArg(lean_object* v_k_137_){
_start:
{
lean_inc(v_k_137_);
return v_k_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___redArg___boxed(lean_object* v_k_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_aesop_Aesop_ScopeName_ctorElim___redArg(v_k_138_);
lean_dec(v_k_138_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim(lean_object* v_motive_140_, lean_object* v_ctorIdx_141_, uint8_t v_t_142_, lean_object* v_h_143_, lean_object* v_k_144_){
_start:
{
lean_inc(v_k_144_);
return v_k_144_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_ctorElim___boxed(lean_object* v_motive_145_, lean_object* v_ctorIdx_146_, lean_object* v_t_147_, lean_object* v_h_148_, lean_object* v_k_149_){
_start:
{
uint8_t v_t_boxed_150_; lean_object* v_res_151_; 
v_t_boxed_150_ = lean_unbox(v_t_147_);
v_res_151_ = lp_aesop_Aesop_ScopeName_ctorElim(v_motive_145_, v_ctorIdx_146_, v_t_boxed_150_, v_h_148_, v_k_149_);
lean_dec(v_k_149_);
lean_dec(v_ctorIdx_146_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___redArg(lean_object* v_global_152_){
_start:
{
lean_inc(v_global_152_);
return v_global_152_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___redArg___boxed(lean_object* v_global_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_aesop_Aesop_ScopeName_global_elim___redArg(v_global_153_);
lean_dec(v_global_153_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim(lean_object* v_motive_155_, uint8_t v_t_156_, lean_object* v_h_157_, lean_object* v_global_158_){
_start:
{
lean_inc(v_global_158_);
return v_global_158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_global_elim___boxed(lean_object* v_motive_159_, lean_object* v_t_160_, lean_object* v_h_161_, lean_object* v_global_162_){
_start:
{
uint8_t v_t_boxed_163_; lean_object* v_res_164_; 
v_t_boxed_163_ = lean_unbox(v_t_160_);
v_res_164_ = lp_aesop_Aesop_ScopeName_global_elim(v_motive_159_, v_t_boxed_163_, v_h_161_, v_global_162_);
lean_dec(v_global_162_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___redArg(lean_object* v_local_165_){
_start:
{
lean_inc(v_local_165_);
return v_local_165_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___redArg___boxed(lean_object* v_local_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_aesop_Aesop_ScopeName_local_elim___redArg(v_local_166_);
lean_dec(v_local_166_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim(lean_object* v_motive_168_, uint8_t v_t_169_, lean_object* v_h_170_, lean_object* v_local_171_){
_start:
{
lean_inc(v_local_171_);
return v_local_171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_local_elim___boxed(lean_object* v_motive_172_, lean_object* v_t_173_, lean_object* v_h_174_, lean_object* v_local_175_){
_start:
{
uint8_t v_t_boxed_176_; lean_object* v_res_177_; 
v_t_boxed_176_ = lean_unbox(v_t_173_);
v_res_177_ = lp_aesop_Aesop_ScopeName_local_elim(v_motive_172_, v_t_boxed_176_, v_h_174_, v_local_175_);
lean_dec(v_local_175_);
return v_res_177_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedScopeName_default(void){
_start:
{
uint8_t v___x_178_; 
v___x_178_ = 0;
return v___x_178_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedScopeName(void){
_start:
{
uint8_t v___x_179_; 
v___x_179_ = 0;
return v___x_179_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t v_x_180_, uint8_t v_y_181_){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_182_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_x_180_);
v___x_183_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_y_181_);
v___x_184_ = lean_nat_dec_eq(v___x_182_, v___x_183_);
lean_dec(v___x_183_);
lean_dec(v___x_182_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqScopeName_beq___boxed(lean_object* v_x_185_, lean_object* v_y_186_){
_start:
{
uint8_t v_x_17__boxed_187_; uint8_t v_y_18__boxed_188_; uint8_t v_res_189_; lean_object* v_r_190_; 
v_x_17__boxed_187_ = lean_unbox(v_x_185_);
v_y_18__boxed_188_ = lean_unbox(v_y_186_);
v_res_189_ = lp_aesop_Aesop_instBEqScopeName_beq(v_x_17__boxed_187_, v_y_18__boxed_188_);
v_r_190_ = lean_box(v_res_189_);
return v_r_190_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t v_x_193_){
_start:
{
if (v_x_193_ == 0)
{
uint64_t v___x_194_; 
v___x_194_ = 0ULL;
return v___x_194_;
}
else
{
uint64_t v___x_195_; 
v___x_195_ = 1ULL;
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableScopeName_hash___boxed(lean_object* v_x_196_){
_start:
{
uint8_t v_x_28__boxed_197_; uint64_t v_res_198_; lean_object* v_r_199_; 
v_x_28__boxed_197_ = lean_unbox(v_x_196_);
v_res_198_ = lp_aesop_Aesop_instHashableScopeName_hash(v_x_28__boxed_197_);
v_r_199_ = lean_box_uint64(v_res_198_);
return v_r_199_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson(uint8_t v_x_208_){
_start:
{
if (v_x_208_ == 0)
{
lean_object* v___x_209_; 
v___x_209_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__1));
return v___x_209_;
}
else
{
lean_object* v___x_210_; 
v___x_210_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__3));
return v___x_210_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScopeName_toJson___boxed(lean_object* v_x_211_){
_start:
{
uint8_t v_x_46__boxed_212_; lean_object* v_res_213_; 
v_x_46__boxed_212_ = lean_unbox(v_x_211_);
v_res_213_ = lp_aesop_Aesop_instToJsonScopeName_toJson(v_x_46__boxed_212_);
return v_res_213_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ScopeName_instOrd___lam__0(uint8_t v_s_u2081_216_, uint8_t v_s_u2082_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; uint8_t v___x_220_; 
v___x_218_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_s_u2081_216_);
v___x_219_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_s_u2082_217_);
v___x_220_ = lean_nat_dec_lt(v___x_218_, v___x_219_);
if (v___x_220_ == 0)
{
uint8_t v___x_221_; 
v___x_221_ = lean_nat_dec_eq(v___x_218_, v___x_219_);
lean_dec(v___x_219_);
lean_dec(v___x_218_);
if (v___x_221_ == 0)
{
uint8_t v___x_222_; 
v___x_222_ = 2;
return v___x_222_;
}
else
{
uint8_t v___x_223_; 
v___x_223_ = 1;
return v___x_223_;
}
}
else
{
uint8_t v___x_224_; 
lean_dec(v___x_219_);
lean_dec(v___x_218_);
v___x_224_ = 0;
return v___x_224_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instOrd___lam__0___boxed(lean_object* v_s_u2081_225_, lean_object* v_s_u2082_226_){
_start:
{
uint8_t v_s_u2081_boxed_227_; uint8_t v_s_u2082_boxed_228_; uint8_t v_res_229_; lean_object* v_r_230_; 
v_s_u2081_boxed_227_ = lean_unbox(v_s_u2081_225_);
v_s_u2082_boxed_228_ = lean_unbox(v_s_u2082_226_);
v_res_229_ = lp_aesop_Aesop_ScopeName_instOrd___lam__0(v_s_u2081_boxed_227_, v_s_u2082_boxed_228_);
v_r_230_ = lean_box(v_res_229_);
return v_r_230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instToString___lam__0(uint8_t v_x_233_){
_start:
{
if (v_x_233_ == 0)
{
lean_object* v___x_234_; 
v___x_234_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
return v___x_234_;
}
else
{
lean_object* v___x_235_; 
v___x_235_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
return v___x_235_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScopeName_instToString___lam__0___boxed(lean_object* v_x_236_){
_start:
{
uint8_t v_x_24__boxed_237_; lean_object* v_res_238_; 
v_x_24__boxed_237_ = lean_unbox(v_x_236_);
v_res_238_ = lp_aesop_Aesop_ScopeName_instToString___lam__0(v_x_24__boxed_237_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorIdx(uint8_t v_x_241_){
_start:
{
switch(v_x_241_)
{
case 0:
{
lean_object* v___x_242_; 
v___x_242_ = lean_unsigned_to_nat(0u);
return v___x_242_;
}
case 1:
{
lean_object* v___x_243_; 
v___x_243_ = lean_unsigned_to_nat(1u);
return v___x_243_;
}
case 2:
{
lean_object* v___x_244_; 
v___x_244_ = lean_unsigned_to_nat(2u);
return v___x_244_;
}
case 3:
{
lean_object* v___x_245_; 
v___x_245_ = lean_unsigned_to_nat(3u);
return v___x_245_;
}
case 4:
{
lean_object* v___x_246_; 
v___x_246_ = lean_unsigned_to_nat(4u);
return v___x_246_;
}
case 5:
{
lean_object* v___x_247_; 
v___x_247_ = lean_unsigned_to_nat(5u);
return v___x_247_;
}
case 6:
{
lean_object* v___x_248_; 
v___x_248_ = lean_unsigned_to_nat(6u);
return v___x_248_;
}
default: 
{
lean_object* v___x_249_; 
v___x_249_ = lean_unsigned_to_nat(7u);
return v___x_249_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorIdx___boxed(lean_object* v_x_250_){
_start:
{
uint8_t v_x_boxed_251_; lean_object* v_res_252_; 
v_x_boxed_251_ = lean_unbox(v_x_250_);
v_res_252_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_x_boxed_251_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___redArg(lean_object* v_k_253_){
_start:
{
lean_inc(v_k_253_);
return v_k_253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___redArg___boxed(lean_object* v_k_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_aesop_Aesop_BuilderName_ctorElim___redArg(v_k_254_);
lean_dec(v_k_254_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim(lean_object* v_motive_256_, lean_object* v_ctorIdx_257_, uint8_t v_t_258_, lean_object* v_h_259_, lean_object* v_k_260_){
_start:
{
lean_inc(v_k_260_);
return v_k_260_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_ctorElim___boxed(lean_object* v_motive_261_, lean_object* v_ctorIdx_262_, lean_object* v_t_263_, lean_object* v_h_264_, lean_object* v_k_265_){
_start:
{
uint8_t v_t_boxed_266_; lean_object* v_res_267_; 
v_t_boxed_266_ = lean_unbox(v_t_263_);
v_res_267_ = lp_aesop_Aesop_BuilderName_ctorElim(v_motive_261_, v_ctorIdx_262_, v_t_boxed_266_, v_h_264_, v_k_265_);
lean_dec(v_k_265_);
lean_dec(v_ctorIdx_262_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___redArg(lean_object* v_apply_268_){
_start:
{
lean_inc(v_apply_268_);
return v_apply_268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___redArg___boxed(lean_object* v_apply_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_aesop_Aesop_BuilderName_apply_elim___redArg(v_apply_269_);
lean_dec(v_apply_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim(lean_object* v_motive_271_, uint8_t v_t_272_, lean_object* v_h_273_, lean_object* v_apply_274_){
_start:
{
lean_inc(v_apply_274_);
return v_apply_274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_apply_elim___boxed(lean_object* v_motive_275_, lean_object* v_t_276_, lean_object* v_h_277_, lean_object* v_apply_278_){
_start:
{
uint8_t v_t_boxed_279_; lean_object* v_res_280_; 
v_t_boxed_279_ = lean_unbox(v_t_276_);
v_res_280_ = lp_aesop_Aesop_BuilderName_apply_elim(v_motive_275_, v_t_boxed_279_, v_h_277_, v_apply_278_);
lean_dec(v_apply_278_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___redArg(lean_object* v_cases_281_){
_start:
{
lean_inc(v_cases_281_);
return v_cases_281_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___redArg___boxed(lean_object* v_cases_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_aesop_Aesop_BuilderName_cases_elim___redArg(v_cases_282_);
lean_dec(v_cases_282_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim(lean_object* v_motive_284_, uint8_t v_t_285_, lean_object* v_h_286_, lean_object* v_cases_287_){
_start:
{
lean_inc(v_cases_287_);
return v_cases_287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_cases_elim___boxed(lean_object* v_motive_288_, lean_object* v_t_289_, lean_object* v_h_290_, lean_object* v_cases_291_){
_start:
{
uint8_t v_t_boxed_292_; lean_object* v_res_293_; 
v_t_boxed_292_ = lean_unbox(v_t_289_);
v_res_293_ = lp_aesop_Aesop_BuilderName_cases_elim(v_motive_288_, v_t_boxed_292_, v_h_290_, v_cases_291_);
lean_dec(v_cases_291_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___redArg(lean_object* v_constructors_294_){
_start:
{
lean_inc(v_constructors_294_);
return v_constructors_294_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___redArg___boxed(lean_object* v_constructors_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_aesop_Aesop_BuilderName_constructors_elim___redArg(v_constructors_295_);
lean_dec(v_constructors_295_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim(lean_object* v_motive_297_, uint8_t v_t_298_, lean_object* v_h_299_, lean_object* v_constructors_300_){
_start:
{
lean_inc(v_constructors_300_);
return v_constructors_300_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_constructors_elim___boxed(lean_object* v_motive_301_, lean_object* v_t_302_, lean_object* v_h_303_, lean_object* v_constructors_304_){
_start:
{
uint8_t v_t_boxed_305_; lean_object* v_res_306_; 
v_t_boxed_305_ = lean_unbox(v_t_302_);
v_res_306_ = lp_aesop_Aesop_BuilderName_constructors_elim(v_motive_301_, v_t_boxed_305_, v_h_303_, v_constructors_304_);
lean_dec(v_constructors_304_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___redArg(lean_object* v_destruct_307_){
_start:
{
lean_inc(v_destruct_307_);
return v_destruct_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___redArg___boxed(lean_object* v_destruct_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_aesop_Aesop_BuilderName_destruct_elim___redArg(v_destruct_308_);
lean_dec(v_destruct_308_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim(lean_object* v_motive_310_, uint8_t v_t_311_, lean_object* v_h_312_, lean_object* v_destruct_313_){
_start:
{
lean_inc(v_destruct_313_);
return v_destruct_313_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_destruct_elim___boxed(lean_object* v_motive_314_, lean_object* v_t_315_, lean_object* v_h_316_, lean_object* v_destruct_317_){
_start:
{
uint8_t v_t_boxed_318_; lean_object* v_res_319_; 
v_t_boxed_318_ = lean_unbox(v_t_315_);
v_res_319_ = lp_aesop_Aesop_BuilderName_destruct_elim(v_motive_314_, v_t_boxed_318_, v_h_316_, v_destruct_317_);
lean_dec(v_destruct_317_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___redArg(lean_object* v_forward_320_){
_start:
{
lean_inc(v_forward_320_);
return v_forward_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___redArg___boxed(lean_object* v_forward_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_aesop_Aesop_BuilderName_forward_elim___redArg(v_forward_321_);
lean_dec(v_forward_321_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim(lean_object* v_motive_323_, uint8_t v_t_324_, lean_object* v_h_325_, lean_object* v_forward_326_){
_start:
{
lean_inc(v_forward_326_);
return v_forward_326_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_forward_elim___boxed(lean_object* v_motive_327_, lean_object* v_t_328_, lean_object* v_h_329_, lean_object* v_forward_330_){
_start:
{
uint8_t v_t_boxed_331_; lean_object* v_res_332_; 
v_t_boxed_331_ = lean_unbox(v_t_328_);
v_res_332_ = lp_aesop_Aesop_BuilderName_forward_elim(v_motive_327_, v_t_boxed_331_, v_h_329_, v_forward_330_);
lean_dec(v_forward_330_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___redArg(lean_object* v_simp_333_){
_start:
{
lean_inc(v_simp_333_);
return v_simp_333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___redArg___boxed(lean_object* v_simp_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_aesop_Aesop_BuilderName_simp_elim___redArg(v_simp_334_);
lean_dec(v_simp_334_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim(lean_object* v_motive_336_, uint8_t v_t_337_, lean_object* v_h_338_, lean_object* v_simp_339_){
_start:
{
lean_inc(v_simp_339_);
return v_simp_339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_simp_elim___boxed(lean_object* v_motive_340_, lean_object* v_t_341_, lean_object* v_h_342_, lean_object* v_simp_343_){
_start:
{
uint8_t v_t_boxed_344_; lean_object* v_res_345_; 
v_t_boxed_344_ = lean_unbox(v_t_341_);
v_res_345_ = lp_aesop_Aesop_BuilderName_simp_elim(v_motive_340_, v_t_boxed_344_, v_h_342_, v_simp_343_);
lean_dec(v_simp_343_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___redArg(lean_object* v_tactic_346_){
_start:
{
lean_inc(v_tactic_346_);
return v_tactic_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___redArg___boxed(lean_object* v_tactic_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_aesop_Aesop_BuilderName_tactic_elim___redArg(v_tactic_347_);
lean_dec(v_tactic_347_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim(lean_object* v_motive_349_, uint8_t v_t_350_, lean_object* v_h_351_, lean_object* v_tactic_352_){
_start:
{
lean_inc(v_tactic_352_);
return v_tactic_352_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_tactic_elim___boxed(lean_object* v_motive_353_, lean_object* v_t_354_, lean_object* v_h_355_, lean_object* v_tactic_356_){
_start:
{
uint8_t v_t_boxed_357_; lean_object* v_res_358_; 
v_t_boxed_357_ = lean_unbox(v_t_354_);
v_res_358_ = lp_aesop_Aesop_BuilderName_tactic_elim(v_motive_353_, v_t_boxed_357_, v_h_355_, v_tactic_356_);
lean_dec(v_tactic_356_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___redArg(lean_object* v_unfold_359_){
_start:
{
lean_inc(v_unfold_359_);
return v_unfold_359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___redArg___boxed(lean_object* v_unfold_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_aesop_Aesop_BuilderName_unfold_elim___redArg(v_unfold_360_);
lean_dec(v_unfold_360_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim(lean_object* v_motive_362_, uint8_t v_t_363_, lean_object* v_h_364_, lean_object* v_unfold_365_){
_start:
{
lean_inc(v_unfold_365_);
return v_unfold_365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_unfold_elim___boxed(lean_object* v_motive_366_, lean_object* v_t_367_, lean_object* v_h_368_, lean_object* v_unfold_369_){
_start:
{
uint8_t v_t_boxed_370_; lean_object* v_res_371_; 
v_t_boxed_370_ = lean_unbox(v_t_367_);
v_res_371_ = lp_aesop_Aesop_BuilderName_unfold_elim(v_motive_366_, v_t_boxed_370_, v_h_368_, v_unfold_369_);
lean_dec(v_unfold_369_);
return v_res_371_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedBuilderName_default(void){
_start:
{
uint8_t v___x_372_; 
v___x_372_ = 0;
return v___x_372_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedBuilderName(void){
_start:
{
uint8_t v___x_373_; 
v___x_373_ = 0;
return v___x_373_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t v_x_374_, uint8_t v_y_375_){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; uint8_t v___x_378_; 
v___x_376_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_x_374_);
v___x_377_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_y_375_);
v___x_378_ = lean_nat_dec_eq(v___x_376_, v___x_377_);
lean_dec(v___x_377_);
lean_dec(v___x_376_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqBuilderName_beq___boxed(lean_object* v_x_379_, lean_object* v_y_380_){
_start:
{
uint8_t v_x_17__boxed_381_; uint8_t v_y_18__boxed_382_; uint8_t v_res_383_; lean_object* v_r_384_; 
v_x_17__boxed_381_ = lean_unbox(v_x_379_);
v_y_18__boxed_382_ = lean_unbox(v_y_380_);
v_res_383_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_x_17__boxed_381_, v_y_18__boxed_382_);
v_r_384_ = lean_box(v_res_383_);
return v_r_384_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t v_x_387_){
_start:
{
switch(v_x_387_)
{
case 0:
{
uint64_t v___x_388_; 
v___x_388_ = 0ULL;
return v___x_388_;
}
case 1:
{
uint64_t v___x_389_; 
v___x_389_ = 1ULL;
return v___x_389_;
}
case 2:
{
uint64_t v___x_390_; 
v___x_390_ = 2ULL;
return v___x_390_;
}
case 3:
{
uint64_t v___x_391_; 
v___x_391_ = 3ULL;
return v___x_391_;
}
case 4:
{
uint64_t v___x_392_; 
v___x_392_ = 4ULL;
return v___x_392_;
}
case 5:
{
uint64_t v___x_393_; 
v___x_393_ = 5ULL;
return v___x_393_;
}
case 6:
{
uint64_t v___x_394_; 
v___x_394_ = 6ULL;
return v___x_394_;
}
default: 
{
uint64_t v___x_395_; 
v___x_395_ = 7ULL;
return v___x_395_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableBuilderName_hash___boxed(lean_object* v_x_396_){
_start:
{
uint8_t v_x_100__boxed_397_; uint64_t v_res_398_; lean_object* v_r_399_; 
v_x_100__boxed_397_ = lean_unbox(v_x_396_);
v_res_398_ = lp_aesop_Aesop_instHashableBuilderName_hash(v_x_100__boxed_397_);
v_r_399_ = lean_box_uint64(v_res_398_);
return v_r_399_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson(uint8_t v_x_426_){
_start:
{
switch(v_x_426_)
{
case 0:
{
lean_object* v___x_427_; 
v___x_427_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__1));
return v___x_427_;
}
case 1:
{
lean_object* v___x_428_; 
v___x_428_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__3));
return v___x_428_;
}
case 2:
{
lean_object* v___x_429_; 
v___x_429_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__5));
return v___x_429_;
}
case 3:
{
lean_object* v___x_430_; 
v___x_430_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__7));
return v___x_430_;
}
case 4:
{
lean_object* v___x_431_; 
v___x_431_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__9));
return v___x_431_;
}
case 5:
{
lean_object* v___x_432_; 
v___x_432_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__11));
return v___x_432_;
}
case 6:
{
lean_object* v___x_433_; 
v___x_433_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__13));
return v___x_433_;
}
default: 
{
lean_object* v___x_434_; 
v___x_434_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__15));
return v___x_434_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonBuilderName_toJson___boxed(lean_object* v_x_435_){
_start:
{
uint8_t v_x_172__boxed_436_; lean_object* v_res_437_; 
v_x_172__boxed_436_ = lean_unbox(v_x_435_);
v_res_437_ = lp_aesop_Aesop_instToJsonBuilderName_toJson(v_x_172__boxed_436_);
return v_res_437_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_BuilderName_instOrd___lam__0(uint8_t v_b_u2081_440_, uint8_t v_b_u2082_441_){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; uint8_t v___x_444_; 
v___x_442_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_b_u2081_440_);
v___x_443_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_b_u2082_441_);
v___x_444_ = lean_nat_dec_lt(v___x_442_, v___x_443_);
if (v___x_444_ == 0)
{
uint8_t v___x_445_; 
v___x_445_ = lean_nat_dec_eq(v___x_442_, v___x_443_);
lean_dec(v___x_443_);
lean_dec(v___x_442_);
if (v___x_445_ == 0)
{
uint8_t v___x_446_; 
v___x_446_ = 2;
return v___x_446_;
}
else
{
uint8_t v___x_447_; 
v___x_447_ = 1;
return v___x_447_;
}
}
else
{
uint8_t v___x_448_; 
lean_dec(v___x_443_);
lean_dec(v___x_442_);
v___x_448_ = 0;
return v___x_448_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instOrd___lam__0___boxed(lean_object* v_b_u2081_449_, lean_object* v_b_u2082_450_){
_start:
{
uint8_t v_b_u2081_boxed_451_; uint8_t v_b_u2082_boxed_452_; uint8_t v_res_453_; lean_object* v_r_454_; 
v_b_u2081_boxed_451_ = lean_unbox(v_b_u2081_449_);
v_b_u2082_boxed_452_ = lean_unbox(v_b_u2082_450_);
v_res_453_ = lp_aesop_Aesop_BuilderName_instOrd___lam__0(v_b_u2081_boxed_451_, v_b_u2082_boxed_452_);
v_r_454_ = lean_box(v_res_453_);
return v_r_454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instToString___lam__0(uint8_t v_x_457_){
_start:
{
switch(v_x_457_)
{
case 0:
{
lean_object* v___x_458_; 
v___x_458_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
return v___x_458_;
}
case 1:
{
lean_object* v___x_459_; 
v___x_459_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
return v___x_459_;
}
case 2:
{
lean_object* v___x_460_; 
v___x_460_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
return v___x_460_;
}
case 3:
{
lean_object* v___x_461_; 
v___x_461_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
return v___x_461_;
}
case 4:
{
lean_object* v___x_462_; 
v___x_462_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
return v___x_462_;
}
case 5:
{
lean_object* v___x_463_; 
v___x_463_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
return v___x_463_;
}
case 6:
{
lean_object* v___x_464_; 
v___x_464_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
return v___x_464_;
}
default: 
{
lean_object* v___x_465_; 
v___x_465_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
return v___x_465_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuilderName_instToString___lam__0___boxed(lean_object* v_x_466_){
_start:
{
uint8_t v_x_78__boxed_467_; lean_object* v_res_468_; 
v_x_78__boxed_467_ = lean_unbox(v_x_466_);
v_res_468_ = lp_aesop_Aesop_BuilderName_instToString___lam__0(v_x_78__boxed_467_);
return v_res_468_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__0(void){
_start:
{
uint8_t v___x_471_; uint64_t v___x_472_; 
v___x_471_ = 0;
v___x_472_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_471_);
return v___x_472_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__1(void){
_start:
{
uint8_t v___x_473_; uint64_t v___x_474_; 
v___x_473_ = 0;
v___x_474_ = lp_aesop_Aesop_instHashablePhaseName_hash(v___x_473_);
return v___x_474_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__2(void){
_start:
{
uint8_t v___x_475_; uint64_t v___x_476_; 
v___x_475_ = 0;
v___x_476_ = lp_aesop_Aesop_instHashableScopeName_hash(v___x_475_);
return v___x_476_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__3(void){
_start:
{
uint64_t v___x_477_; uint64_t v___x_478_; uint64_t v___x_479_; 
v___x_477_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__2, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__2);
v___x_478_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__1, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__1);
v___x_479_ = lean_uint64_mix_hash(v___x_478_, v___x_477_);
return v___x_479_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__4(void){
_start:
{
uint64_t v___x_480_; uint64_t v___x_481_; uint64_t v___x_482_; 
v___x_480_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__3, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__3);
v___x_481_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__0, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__0);
v___x_482_ = lean_uint64_mix_hash(v___x_481_, v___x_480_);
return v___x_482_;
}
}
static uint64_t _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__5(void){
_start:
{
uint64_t v___x_483_; uint64_t v___x_484_; uint64_t v___x_485_; 
v___x_483_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__4, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__4_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__4);
v___x_484_ = 1723ULL;
v___x_485_ = lean_uint64_mix_hash(v___x_484_, v___x_483_);
return v___x_485_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__6(void){
_start:
{
uint64_t v___x_486_; uint8_t v___x_487_; uint8_t v___x_488_; uint8_t v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_486_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__5, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__5_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__5);
v___x_487_ = 0;
v___x_488_ = 0;
v___x_489_ = 0;
v___x_490_ = lean_box(0);
v___x_491_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_491_, 0, v___x_490_);
lean_ctor_set_uint8(v___x_491_, sizeof(void*)*1 + 8, v___x_489_);
lean_ctor_set_uint8(v___x_491_, sizeof(void*)*1 + 9, v___x_488_);
lean_ctor_set_uint8(v___x_491_, sizeof(void*)*1 + 10, v___x_487_);
lean_ctor_set_uint64(v___x_491_, sizeof(void*)*1, v___x_486_);
return v___x_491_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleName_default(void){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__6, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__6_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__6);
return v___x_492_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleName(void){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_aesop_Aesop_instInhabitedRuleName_default;
return v___x_493_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_RuleName_instHashable___lam__0(lean_object* v_n_494_){
_start:
{
uint64_t v_hash_495_; 
v_hash_495_ = lean_ctor_get_uint64(v_n_494_, sizeof(void*)*1);
return v_hash_495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instHashable___lam__0___boxed(lean_object* v_n_496_){
_start:
{
uint64_t v_res_497_; lean_object* v_r_498_; 
v_res_497_ = lp_aesop_Aesop_RuleName_instHashable___lam__0(v_n_496_);
lean_dec_ref(v_n_496_);
v_r_498_ = lean_box_uint64(v_res_497_);
return v_r_498_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_instBEq___lam__0(lean_object* v_n_u2081_501_, lean_object* v_n_u2082_502_){
_start:
{
lean_object* v_name_503_; uint8_t v_builder_504_; uint8_t v_phase_505_; uint8_t v_scope_506_; uint64_t v_hash_507_; lean_object* v_name_508_; uint8_t v_builder_509_; uint8_t v_phase_510_; uint8_t v_scope_511_; uint64_t v_hash_512_; uint8_t v___x_513_; 
v_name_503_ = lean_ctor_get(v_n_u2081_501_, 0);
v_builder_504_ = lean_ctor_get_uint8(v_n_u2081_501_, sizeof(void*)*1 + 8);
v_phase_505_ = lean_ctor_get_uint8(v_n_u2081_501_, sizeof(void*)*1 + 9);
v_scope_506_ = lean_ctor_get_uint8(v_n_u2081_501_, sizeof(void*)*1 + 10);
v_hash_507_ = lean_ctor_get_uint64(v_n_u2081_501_, sizeof(void*)*1);
v_name_508_ = lean_ctor_get(v_n_u2082_502_, 0);
v_builder_509_ = lean_ctor_get_uint8(v_n_u2082_502_, sizeof(void*)*1 + 8);
v_phase_510_ = lean_ctor_get_uint8(v_n_u2082_502_, sizeof(void*)*1 + 9);
v_scope_511_ = lean_ctor_get_uint8(v_n_u2082_502_, sizeof(void*)*1 + 10);
v_hash_512_ = lean_ctor_get_uint64(v_n_u2082_502_, sizeof(void*)*1);
v___x_513_ = lean_uint64_dec_eq(v_hash_507_, v_hash_512_);
if (v___x_513_ == 0)
{
return v___x_513_;
}
else
{
uint8_t v___x_514_; 
v___x_514_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_504_, v_builder_509_);
if (v___x_514_ == 0)
{
return v___x_514_;
}
else
{
uint8_t v___x_515_; 
v___x_515_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_505_, v_phase_510_);
if (v___x_515_ == 0)
{
return v___x_515_;
}
else
{
uint8_t v___x_516_; 
v___x_516_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_506_, v_scope_511_);
if (v___x_516_ == 0)
{
return v___x_516_;
}
else
{
uint8_t v___x_517_; 
v___x_517_ = lean_name_eq(v_name_503_, v_name_508_);
return v___x_517_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instBEq___lam__0___boxed(lean_object* v_n_u2081_518_, lean_object* v_n_u2082_519_){
_start:
{
uint8_t v_res_520_; lean_object* v_r_521_; 
v_res_520_ = lp_aesop_Aesop_RuleName_instBEq___lam__0(v_n_u2081_518_, v_n_u2082_519_);
lean_dec_ref(v_n_u2082_519_);
lean_dec_ref(v_n_u2081_518_);
v_r_521_ = lean_box(v_res_520_);
return v_r_521_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_compare(lean_object* v_a_524_, lean_object* v_b_525_){
_start:
{
lean_object* v_name_526_; uint8_t v_builder_527_; uint8_t v_phase_528_; uint8_t v_scope_529_; lean_object* v_name_530_; uint8_t v_builder_531_; uint8_t v_phase_532_; uint8_t v_scope_533_; lean_object* v___x_534_; lean_object* v___x_535_; uint8_t v___x_536_; 
v_name_526_ = lean_ctor_get(v_a_524_, 0);
v_builder_527_ = lean_ctor_get_uint8(v_a_524_, sizeof(void*)*1 + 8);
v_phase_528_ = lean_ctor_get_uint8(v_a_524_, sizeof(void*)*1 + 9);
v_scope_529_ = lean_ctor_get_uint8(v_a_524_, sizeof(void*)*1 + 10);
v_name_530_ = lean_ctor_get(v_b_525_, 0);
v_builder_531_ = lean_ctor_get_uint8(v_b_525_, sizeof(void*)*1 + 8);
v_phase_532_ = lean_ctor_get_uint8(v_b_525_, sizeof(void*)*1 + 9);
v_scope_533_ = lean_ctor_get_uint8(v_b_525_, sizeof(void*)*1 + 10);
v___x_534_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_builder_527_);
v___x_535_ = lp_aesop_Aesop_BuilderName_ctorIdx(v_builder_531_);
v___x_536_ = lean_nat_dec_lt(v___x_534_, v___x_535_);
if (v___x_536_ == 0)
{
uint8_t v___x_537_; 
v___x_537_ = lean_nat_dec_eq(v___x_534_, v___x_535_);
lean_dec(v___x_535_);
lean_dec(v___x_534_);
if (v___x_537_ == 0)
{
uint8_t v___x_538_; 
v___x_538_ = 2;
return v___x_538_;
}
else
{
lean_object* v___x_539_; lean_object* v___x_540_; uint8_t v___x_541_; 
v___x_539_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_phase_528_);
v___x_540_ = lp_aesop_Aesop_PhaseName_ctorIdx(v_phase_532_);
v___x_541_ = lean_nat_dec_lt(v___x_539_, v___x_540_);
if (v___x_541_ == 0)
{
uint8_t v___x_542_; 
v___x_542_ = lean_nat_dec_eq(v___x_539_, v___x_540_);
lean_dec(v___x_540_);
lean_dec(v___x_539_);
if (v___x_542_ == 0)
{
uint8_t v___x_543_; 
v___x_543_ = 2;
return v___x_543_;
}
else
{
lean_object* v___x_544_; lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_544_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_scope_529_);
v___x_545_ = lp_aesop_Aesop_ScopeName_ctorIdx(v_scope_533_);
v___x_546_ = lean_nat_dec_lt(v___x_544_, v___x_545_);
if (v___x_546_ == 0)
{
uint8_t v___x_547_; 
v___x_547_ = lean_nat_dec_eq(v___x_544_, v___x_545_);
lean_dec(v___x_545_);
lean_dec(v___x_544_);
if (v___x_547_ == 0)
{
uint8_t v___x_548_; 
v___x_548_ = 2;
return v___x_548_;
}
else
{
uint8_t v___x_549_; 
v___x_549_ = l_Lean_Name_cmp(v_name_526_, v_name_530_);
return v___x_549_;
}
}
else
{
uint8_t v___x_550_; 
lean_dec(v___x_545_);
lean_dec(v___x_544_);
v___x_550_ = 0;
return v___x_550_;
}
}
}
else
{
uint8_t v___x_551_; 
lean_dec(v___x_540_);
lean_dec(v___x_539_);
v___x_551_ = 0;
return v___x_551_;
}
}
}
else
{
uint8_t v___x_552_; 
lean_dec(v___x_535_);
lean_dec(v___x_534_);
v___x_552_ = 0;
return v___x_552_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_compare___boxed(lean_object* v_a_553_, lean_object* v_b_554_){
_start:
{
uint8_t v_res_555_; lean_object* v_r_556_; 
v_res_555_ = lp_aesop_Aesop_RuleName_compare(v_a_553_, v_b_554_);
lean_dec_ref(v_b_554_);
lean_dec_ref(v_a_553_);
v_r_556_ = lean_box(v_res_555_);
return v_r_556_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleName_quickCompare(lean_object* v_n_u2081_557_, lean_object* v_n_u2082_558_){
_start:
{
uint64_t v_hash_559_; uint64_t v_hash_560_; uint8_t v___x_561_; 
v_hash_559_ = lean_ctor_get_uint64(v_n_u2081_557_, sizeof(void*)*1);
v_hash_560_ = lean_ctor_get_uint64(v_n_u2082_558_, sizeof(void*)*1);
v___x_561_ = lean_uint64_dec_lt(v_hash_559_, v_hash_560_);
if (v___x_561_ == 0)
{
uint8_t v___x_562_; 
v___x_562_ = lean_uint64_dec_eq(v_hash_559_, v_hash_560_);
if (v___x_562_ == 0)
{
uint8_t v___x_563_; 
v___x_563_ = 2;
return v___x_563_;
}
else
{
uint8_t v___x_564_; 
v___x_564_ = lp_aesop_Aesop_RuleName_compare(v_n_u2081_557_, v_n_u2082_558_);
return v___x_564_;
}
}
else
{
uint8_t v___x_565_; 
v___x_565_ = 0;
return v___x_565_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_quickCompare___boxed(lean_object* v_n_u2081_566_, lean_object* v_n_u2082_567_){
_start:
{
uint8_t v_res_568_; lean_object* v_r_569_; 
v_res_568_ = lp_aesop_Aesop_RuleName_quickCompare(v_n_u2081_566_, v_n_u2082_567_);
lean_dec_ref(v_n_u2082_567_);
lean_dec_ref(v_n_u2081_566_);
v_r_569_ = lean_box(v_res_568_);
return v_r_569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToString___lam__0(lean_object* v_n_573_){
_start:
{
lean_object* v_name_574_; uint8_t v_builder_575_; uint8_t v_phase_576_; uint8_t v_scope_577_; lean_object* v___y_579_; lean_object* v___y_580_; lean_object* v___y_581_; lean_object* v___y_588_; lean_object* v___y_589_; lean_object* v___y_590_; lean_object* v___y_596_; 
v_name_574_ = lean_ctor_get(v_n_573_, 0);
lean_inc(v_name_574_);
v_builder_575_ = lean_ctor_get_uint8(v_n_573_, sizeof(void*)*1 + 8);
v_phase_576_ = lean_ctor_get_uint8(v_n_573_, sizeof(void*)*1 + 9);
v_scope_577_ = lean_ctor_get_uint8(v_n_573_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_573_);
switch(v_phase_576_)
{
case 0:
{
lean_object* v___x_607_; 
v___x_607_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_596_ = v___x_607_;
goto v___jp_595_;
}
case 1:
{
lean_object* v___x_608_; 
v___x_608_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_596_ = v___x_608_;
goto v___jp_595_;
}
default: 
{
lean_object* v___x_609_; 
v___x_609_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_596_ = v___x_609_;
goto v___jp_595_;
}
}
v___jp_578_:
{
lean_object* v___x_582_; lean_object* v___x_583_; uint8_t v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_582_ = lean_string_append(v___y_580_, v___y_581_);
v___x_583_ = lean_string_append(v___x_582_, v___y_579_);
v___x_584_ = 1;
v___x_585_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_574_, v___x_584_);
v___x_586_ = lean_string_append(v___x_583_, v___x_585_);
lean_dec_ref(v___x_585_);
return v___x_586_;
}
v___jp_587_:
{
lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_591_ = lean_string_append(v___y_589_, v___y_590_);
v___x_592_ = lean_string_append(v___x_591_, v___y_588_);
if (v_scope_577_ == 0)
{
lean_object* v___x_593_; 
v___x_593_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_579_ = v___y_588_;
v___y_580_ = v___x_592_;
v___y_581_ = v___x_593_;
goto v___jp_578_;
}
else
{
lean_object* v___x_594_; 
v___x_594_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_579_ = v___y_588_;
v___y_580_ = v___x_592_;
v___y_581_ = v___x_594_;
goto v___jp_578_;
}
}
v___jp_595_:
{
lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_597_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_596_);
v___x_598_ = lean_string_append(v___y_596_, v___x_597_);
switch(v_builder_575_)
{
case 0:
{
lean_object* v___x_599_; 
v___x_599_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_599_;
goto v___jp_587_;
}
case 1:
{
lean_object* v___x_600_; 
v___x_600_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_600_;
goto v___jp_587_;
}
case 2:
{
lean_object* v___x_601_; 
v___x_601_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_601_;
goto v___jp_587_;
}
case 3:
{
lean_object* v___x_602_; 
v___x_602_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_602_;
goto v___jp_587_;
}
case 4:
{
lean_object* v___x_603_; 
v___x_603_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_603_;
goto v___jp_587_;
}
case 5:
{
lean_object* v___x_604_; 
v___x_604_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_604_;
goto v___jp_587_;
}
case 6:
{
lean_object* v___x_605_; 
v___x_605_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_605_;
goto v___jp_587_;
}
default: 
{
lean_object* v___x_606_; 
v___x_606_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_588_ = v___x_597_;
v___y_589_ = v___x_598_;
v___y_590_ = v___x_606_;
goto v___jp_587_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson_spec__0(lean_object* v_a_612_, lean_object* v_a_613_){
_start:
{
if (lean_obj_tag(v_a_612_) == 0)
{
lean_object* v___x_614_; 
v___x_614_ = lean_array_to_list(v_a_613_);
return v___x_614_;
}
else
{
lean_object* v_head_615_; lean_object* v_tail_616_; lean_object* v___x_617_; 
v_head_615_ = lean_ctor_get(v_a_612_, 0);
lean_inc(v_head_615_);
v_tail_616_ = lean_ctor_get(v_a_612_, 1);
lean_inc(v_tail_616_);
lean_dec_ref_known(v_a_612_, 2);
v___x_617_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_613_, v_head_615_);
v_a_612_ = v_tail_616_;
v_a_613_ = v___x_617_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(lean_object* v_x_626_){
_start:
{
lean_object* v_name_627_; uint8_t v_builder_628_; uint8_t v_phase_629_; uint8_t v_scope_630_; lean_object* v_rendered_631_; lean_object* v___x_632_; uint8_t v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v_name_627_ = lean_ctor_get(v_x_626_, 0);
lean_inc(v_name_627_);
v_builder_628_ = lean_ctor_get_uint8(v_x_626_, sizeof(void*)*2);
v_phase_629_ = lean_ctor_get_uint8(v_x_626_, sizeof(void*)*2 + 1);
v_scope_630_ = lean_ctor_get_uint8(v_x_626_, sizeof(void*)*2 + 2);
v_rendered_631_ = lean_ctor_get(v_x_626_, 1);
lean_inc_ref(v_rendered_631_);
lean_dec_ref(v_x_626_);
v___x_632_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__0));
v___x_633_ = 1;
v___x_634_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_627_, v___x_633_);
v___x_635_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_635_, 0, v___x_634_);
v___x_636_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_632_);
lean_ctor_set(v___x_636_, 1, v___x_635_);
v___x_637_ = lean_box(0);
v___x_638_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_636_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__1));
v___x_640_ = lp_aesop_Aesop_instToJsonBuilderName_toJson(v_builder_628_);
v___x_641_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_641_, 0, v___x_639_);
lean_ctor_set(v___x_641_, 1, v___x_640_);
v___x_642_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_642_, 0, v___x_641_);
lean_ctor_set(v___x_642_, 1, v___x_637_);
v___x_643_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__2));
v___x_644_ = lp_aesop_Aesop_instToJsonPhaseName_toJson(v_phase_629_);
v___x_645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_645_, 0, v___x_643_);
lean_ctor_set(v___x_645_, 1, v___x_644_);
v___x_646_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_646_, 0, v___x_645_);
lean_ctor_set(v___x_646_, 1, v___x_637_);
v___x_647_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__3));
v___x_648_ = lp_aesop_Aesop_instToJsonScopeName_toJson(v_scope_630_);
v___x_649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_649_, 0, v___x_647_);
lean_ctor_set(v___x_649_, 1, v___x_648_);
v___x_650_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_650_, 0, v___x_649_);
lean_ctor_set(v___x_650_, 1, v___x_637_);
v___x_651_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__4));
v___x_652_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_652_, 0, v_rendered_631_);
v___x_653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_653_, 0, v___x_651_);
lean_ctor_set(v___x_653_, 1, v___x_652_);
v___x_654_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
lean_ctor_set(v___x_654_, 1, v___x_637_);
v___x_655_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_655_, 0, v___x_654_);
lean_ctor_set(v___x_655_, 1, v___x_637_);
v___x_656_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_656_, 0, v___x_650_);
lean_ctor_set(v___x_656_, 1, v___x_655_);
v___x_657_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_646_);
lean_ctor_set(v___x_657_, 1, v___x_656_);
v___x_658_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_658_, 0, v___x_642_);
lean_ctor_set(v___x_658_, 1, v___x_657_);
v___x_659_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_659_, 0, v___x_638_);
lean_ctor_set(v___x_659_, 1, v___x_658_);
v___x_660_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson___closed__5));
v___x_661_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00__private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson_spec__0(v___x_659_, v___x_660_);
v___x_662_ = l_Lean_Json_mkObj(v___x_661_);
lean_dec(v___x_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToJson___private__1(lean_object* v_n_665_){
_start:
{
lean_object* v_name_666_; uint8_t v_builder_667_; uint8_t v_phase_668_; uint8_t v_scope_669_; lean_object* v___y_671_; lean_object* v___y_672_; lean_object* v___y_673_; lean_object* v___y_682_; lean_object* v___y_683_; lean_object* v___y_684_; lean_object* v___y_690_; 
v_name_666_ = lean_ctor_get(v_n_665_, 0);
lean_inc(v_name_666_);
v_builder_667_ = lean_ctor_get_uint8(v_n_665_, sizeof(void*)*1 + 8);
v_phase_668_ = lean_ctor_get_uint8(v_n_665_, sizeof(void*)*1 + 9);
v_scope_669_ = lean_ctor_get_uint8(v_n_665_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_665_);
switch(v_phase_668_)
{
case 0:
{
lean_object* v___x_701_; 
v___x_701_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_690_ = v___x_701_;
goto v___jp_689_;
}
case 1:
{
lean_object* v___x_702_; 
v___x_702_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_690_ = v___x_702_;
goto v___jp_689_;
}
default: 
{
lean_object* v___x_703_; 
v___x_703_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_690_ = v___x_703_;
goto v___jp_689_;
}
}
v___jp_670_:
{
lean_object* v___x_674_; lean_object* v___x_675_; uint8_t v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_674_ = lean_string_append(v___y_671_, v___y_673_);
v___x_675_ = lean_string_append(v___x_674_, v___y_672_);
v___x_676_ = 1;
lean_inc(v_name_666_);
v___x_677_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_666_, v___x_676_);
v___x_678_ = lean_string_append(v___x_675_, v___x_677_);
lean_dec_ref(v___x_677_);
v___x_679_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_679_, 0, v_name_666_);
lean_ctor_set(v___x_679_, 1, v___x_678_);
lean_ctor_set_uint8(v___x_679_, sizeof(void*)*2, v_builder_667_);
lean_ctor_set_uint8(v___x_679_, sizeof(void*)*2 + 1, v_phase_668_);
lean_ctor_set_uint8(v___x_679_, sizeof(void*)*2 + 2, v_scope_669_);
v___x_680_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_679_);
return v___x_680_;
}
v___jp_681_:
{
lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_685_ = lean_string_append(v___y_682_, v___y_684_);
v___x_686_ = lean_string_append(v___x_685_, v___y_683_);
if (v_scope_669_ == 0)
{
lean_object* v___x_687_; 
v___x_687_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_671_ = v___x_686_;
v___y_672_ = v___y_683_;
v___y_673_ = v___x_687_;
goto v___jp_670_;
}
else
{
lean_object* v___x_688_; 
v___x_688_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_671_ = v___x_686_;
v___y_672_ = v___y_683_;
v___y_673_ = v___x_688_;
goto v___jp_670_;
}
}
v___jp_689_:
{
lean_object* v___x_691_; lean_object* v___x_692_; 
v___x_691_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_690_);
v___x_692_ = lean_string_append(v___y_690_, v___x_691_);
switch(v_builder_667_)
{
case 0:
{
lean_object* v___x_693_; 
v___x_693_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_693_;
goto v___jp_681_;
}
case 1:
{
lean_object* v___x_694_; 
v___x_694_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_694_;
goto v___jp_681_;
}
case 2:
{
lean_object* v___x_695_; 
v___x_695_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_695_;
goto v___jp_681_;
}
case 3:
{
lean_object* v___x_696_; 
v___x_696_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_696_;
goto v___jp_681_;
}
case 4:
{
lean_object* v___x_697_; 
v___x_697_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_697_;
goto v___jp_681_;
}
case 5:
{
lean_object* v___x_698_; 
v___x_698_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_698_;
goto v___jp_681_;
}
case 6:
{
lean_object* v___x_699_; 
v___x_699_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_699_;
goto v___jp_681_;
}
default: 
{
lean_object* v___x_700_; 
v___x_700_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_682_ = v___x_692_;
v___y_683_ = v___x_691_;
v___y_684_ = v___x_700_;
goto v___jp_681_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleName_instToJson___lam__0(lean_object* v_n_704_){
_start:
{
lean_object* v_name_705_; uint8_t v_builder_706_; uint8_t v_phase_707_; uint8_t v_scope_708_; lean_object* v___y_710_; lean_object* v___y_711_; lean_object* v___y_712_; lean_object* v___y_721_; lean_object* v___y_722_; lean_object* v___y_723_; lean_object* v___y_729_; 
v_name_705_ = lean_ctor_get(v_n_704_, 0);
lean_inc(v_name_705_);
v_builder_706_ = lean_ctor_get_uint8(v_n_704_, sizeof(void*)*1 + 8);
v_phase_707_ = lean_ctor_get_uint8(v_n_704_, sizeof(void*)*1 + 9);
v_scope_708_ = lean_ctor_get_uint8(v_n_704_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_704_);
switch(v_phase_707_)
{
case 0:
{
lean_object* v___x_740_; 
v___x_740_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_729_ = v___x_740_;
goto v___jp_728_;
}
case 1:
{
lean_object* v___x_741_; 
v___x_741_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_729_ = v___x_741_;
goto v___jp_728_;
}
default: 
{
lean_object* v___x_742_; 
v___x_742_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_729_ = v___x_742_;
goto v___jp_728_;
}
}
v___jp_709_:
{
lean_object* v___x_713_; lean_object* v___x_714_; uint8_t v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_713_ = lean_string_append(v___y_710_, v___y_712_);
v___x_714_ = lean_string_append(v___x_713_, v___y_711_);
v___x_715_ = 1;
lean_inc(v_name_705_);
v___x_716_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_705_, v___x_715_);
v___x_717_ = lean_string_append(v___x_714_, v___x_716_);
lean_dec_ref(v___x_716_);
v___x_718_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_718_, 0, v_name_705_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
lean_ctor_set_uint8(v___x_718_, sizeof(void*)*2, v_builder_706_);
lean_ctor_set_uint8(v___x_718_, sizeof(void*)*2 + 1, v_phase_707_);
lean_ctor_set_uint8(v___x_718_, sizeof(void*)*2 + 2, v_scope_708_);
v___x_719_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_718_);
return v___x_719_;
}
v___jp_720_:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = lean_string_append(v___y_721_, v___y_723_);
v___x_725_ = lean_string_append(v___x_724_, v___y_722_);
if (v_scope_708_ == 0)
{
lean_object* v___x_726_; 
v___x_726_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_710_ = v___x_725_;
v___y_711_ = v___y_722_;
v___y_712_ = v___x_726_;
goto v___jp_709_;
}
else
{
lean_object* v___x_727_; 
v___x_727_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_710_ = v___x_725_;
v___y_711_ = v___y_722_;
v___y_712_ = v___x_727_;
goto v___jp_709_;
}
}
v___jp_728_:
{
lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_730_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_729_);
v___x_731_ = lean_string_append(v___y_729_, v___x_730_);
switch(v_builder_706_)
{
case 0:
{
lean_object* v___x_732_; 
v___x_732_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_732_;
goto v___jp_720_;
}
case 1:
{
lean_object* v___x_733_; 
v___x_733_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_733_;
goto v___jp_720_;
}
case 2:
{
lean_object* v___x_734_; 
v___x_734_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_734_;
goto v___jp_720_;
}
case 3:
{
lean_object* v___x_735_; 
v___x_735_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_735_;
goto v___jp_720_;
}
case 4:
{
lean_object* v___x_736_; 
v___x_736_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_736_;
goto v___jp_720_;
}
case 5:
{
lean_object* v___x_737_; 
v___x_737_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_737_;
goto v___jp_720_;
}
case 6:
{
lean_object* v___x_738_; 
v___x_738_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_738_;
goto v___jp_720_;
}
default: 
{
lean_object* v___x_739_; 
v___x_739_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_721_ = v___x_731_;
v___y_722_ = v___x_730_;
v___y_723_ = v___x_739_;
goto v___jp_720_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg(lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; lean_object* v_ngen_748_; lean_object* v_namePrefix_749_; lean_object* v_idx_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_779_; 
v___x_747_ = lean_st_ref_get(v___y_745_);
v_ngen_748_ = lean_ctor_get(v___x_747_, 2);
lean_inc_ref(v_ngen_748_);
lean_dec(v___x_747_);
v_namePrefix_749_ = lean_ctor_get(v_ngen_748_, 0);
v_idx_750_ = lean_ctor_get(v_ngen_748_, 1);
v_isSharedCheck_779_ = !lean_is_exclusive(v_ngen_748_);
if (v_isSharedCheck_779_ == 0)
{
v___x_752_ = v_ngen_748_;
v_isShared_753_ = v_isSharedCheck_779_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_idx_750_);
lean_inc(v_namePrefix_749_);
lean_dec(v_ngen_748_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_779_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_754_; lean_object* v_env_755_; lean_object* v_nextMacroScope_756_; lean_object* v_auxDeclNGen_757_; lean_object* v_traceState_758_; lean_object* v_cache_759_; lean_object* v_messages_760_; lean_object* v_infoState_761_; lean_object* v_snapshotTasks_762_; lean_object* v___x_764_; uint8_t v_isShared_765_; uint8_t v_isSharedCheck_777_; 
v___x_754_ = lean_st_ref_take(v___y_745_);
v_env_755_ = lean_ctor_get(v___x_754_, 0);
v_nextMacroScope_756_ = lean_ctor_get(v___x_754_, 1);
v_auxDeclNGen_757_ = lean_ctor_get(v___x_754_, 3);
v_traceState_758_ = lean_ctor_get(v___x_754_, 4);
v_cache_759_ = lean_ctor_get(v___x_754_, 5);
v_messages_760_ = lean_ctor_get(v___x_754_, 6);
v_infoState_761_ = lean_ctor_get(v___x_754_, 7);
v_snapshotTasks_762_ = lean_ctor_get(v___x_754_, 8);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_754_);
if (v_isSharedCheck_777_ == 0)
{
lean_object* v_unused_778_; 
v_unused_778_ = lean_ctor_get(v___x_754_, 2);
lean_dec(v_unused_778_);
v___x_764_ = v___x_754_;
v_isShared_765_ = v_isSharedCheck_777_;
goto v_resetjp_763_;
}
else
{
lean_inc(v_snapshotTasks_762_);
lean_inc(v_infoState_761_);
lean_inc(v_messages_760_);
lean_inc(v_cache_759_);
lean_inc(v_traceState_758_);
lean_inc(v_auxDeclNGen_757_);
lean_inc(v_nextMacroScope_756_);
lean_inc(v_env_755_);
lean_dec(v___x_754_);
v___x_764_ = lean_box(0);
v_isShared_765_ = v_isSharedCheck_777_;
goto v_resetjp_763_;
}
v_resetjp_763_:
{
lean_object* v_r_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_770_; 
lean_inc(v_idx_750_);
lean_inc(v_namePrefix_749_);
v_r_766_ = l_Lean_Name_num___override(v_namePrefix_749_, v_idx_750_);
v___x_767_ = lean_unsigned_to_nat(1u);
v___x_768_ = lean_nat_add(v_idx_750_, v___x_767_);
lean_dec(v_idx_750_);
if (v_isShared_753_ == 0)
{
lean_ctor_set(v___x_752_, 1, v___x_768_);
v___x_770_ = v___x_752_;
goto v_reusejp_769_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_namePrefix_749_);
lean_ctor_set(v_reuseFailAlloc_776_, 1, v___x_768_);
v___x_770_ = v_reuseFailAlloc_776_;
goto v_reusejp_769_;
}
v_reusejp_769_:
{
lean_object* v___x_772_; 
if (v_isShared_765_ == 0)
{
lean_ctor_set(v___x_764_, 2, v___x_770_);
v___x_772_ = v___x_764_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_env_755_);
lean_ctor_set(v_reuseFailAlloc_775_, 1, v_nextMacroScope_756_);
lean_ctor_set(v_reuseFailAlloc_775_, 2, v___x_770_);
lean_ctor_set(v_reuseFailAlloc_775_, 3, v_auxDeclNGen_757_);
lean_ctor_set(v_reuseFailAlloc_775_, 4, v_traceState_758_);
lean_ctor_set(v_reuseFailAlloc_775_, 5, v_cache_759_);
lean_ctor_set(v_reuseFailAlloc_775_, 6, v_messages_760_);
lean_ctor_set(v_reuseFailAlloc_775_, 7, v_infoState_761_);
lean_ctor_set(v_reuseFailAlloc_775_, 8, v_snapshotTasks_762_);
v___x_772_ = v_reuseFailAlloc_775_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_773_ = lean_st_ref_set(v___y_745_, v___x_772_);
v___x_774_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_774_, 0, v_r_766_);
return v___x_774_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg___boxed(lean_object* v___y_780_, lean_object* v___y_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg(v___y_780_);
lean_dec(v___y_780_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0(lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v___x_788_; 
v___x_788_ = lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg(v___y_786_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___boxed(lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0(v___y_789_, v___y_790_, v___y_791_, v___y_792_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRuleNameForExpr(lean_object* v_x_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_){
_start:
{
switch(lean_obj_tag(v_x_795_))
{
case 4:
{
lean_object* v_declName_801_; lean_object* v___x_802_; 
v_declName_801_ = lean_ctor_get(v_x_795_, 0);
lean_inc(v_declName_801_);
lean_dec_ref_known(v_x_795_, 2);
v___x_802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_802_, 0, v_declName_801_);
return v___x_802_;
}
case 1:
{
lean_object* v_fvarId_803_; lean_object* v___x_804_; 
v_fvarId_803_ = lean_ctor_get(v_x_795_, 0);
lean_inc(v_fvarId_803_);
lean_dec_ref_known(v_x_795_, 1);
v___x_804_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_803_, v_a_796_, v_a_798_, v_a_799_);
if (lean_obj_tag(v___x_804_) == 0)
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_813_; 
v_a_805_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_813_ == 0)
{
v___x_807_ = v___x_804_;
v_isShared_808_ = v_isSharedCheck_813_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_804_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_813_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_809_; lean_object* v___x_811_; 
v___x_809_ = l_Lean_LocalDecl_userName(v_a_805_);
lean_dec(v_a_805_);
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 0, v___x_809_);
v___x_811_ = v___x_807_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v___x_809_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
}
else
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_821_; 
v_a_814_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_821_ == 0)
{
v___x_816_ = v___x_804_;
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_804_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_819_; 
if (v_isShared_817_ == 0)
{
v___x_819_ = v___x_816_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_a_814_);
v___x_819_ = v_reuseFailAlloc_820_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
return v___x_819_;
}
}
}
}
default: 
{
lean_object* v___x_822_; 
lean_dec_ref(v_x_795_);
v___x_822_ = lp_aesop_Lean_mkFreshId___at___00Aesop_getRuleNameForExpr_spec__0___redArg(v_a_799_);
return v___x_822_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getRuleNameForExpr___boxed(lean_object* v_x_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_aesop_Aesop_getRuleNameForExpr(v_x_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_);
lean_dec(v_a_827_);
lean_dec_ref(v_a_826_);
lean_dec(v_a_825_);
lean_dec_ref(v_a_824_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorIdx(lean_object* v_x_830_){
_start:
{
switch(lean_obj_tag(v_x_830_))
{
case 0:
{
lean_object* v___x_831_; 
v___x_831_ = lean_unsigned_to_nat(0u);
return v___x_831_;
}
case 1:
{
lean_object* v___x_832_; 
v___x_832_ = lean_unsigned_to_nat(1u);
return v___x_832_;
}
default: 
{
lean_object* v___x_833_; 
v___x_833_ = lean_unsigned_to_nat(2u);
return v___x_833_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorIdx___boxed(lean_object* v_x_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_aesop_Aesop_DisplayRuleName_ctorIdx(v_x_834_);
lean_dec(v_x_834_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(lean_object* v_t_836_, lean_object* v_k_837_){
_start:
{
if (lean_obj_tag(v_t_836_) == 0)
{
lean_object* v_n_838_; lean_object* v___x_839_; 
v_n_838_ = lean_ctor_get(v_t_836_, 0);
lean_inc_ref(v_n_838_);
lean_dec_ref_known(v_t_836_, 1);
v___x_839_ = lean_apply_1(v_k_837_, v_n_838_);
return v___x_839_;
}
else
{
lean_dec(v_t_836_);
return v_k_837_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim(lean_object* v_motive_840_, lean_object* v_ctorIdx_841_, lean_object* v_t_842_, lean_object* v_h_843_, lean_object* v_k_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_842_, v_k_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ctorElim___boxed(lean_object* v_motive_846_, lean_object* v_ctorIdx_847_, lean_object* v_t_848_, lean_object* v_h_849_, lean_object* v_k_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_aesop_Aesop_DisplayRuleName_ctorElim(v_motive_846_, v_ctorIdx_847_, v_t_848_, v_h_849_, v_k_850_);
lean_dec(v_ctorIdx_847_);
return v_res_851_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ruleName_elim___redArg(lean_object* v_t_852_, lean_object* v_ruleName_853_){
_start:
{
lean_object* v___x_854_; 
v___x_854_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_852_, v_ruleName_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_ruleName_elim(lean_object* v_motive_855_, lean_object* v_t_856_, lean_object* v_h_857_, lean_object* v_ruleName_858_){
_start:
{
lean_object* v___x_859_; 
v___x_859_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_856_, v_ruleName_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normSimp_elim___redArg(lean_object* v_t_860_, lean_object* v_normSimp_861_){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_860_, v_normSimp_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normSimp_elim(lean_object* v_motive_863_, lean_object* v_t_864_, lean_object* v_h_865_, lean_object* v_normSimp_866_){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_864_, v_normSimp_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normUnfold_elim___redArg(lean_object* v_t_868_, lean_object* v_normUnfold_869_){
_start:
{
lean_object* v___x_870_; 
v___x_870_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_868_, v_normUnfold_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_normUnfold_elim(lean_object* v_motive_871_, lean_object* v_t_872_, lean_object* v_h_873_, lean_object* v_normUnfold_874_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = lp_aesop_Aesop_DisplayRuleName_ctorElim___redArg(v_t_872_, v_normUnfold_874_);
return v___x_875_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0(void){
_start:
{
lean_object* v___x_876_; lean_object* v___x_877_; 
v___x_876_ = lp_aesop_Aesop_instInhabitedRuleName_default;
v___x_877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_877_, 0, v___x_876_);
return v___x_877_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDisplayRuleName_default(void){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0, &lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedDisplayRuleName_default___closed__0);
return v___x_878_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedDisplayRuleName(void){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_aesop_Aesop_instInhabitedDisplayRuleName_default;
return v___x_879_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqDisplayRuleName_beq(lean_object* v_x_880_, lean_object* v_x_881_){
_start:
{
switch(lean_obj_tag(v_x_880_))
{
case 0:
{
if (lean_obj_tag(v_x_881_) == 0)
{
lean_object* v_n_882_; lean_object* v_n_883_; lean_object* v_name_884_; uint8_t v_builder_885_; uint8_t v_phase_886_; uint8_t v_scope_887_; uint64_t v_hash_888_; lean_object* v_name_889_; uint8_t v_builder_890_; uint8_t v_phase_891_; uint8_t v_scope_892_; uint64_t v_hash_893_; uint8_t v___y_895_; uint8_t v___x_899_; 
v_n_882_ = lean_ctor_get(v_x_880_, 0);
v_n_883_ = lean_ctor_get(v_x_881_, 0);
v_name_884_ = lean_ctor_get(v_n_882_, 0);
v_builder_885_ = lean_ctor_get_uint8(v_n_882_, sizeof(void*)*1 + 8);
v_phase_886_ = lean_ctor_get_uint8(v_n_882_, sizeof(void*)*1 + 9);
v_scope_887_ = lean_ctor_get_uint8(v_n_882_, sizeof(void*)*1 + 10);
v_hash_888_ = lean_ctor_get_uint64(v_n_882_, sizeof(void*)*1);
v_name_889_ = lean_ctor_get(v_n_883_, 0);
v_builder_890_ = lean_ctor_get_uint8(v_n_883_, sizeof(void*)*1 + 8);
v_phase_891_ = lean_ctor_get_uint8(v_n_883_, sizeof(void*)*1 + 9);
v_scope_892_ = lean_ctor_get_uint8(v_n_883_, sizeof(void*)*1 + 10);
v_hash_893_ = lean_ctor_get_uint64(v_n_883_, sizeof(void*)*1);
v___x_899_ = lean_uint64_dec_eq(v_hash_888_, v_hash_893_);
if (v___x_899_ == 0)
{
v___y_895_ = v___x_899_;
goto v___jp_894_;
}
else
{
uint8_t v___x_900_; 
v___x_900_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_885_, v_builder_890_);
v___y_895_ = v___x_900_;
goto v___jp_894_;
}
v___jp_894_:
{
if (v___y_895_ == 0)
{
return v___y_895_;
}
else
{
uint8_t v___x_896_; 
v___x_896_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_886_, v_phase_891_);
if (v___x_896_ == 0)
{
return v___x_896_;
}
else
{
uint8_t v___x_897_; 
v___x_897_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_887_, v_scope_892_);
if (v___x_897_ == 0)
{
return v___x_897_;
}
else
{
uint8_t v___x_898_; 
v___x_898_ = lean_name_eq(v_name_884_, v_name_889_);
return v___x_898_;
}
}
}
}
}
else
{
uint8_t v___x_901_; 
v___x_901_ = 0;
return v___x_901_;
}
}
case 1:
{
if (lean_obj_tag(v_x_881_) == 1)
{
uint8_t v___x_902_; 
v___x_902_ = 1;
return v___x_902_;
}
else
{
uint8_t v___x_903_; 
v___x_903_ = 0;
return v___x_903_;
}
}
default: 
{
if (lean_obj_tag(v_x_881_) == 2)
{
uint8_t v___x_904_; 
v___x_904_ = 1;
return v___x_904_;
}
else
{
uint8_t v___x_905_; 
v___x_905_ = 0;
return v___x_905_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqDisplayRuleName_beq___boxed(lean_object* v_x_906_, lean_object* v_x_907_){
_start:
{
uint8_t v_res_908_; lean_object* v_r_909_; 
v_res_908_ = lp_aesop_Aesop_instBEqDisplayRuleName_beq(v_x_906_, v_x_907_);
lean_dec(v_x_907_);
lean_dec(v_x_906_);
v_r_909_ = lean_box(v_res_908_);
return v_r_909_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdDisplayRuleName_ord(lean_object* v_x_912_, lean_object* v_x_913_){
_start:
{
switch(lean_obj_tag(v_x_912_))
{
case 0:
{
switch(lean_obj_tag(v_x_913_))
{
case 0:
{
lean_object* v_n_914_; lean_object* v_n_915_; uint8_t v___x_916_; 
v_n_914_ = lean_ctor_get(v_x_912_, 0);
v_n_915_ = lean_ctor_get(v_x_913_, 0);
v___x_916_ = lp_aesop_Aesop_RuleName_compare(v_n_914_, v_n_915_);
if (v___x_916_ == 1)
{
return v___x_916_;
}
else
{
return v___x_916_;
}
}
case 1:
{
uint8_t v___x_917_; 
v___x_917_ = 0;
return v___x_917_;
}
default: 
{
uint8_t v___x_918_; 
v___x_918_ = 0;
return v___x_918_;
}
}
}
case 1:
{
switch(lean_obj_tag(v_x_913_))
{
case 0:
{
uint8_t v___x_919_; 
v___x_919_ = 2;
return v___x_919_;
}
case 1:
{
uint8_t v___x_920_; 
v___x_920_ = 1;
return v___x_920_;
}
default: 
{
uint8_t v___x_921_; 
v___x_921_ = 0;
return v___x_921_;
}
}
}
default: 
{
if (lean_obj_tag(v_x_913_) == 2)
{
uint8_t v___x_922_; 
v___x_922_ = 1;
return v___x_922_;
}
else
{
uint8_t v___x_923_; 
v___x_923_ = 2;
return v___x_923_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdDisplayRuleName_ord___boxed(lean_object* v_x_924_, lean_object* v_x_925_){
_start:
{
uint8_t v_res_926_; lean_object* v_r_927_; 
v_res_926_ = lp_aesop_Aesop_instOrdDisplayRuleName_ord(v_x_924_, v_x_925_);
lean_dec(v_x_925_);
lean_dec(v_x_924_);
v_r_927_ = lean_box(v_res_926_);
return v_r_927_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableDisplayRuleName_hash(lean_object* v_x_930_){
_start:
{
switch(lean_obj_tag(v_x_930_))
{
case 0:
{
lean_object* v_n_931_; uint64_t v_hash_932_; uint64_t v___x_933_; uint64_t v___x_934_; 
v_n_931_ = lean_ctor_get(v_x_930_, 0);
v_hash_932_ = lean_ctor_get_uint64(v_n_931_, sizeof(void*)*1);
v___x_933_ = 0ULL;
v___x_934_ = lean_uint64_mix_hash(v___x_933_, v_hash_932_);
return v___x_934_;
}
case 1:
{
uint64_t v___x_935_; 
v___x_935_ = 1ULL;
return v___x_935_;
}
default: 
{
uint64_t v___x_936_; 
v___x_936_ = 2ULL;
return v___x_936_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableDisplayRuleName_hash___boxed(lean_object* v_x_937_){
_start:
{
uint64_t v_res_938_; lean_object* v_r_939_; 
v_res_938_ = lp_aesop_Aesop_instHashableDisplayRuleName_hash(v_x_937_);
lean_dec(v_x_937_);
v_r_939_ = lean_box_uint64(v_res_938_);
return v_r_939_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instCoeRuleName___lam__0(lean_object* v_n_942_){
_start:
{
lean_object* v___x_943_; 
v___x_943_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_943_, 0, v_n_942_);
return v___x_943_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToString___lam__0(lean_object* v_x_948_){
_start:
{
switch(lean_obj_tag(v_x_948_))
{
case 0:
{
lean_object* v_n_949_; lean_object* v_name_950_; uint8_t v_builder_951_; uint8_t v_phase_952_; uint8_t v_scope_953_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_957_; lean_object* v___y_964_; lean_object* v___y_965_; lean_object* v___y_966_; lean_object* v___y_972_; 
v_n_949_ = lean_ctor_get(v_x_948_, 0);
lean_inc_ref(v_n_949_);
lean_dec_ref_known(v_x_948_, 1);
v_name_950_ = lean_ctor_get(v_n_949_, 0);
lean_inc(v_name_950_);
v_builder_951_ = lean_ctor_get_uint8(v_n_949_, sizeof(void*)*1 + 8);
v_phase_952_ = lean_ctor_get_uint8(v_n_949_, sizeof(void*)*1 + 9);
v_scope_953_ = lean_ctor_get_uint8(v_n_949_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_949_);
switch(v_phase_952_)
{
case 0:
{
lean_object* v___x_983_; 
v___x_983_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_972_ = v___x_983_;
goto v___jp_971_;
}
case 1:
{
lean_object* v___x_984_; 
v___x_984_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_972_ = v___x_984_;
goto v___jp_971_;
}
default: 
{
lean_object* v___x_985_; 
v___x_985_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_972_ = v___x_985_;
goto v___jp_971_;
}
}
v___jp_954_:
{
lean_object* v___x_958_; lean_object* v___x_959_; uint8_t v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; 
v___x_958_ = lean_string_append(v___y_955_, v___y_957_);
v___x_959_ = lean_string_append(v___x_958_, v___y_956_);
v___x_960_ = 1;
v___x_961_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_950_, v___x_960_);
v___x_962_ = lean_string_append(v___x_959_, v___x_961_);
lean_dec_ref(v___x_961_);
return v___x_962_;
}
v___jp_963_:
{
lean_object* v___x_967_; lean_object* v___x_968_; 
v___x_967_ = lean_string_append(v___y_964_, v___y_966_);
v___x_968_ = lean_string_append(v___x_967_, v___y_965_);
if (v_scope_953_ == 0)
{
lean_object* v___x_969_; 
v___x_969_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_955_ = v___x_968_;
v___y_956_ = v___y_965_;
v___y_957_ = v___x_969_;
goto v___jp_954_;
}
else
{
lean_object* v___x_970_; 
v___x_970_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_955_ = v___x_968_;
v___y_956_ = v___y_965_;
v___y_957_ = v___x_970_;
goto v___jp_954_;
}
}
v___jp_971_:
{
lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_973_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_972_);
v___x_974_ = lean_string_append(v___y_972_, v___x_973_);
switch(v_builder_951_)
{
case 0:
{
lean_object* v___x_975_; 
v___x_975_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_975_;
goto v___jp_963_;
}
case 1:
{
lean_object* v___x_976_; 
v___x_976_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_976_;
goto v___jp_963_;
}
case 2:
{
lean_object* v___x_977_; 
v___x_977_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_977_;
goto v___jp_963_;
}
case 3:
{
lean_object* v___x_978_; 
v___x_978_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_978_;
goto v___jp_963_;
}
case 4:
{
lean_object* v___x_979_; 
v___x_979_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_979_;
goto v___jp_963_;
}
case 5:
{
lean_object* v___x_980_; 
v___x_980_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_980_;
goto v___jp_963_;
}
case 6:
{
lean_object* v___x_981_; 
v___x_981_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_981_;
goto v___jp_963_;
}
default: 
{
lean_object* v___x_982_; 
v___x_982_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_964_ = v___x_974_;
v___y_965_ = v___x_973_;
v___y_966_ = v___x_982_;
goto v___jp_963_;
}
}
}
}
case 1:
{
lean_object* v___x_986_; 
v___x_986_ = ((lean_object*)(lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__0));
return v___x_986_;
}
default: 
{
lean_object* v___x_987_; 
v___x_987_ = ((lean_object*)(lp_aesop_Aesop_DisplayRuleName_instToString___lam__0___closed__1));
return v___x_987_;
}
}
}
}
static uint64_t _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1(void){
_start:
{
uint8_t v___x_992_; uint64_t v___x_993_; 
v___x_992_ = 5;
v___x_993_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_992_);
return v___x_993_;
}
}
static uint64_t _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2(void){
_start:
{
uint64_t v___x_994_; uint64_t v___x_995_; uint64_t v___x_996_; 
v___x_994_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__3, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__3);
v___x_995_ = lean_uint64_once(&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1, &lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1_once, _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__1);
v___x_996_ = lean_uint64_mix_hash(v___x_995_, v___x_994_);
return v___x_996_;
}
}
static uint64_t _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4(void){
_start:
{
uint8_t v___x_999_; uint64_t v___x_1000_; 
v___x_999_ = 7;
v___x_1000_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_999_);
return v___x_1000_;
}
}
static uint64_t _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5(void){
_start:
{
uint64_t v___x_1001_; uint64_t v___x_1002_; uint64_t v___x_1003_; 
v___x_1001_ = lean_uint64_once(&lp_aesop_Aesop_instInhabitedRuleName_default___closed__3, &lp_aesop_Aesop_instInhabitedRuleName_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedRuleName_default___closed__3);
v___x_1002_ = lean_uint64_once(&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4, &lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4_once, _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__4);
v___x_1003_ = lean_uint64_mix_hash(v___x_1002_, v___x_1001_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(lean_object* v_x_1004_){
_start:
{
switch(lean_obj_tag(v_x_1004_))
{
case 0:
{
lean_object* v_n_1005_; 
v_n_1005_ = lean_ctor_get(v_x_1004_, 0);
lean_inc_ref(v_n_1005_);
return v_n_1005_;
}
case 1:
{
lean_object* v___x_1006_; uint8_t v___x_1007_; uint8_t v___x_1008_; uint8_t v___x_1009_; uint64_t v___y_1011_; 
v___x_1006_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__0));
v___x_1007_ = 5;
v___x_1008_ = 0;
v___x_1009_ = 0;
if (lean_obj_tag(v___x_1006_) == 0)
{
uint64_t v___x_1015_; 
v___x_1015_ = 1723ULL;
v___y_1011_ = v___x_1015_;
goto v___jp_1010_;
}
else
{
uint64_t v_hash_1016_; 
v_hash_1016_ = lean_ctor_get_uint64(v___x_1006_, sizeof(void*)*2);
v___y_1011_ = v_hash_1016_;
goto v___jp_1010_;
}
v___jp_1010_:
{
uint64_t v___x_1012_; uint64_t v___x_1013_; lean_object* v___x_1014_; 
v___x_1012_ = lean_uint64_once(&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2, &lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2_once, _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__2);
v___x_1013_ = lean_uint64_mix_hash(v___y_1011_, v___x_1012_);
v___x_1014_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_1014_, 0, v___x_1006_);
lean_ctor_set_uint8(v___x_1014_, sizeof(void*)*1 + 8, v___x_1007_);
lean_ctor_set_uint8(v___x_1014_, sizeof(void*)*1 + 9, v___x_1008_);
lean_ctor_set_uint8(v___x_1014_, sizeof(void*)*1 + 10, v___x_1009_);
lean_ctor_set_uint64(v___x_1014_, sizeof(void*)*1, v___x_1013_);
return v___x_1014_;
}
}
default: 
{
lean_object* v___x_1017_; uint8_t v___x_1018_; uint8_t v___x_1019_; uint8_t v___x_1020_; uint64_t v___y_1022_; 
v___x_1017_ = ((lean_object*)(lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__3));
v___x_1018_ = 7;
v___x_1019_ = 0;
v___x_1020_ = 0;
if (lean_obj_tag(v___x_1017_) == 0)
{
uint64_t v___x_1026_; 
v___x_1026_ = 1723ULL;
v___y_1022_ = v___x_1026_;
goto v___jp_1021_;
}
else
{
uint64_t v_hash_1027_; 
v_hash_1027_ = lean_ctor_get_uint64(v___x_1017_, sizeof(void*)*2);
v___y_1022_ = v_hash_1027_;
goto v___jp_1021_;
}
v___jp_1021_:
{
uint64_t v___x_1023_; uint64_t v___x_1024_; lean_object* v___x_1025_; 
v___x_1023_ = lean_uint64_once(&lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5, &lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5_once, _init_lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___closed__5);
v___x_1024_ = lean_uint64_mix_hash(v___y_1022_, v___x_1023_);
v___x_1025_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_1025_, 0, v___x_1017_);
lean_ctor_set_uint8(v___x_1025_, sizeof(void*)*1 + 8, v___x_1018_);
lean_ctor_set_uint8(v___x_1025_, sizeof(void*)*1 + 9, v___x_1019_);
lean_ctor_set_uint8(v___x_1025_, sizeof(void*)*1 + 10, v___x_1020_);
lean_ctor_set_uint64(v___x_1025_, sizeof(void*)*1, v___x_1024_);
return v___x_1025_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName___boxed(lean_object* v_x_1028_){
_start:
{
lean_object* v_res_1029_; 
v_res_1029_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(v_x_1028_);
lean_dec(v_x_1028_);
return v_res_1029_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___private__1(lean_object* v_n_1030_){
_start:
{
lean_object* v___x_1031_; lean_object* v_name_1032_; uint8_t v_builder_1033_; uint8_t v_phase_1034_; uint8_t v_scope_1035_; lean_object* v___y_1037_; lean_object* v___y_1038_; lean_object* v___y_1039_; lean_object* v___y_1048_; lean_object* v___y_1049_; lean_object* v___y_1050_; lean_object* v___y_1056_; 
v___x_1031_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(v_n_1030_);
v_name_1032_ = lean_ctor_get(v___x_1031_, 0);
lean_inc(v_name_1032_);
v_builder_1033_ = lean_ctor_get_uint8(v___x_1031_, sizeof(void*)*1 + 8);
v_phase_1034_ = lean_ctor_get_uint8(v___x_1031_, sizeof(void*)*1 + 9);
v_scope_1035_ = lean_ctor_get_uint8(v___x_1031_, sizeof(void*)*1 + 10);
lean_dec_ref(v___x_1031_);
switch(v_phase_1034_)
{
case 0:
{
lean_object* v___x_1067_; 
v___x_1067_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_1056_ = v___x_1067_;
goto v___jp_1055_;
}
case 1:
{
lean_object* v___x_1068_; 
v___x_1068_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_1056_ = v___x_1068_;
goto v___jp_1055_;
}
default: 
{
lean_object* v___x_1069_; 
v___x_1069_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_1056_ = v___x_1069_;
goto v___jp_1055_;
}
}
v___jp_1036_:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; uint8_t v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1040_ = lean_string_append(v___y_1038_, v___y_1039_);
v___x_1041_ = lean_string_append(v___x_1040_, v___y_1037_);
v___x_1042_ = 1;
lean_inc(v_name_1032_);
v___x_1043_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1032_, v___x_1042_);
v___x_1044_ = lean_string_append(v___x_1041_, v___x_1043_);
lean_dec_ref(v___x_1043_);
v___x_1045_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_1045_, 0, v_name_1032_);
lean_ctor_set(v___x_1045_, 1, v___x_1044_);
lean_ctor_set_uint8(v___x_1045_, sizeof(void*)*2, v_builder_1033_);
lean_ctor_set_uint8(v___x_1045_, sizeof(void*)*2 + 1, v_phase_1034_);
lean_ctor_set_uint8(v___x_1045_, sizeof(void*)*2 + 2, v_scope_1035_);
v___x_1046_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_1045_);
return v___x_1046_;
}
v___jp_1047_:
{
lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1051_ = lean_string_append(v___y_1049_, v___y_1050_);
v___x_1052_ = lean_string_append(v___x_1051_, v___y_1048_);
if (v_scope_1035_ == 0)
{
lean_object* v___x_1053_; 
v___x_1053_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_1037_ = v___y_1048_;
v___y_1038_ = v___x_1052_;
v___y_1039_ = v___x_1053_;
goto v___jp_1036_;
}
else
{
lean_object* v___x_1054_; 
v___x_1054_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_1037_ = v___y_1048_;
v___y_1038_ = v___x_1052_;
v___y_1039_ = v___x_1054_;
goto v___jp_1036_;
}
}
v___jp_1055_:
{
lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1057_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_1056_);
v___x_1058_ = lean_string_append(v___y_1056_, v___x_1057_);
switch(v_builder_1033_)
{
case 0:
{
lean_object* v___x_1059_; 
v___x_1059_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1059_;
goto v___jp_1047_;
}
case 1:
{
lean_object* v___x_1060_; 
v___x_1060_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1060_;
goto v___jp_1047_;
}
case 2:
{
lean_object* v___x_1061_; 
v___x_1061_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1061_;
goto v___jp_1047_;
}
case 3:
{
lean_object* v___x_1062_; 
v___x_1062_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1062_;
goto v___jp_1047_;
}
case 4:
{
lean_object* v___x_1063_; 
v___x_1063_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1063_;
goto v___jp_1047_;
}
case 5:
{
lean_object* v___x_1064_; 
v___x_1064_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1064_;
goto v___jp_1047_;
}
case 6:
{
lean_object* v___x_1065_; 
v___x_1065_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1065_;
goto v___jp_1047_;
}
default: 
{
lean_object* v___x_1066_; 
v___x_1066_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_1048_ = v___x_1057_;
v___y_1049_ = v___x_1058_;
v___y_1050_ = v___x_1066_;
goto v___jp_1047_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___private__1___boxed(lean_object* v_n_1070_){
_start:
{
lean_object* v_res_1071_; 
v_res_1071_ = lp_aesop_Aesop_DisplayRuleName_instToJson___private__1(v_n_1070_);
lean_dec(v_n_1070_);
return v_res_1071_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0(lean_object* v_n_1072_){
_start:
{
lean_object* v___x_1073_; lean_object* v_name_1074_; uint8_t v_builder_1075_; uint8_t v_phase_1076_; uint8_t v_scope_1077_; lean_object* v___y_1079_; lean_object* v___y_1080_; lean_object* v___y_1081_; lean_object* v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; lean_object* v___y_1098_; 
v___x_1073_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(v_n_1072_);
v_name_1074_ = lean_ctor_get(v___x_1073_, 0);
lean_inc(v_name_1074_);
v_builder_1075_ = lean_ctor_get_uint8(v___x_1073_, sizeof(void*)*1 + 8);
v_phase_1076_ = lean_ctor_get_uint8(v___x_1073_, sizeof(void*)*1 + 9);
v_scope_1077_ = lean_ctor_get_uint8(v___x_1073_, sizeof(void*)*1 + 10);
lean_dec_ref(v___x_1073_);
switch(v_phase_1076_)
{
case 0:
{
lean_object* v___x_1109_; 
v___x_1109_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__0));
v___y_1098_ = v___x_1109_;
goto v___jp_1097_;
}
case 1:
{
lean_object* v___x_1110_; 
v___x_1110_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__2));
v___y_1098_ = v___x_1110_;
goto v___jp_1097_;
}
default: 
{
lean_object* v___x_1111_; 
v___x_1111_ = ((lean_object*)(lp_aesop_Aesop_instToJsonPhaseName_toJson___closed__4));
v___y_1098_ = v___x_1111_;
goto v___jp_1097_;
}
}
v___jp_1078_:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; uint8_t v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; 
v___x_1082_ = lean_string_append(v___y_1080_, v___y_1081_);
v___x_1083_ = lean_string_append(v___x_1082_, v___y_1079_);
v___x_1084_ = 1;
lean_inc(v_name_1074_);
v___x_1085_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1074_, v___x_1084_);
v___x_1086_ = lean_string_append(v___x_1083_, v___x_1085_);
lean_dec_ref(v___x_1085_);
v___x_1087_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_1087_, 0, v_name_1074_);
lean_ctor_set(v___x_1087_, 1, v___x_1086_);
lean_ctor_set_uint8(v___x_1087_, sizeof(void*)*2, v_builder_1075_);
lean_ctor_set_uint8(v___x_1087_, sizeof(void*)*2 + 1, v_phase_1076_);
lean_ctor_set_uint8(v___x_1087_, sizeof(void*)*2 + 2, v_scope_1077_);
v___x_1088_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_1087_);
return v___x_1088_;
}
v___jp_1089_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1093_ = lean_string_append(v___y_1090_, v___y_1092_);
v___x_1094_ = lean_string_append(v___x_1093_, v___y_1091_);
if (v_scope_1077_ == 0)
{
lean_object* v___x_1095_; 
v___x_1095_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__0));
v___y_1079_ = v___y_1091_;
v___y_1080_ = v___x_1094_;
v___y_1081_ = v___x_1095_;
goto v___jp_1078_;
}
else
{
lean_object* v___x_1096_; 
v___x_1096_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScopeName_toJson___closed__2));
v___y_1079_ = v___y_1091_;
v___y_1080_ = v___x_1094_;
v___y_1081_ = v___x_1096_;
goto v___jp_1078_;
}
}
v___jp_1097_:
{
lean_object* v___x_1099_; lean_object* v___x_1100_; 
v___x_1099_ = ((lean_object*)(lp_aesop_Aesop_RuleName_instToString___lam__0___closed__0));
lean_inc_ref(v___y_1098_);
v___x_1100_ = lean_string_append(v___y_1098_, v___x_1099_);
switch(v_builder_1075_)
{
case 0:
{
lean_object* v___x_1101_; 
v___x_1101_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__0));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1101_;
goto v___jp_1089_;
}
case 1:
{
lean_object* v___x_1102_; 
v___x_1102_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__2));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1102_;
goto v___jp_1089_;
}
case 2:
{
lean_object* v___x_1103_; 
v___x_1103_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__4));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1103_;
goto v___jp_1089_;
}
case 3:
{
lean_object* v___x_1104_; 
v___x_1104_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__6));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1104_;
goto v___jp_1089_;
}
case 4:
{
lean_object* v___x_1105_; 
v___x_1105_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__8));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1105_;
goto v___jp_1089_;
}
case 5:
{
lean_object* v___x_1106_; 
v___x_1106_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__10));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1106_;
goto v___jp_1089_;
}
case 6:
{
lean_object* v___x_1107_; 
v___x_1107_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__12));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1107_;
goto v___jp_1089_;
}
default: 
{
lean_object* v___x_1108_; 
v___x_1108_ = ((lean_object*)(lp_aesop_Aesop_instToJsonBuilderName_toJson___closed__14));
v___y_1090_ = v___x_1100_;
v___y_1091_ = v___x_1099_;
v___y_1092_ = v___x_1108_;
goto v___jp_1089_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0___boxed(lean_object* v_n_1112_){
_start:
{
lean_object* v_res_1113_; 
v_res_1113_ = lp_aesop_Aesop_DisplayRuleName_instToJson___lam__0(v_n_1112_);
lean_dec(v_n_1112_);
return v_res_1113_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Rule_Name(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPhaseName_default = _init_lp_aesop_Aesop_instInhabitedPhaseName_default();
lp_aesop_Aesop_instInhabitedPhaseName = _init_lp_aesop_Aesop_instInhabitedPhaseName();
lp_aesop_Aesop_instInhabitedScopeName_default = _init_lp_aesop_Aesop_instInhabitedScopeName_default();
lp_aesop_Aesop_instInhabitedScopeName = _init_lp_aesop_Aesop_instInhabitedScopeName();
lp_aesop_Aesop_instInhabitedBuilderName_default = _init_lp_aesop_Aesop_instInhabitedBuilderName_default();
lp_aesop_Aesop_instInhabitedBuilderName = _init_lp_aesop_Aesop_instInhabitedBuilderName();
lp_aesop_Aesop_instInhabitedRuleName_default = _init_lp_aesop_Aesop_instInhabitedRuleName_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleName_default);
lp_aesop_Aesop_instInhabitedRuleName = _init_lp_aesop_Aesop_instInhabitedRuleName();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleName);
lp_aesop_Aesop_instInhabitedDisplayRuleName_default = _init_lp_aesop_Aesop_instInhabitedDisplayRuleName_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedDisplayRuleName_default);
lp_aesop_Aesop_instInhabitedDisplayRuleName = _init_lp_aesop_Aesop_instInhabitedDisplayRuleName();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedDisplayRuleName);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Rule_Name(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Rule_Name(builtin);
}
#ifdef __cplusplus
}
#endif
