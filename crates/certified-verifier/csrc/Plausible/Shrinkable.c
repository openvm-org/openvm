// Lean compiler output
// Module: Plausible.Shrinkable
// Imports: public import Init public meta import Init
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
lean_object* l_USize_ofNat___boxed(lean_object*);
lean_object* lean_usize_to_nat(size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_ofNat___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_List_filterMapTR_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_modifyTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapIdx_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_UInt64_ofNat___boxed(lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_UInt32_ofNat___boxed(lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* l_UInt16_ofNat___boxed(lean_object*);
lean_object* lean_uint16_to_nat(uint16_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_UInt8_ofNat___boxed(lean_object*);
lean_object* lean_uint8_to_nat(uint8_t);
lean_object* l_String_ofList___boxed(lean_object*);
lean_object* lean_string_data(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_BitVec_ofNat___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_instShrinkableSum___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instShrinkableSum___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_instShrinkableSum___redArg___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_instShrinkableSum___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instShrinkableSum___redArg___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_instShrinkableSum___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Unit_shrinkable___lam__0(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Unit_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Unit_shrinkable___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Unit_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_Unit_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Unit_shrinkable = (const lean_object*)&lp_plausible_Plausible_Unit_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Nat_shrink(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Nat_shrink___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Nat_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Nat_shrink___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Nat_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_Nat_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Nat_shrinkable = (const lean_object*)&lp_plausible_Plausible_Nat_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable(lean_object*);
static const lean_closure_object lp_plausible_Plausible_UInt8_shrinkable___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt8_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt8_shrinkable___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt8_shrinkable___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt8_shrinkable___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt8_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_UInt8_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_UInt8_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt8_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt8_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_UInt8_shrinkable = (const lean_object*)&lp_plausible_Plausible_UInt8_shrinkable___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_UInt16_shrinkable___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt16_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt16_shrinkable___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt16_shrinkable___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt16_shrinkable___lam__0(uint16_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt16_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_UInt16_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_UInt16_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt16_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt16_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_UInt16_shrinkable = (const lean_object*)&lp_plausible_Plausible_UInt16_shrinkable___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_UInt32_shrinkable___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt32_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt32_shrinkable___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt32_shrinkable___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt32_shrinkable___lam__0(uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt32_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_UInt32_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_UInt32_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt32_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt32_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_UInt32_shrinkable = (const lean_object*)&lp_plausible_Plausible_UInt32_shrinkable___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_UInt64_shrinkable___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt64_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt64_shrinkable___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt64_shrinkable___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt64_shrinkable___lam__0(uint64_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt64_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_UInt64_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_UInt64_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_UInt64_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_UInt64_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_UInt64_shrinkable = (const lean_object*)&lp_plausible_Plausible_UInt64_shrinkable___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_USize_shrinkable___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_USize_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_USize_shrinkable___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_USize_shrinkable___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_USize_shrinkable___lam__0(size_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_USize_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_USize_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_USize_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_USize_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_USize_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_USize_shrinkable = (const lean_object*)&lp_plausible_Plausible_USize_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__0(lean_object*);
static const lean_array_object lp_plausible_Plausible_Int_shrinkable___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_Int_shrinkable___lam__1___closed__0 = (const lean_object*)&lp_plausible_Plausible_Int_shrinkable___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Int_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Int_shrinkable___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Int_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_Int_shrinkable___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_Int_shrinkable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Int_shrinkable___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_plausible_Plausible_Int_shrinkable___closed__0_value)} };
static const lean_object* lp_plausible_Plausible_Int_shrinkable___closed__1 = (const lean_object*)&lp_plausible_Plausible_Int_shrinkable___closed__1_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Int_shrinkable = (const lean_object*)&lp_plausible_Plausible_Int_shrinkable___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Bool_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Bool_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_Bool_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Bool_shrinkable = (const lean_object*)&lp_plausible_Plausible_Bool_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Char_shrinkable___lam__0(uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Char_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Char_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Char_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Char_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_Char_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Char_shrinkable = (const lean_object*)&lp_plausible_Plausible_Char_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Option_shrinkable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Option_shrinkable___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Option_shrinkable___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Option_shrinkable___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__0 = (const lean_object*)&lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__1 = (const lean_object*)&lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_ULift_shrinkable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_ULift_shrinkable___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_plausible_Plausible_String_shrinkable___lam__0(uint32_t, uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__1(lean_object*, lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__2(lean_object*, lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Plausible_String_shrinkable___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_String_shrinkable___lam__3___closed__0 = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__3(lean_object*, lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_String_shrinkable___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_String_ofList___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_String_shrinkable___lam__4___closed__0 = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___lam__4___closed__0_value;
static const lean_array_object lp_plausible_Plausible_String_shrinkable___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_String_shrinkable___lam__4___closed__1 = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___lam__4___closed__1_value;
static const lean_closure_object lp_plausible_Plausible_String_shrinkable___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_String_shrinkable___lam__4___closed__2 = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___lam__4___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__4(lean_object*);
static const lean_closure_object lp_plausible_Plausible_String_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_String_shrinkable___lam__4, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_String_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_String_shrinkable = (const lean_object*)&lp_plausible_Plausible_String_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__4(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Array_shrinkable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Array_shrinkable___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Array_shrinkable___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__0(lean_object* v_val_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2_, 0, v_val_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__1(lean_object* v_val_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4_, 0, v_val_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg___lam__2(lean_object* v_inst_5_, lean_object* v___f_6_, lean_object* v_inst_7_, lean_object* v___f_8_, lean_object* v_s_9_){
_start:
{
if (lean_obj_tag(v_s_9_) == 0)
{
lean_object* v_val_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
lean_dec_ref(v___f_8_);
lean_dec_ref(v_inst_7_);
v_val_10_ = lean_ctor_get(v_s_9_, 0);
lean_inc(v_val_10_);
lean_dec_ref_known(v_s_9_, 1);
v___x_11_ = lean_apply_1(v_inst_5_, v_val_10_);
v___x_12_ = lean_box(0);
v___x_13_ = l_List_mapTR_loop___redArg(v___f_6_, v___x_11_, v___x_12_);
return v___x_13_;
}
else
{
lean_object* v_val_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
lean_dec_ref(v___f_6_);
lean_dec_ref(v_inst_5_);
v_val_14_ = lean_ctor_get(v_s_9_, 0);
lean_inc(v_val_14_);
lean_dec_ref_known(v_s_9_, 1);
v___x_15_ = lean_apply_1(v_inst_7_, v_val_14_);
v___x_16_ = lean_box(0);
v___x_17_ = l_List_mapTR_loop___redArg(v___f_8_, v___x_15_, v___x_16_);
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; lean_object* v___f_23_; lean_object* v___f_24_; 
v___f_22_ = ((lean_object*)(lp_plausible_Plausible_instShrinkableSum___redArg___closed__0));
v___f_23_ = ((lean_object*)(lp_plausible_Plausible_instShrinkableSum___redArg___closed__1));
v___f_24_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instShrinkableSum___redArg___lam__2), 5, 4);
lean_closure_set(v___f_24_, 0, v_inst_20_);
lean_closure_set(v___f_24_, 1, v___f_22_);
lean_closure_set(v___f_24_, 2, v_inst_21_);
lean_closure_set(v___f_24_, 3, v___f_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instShrinkableSum(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_plausible_Plausible_instShrinkableSum___redArg(v_inst_27_, v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Unit_shrinkable___lam__0(lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_box(0);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Nat_shrink(lean_object* v_n_34_){
_start:
{
lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_35_ = lean_unsigned_to_nat(0u);
v___x_36_ = lean_nat_dec_lt(v___x_35_, v_n_34_);
if (v___x_36_ == 0)
{
lean_object* v___x_37_; 
v___x_37_ = lean_box(0);
return v___x_37_;
}
else
{
lean_object* v___x_38_; lean_object* v_m_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_38_ = lean_unsigned_to_nat(1u);
v_m_39_ = lean_nat_shiftr(v_n_34_, v___x_38_);
v___x_40_ = lp_plausible_Plausible_Nat_shrink(v_m_39_);
v___x_41_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_41_, 0, v_m_39_);
lean_ctor_set(v___x_41_, 1, v___x_40_);
return v___x_41_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Nat_shrink___boxed(lean_object* v_n_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_plausible_Plausible_Nat_shrink(v_n_42_);
lean_dec(v_n_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable___lam__0(lean_object* v_n_46_, lean_object* v_m_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_48_ = lean_unsigned_to_nat(1u);
v___x_49_ = lean_nat_add(v_n_46_, v___x_48_);
v___x_50_ = lean_alloc_closure((void*)(l_Fin_ofNat___boxed), 3, 2);
lean_closure_set(v___x_50_, 0, v___x_49_);
lean_closure_set(v___x_50_, 1, lean_box(0));
v___x_51_ = lp_plausible_Plausible_Nat_shrink(v_m_47_);
v___x_52_ = lean_box(0);
v___x_53_ = l_List_mapTR_loop___redArg(v___x_50_, v___x_51_, v___x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable___lam__0___boxed(lean_object* v_n_54_, lean_object* v_m_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_plausible_Plausible_Fin_shrinkable___lam__0(v_n_54_, v_m_55_);
lean_dec(v_m_55_);
lean_dec(v_n_54_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Fin_shrinkable(lean_object* v_n_57_){
_start:
{
lean_object* v___f_58_; 
v___f_58_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Fin_shrinkable___lam__0___boxed), 2, 1);
lean_closure_set(v___f_58_, 0, v_n_57_);
return v___f_58_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable___lam__0(lean_object* v_n_59_, lean_object* v_m_60_){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = lean_alloc_closure((void*)(l_BitVec_ofNat___boxed), 2, 1);
lean_closure_set(v___x_61_, 0, v_n_59_);
v___x_62_ = lp_plausible_Plausible_Nat_shrink(v_m_60_);
v___x_63_ = lean_box(0);
v___x_64_ = l_List_mapTR_loop___redArg(v___x_61_, v___x_62_, v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable___lam__0___boxed(lean_object* v_n_65_, lean_object* v_m_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_plausible_Plausible_BitVec_shrinkable___lam__0(v_n_65_, v_m_66_);
lean_dec(v_m_66_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_BitVec_shrinkable(lean_object* v_n_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_plausible_Plausible_BitVec_shrinkable___lam__0___boxed), 2, 1);
lean_closure_set(v___f_69_, 0, v_n_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt8_shrinkable___lam__0(uint8_t v_m_71_){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_72_ = ((lean_object*)(lp_plausible_Plausible_UInt8_shrinkable___lam__0___closed__0));
v___x_73_ = lean_uint8_to_nat(v_m_71_);
v___x_74_ = lp_plausible_Plausible_Nat_shrink(v___x_73_);
v___x_75_ = lean_box(0);
v___x_76_ = l_List_mapTR_loop___redArg(v___x_72_, v___x_74_, v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt8_shrinkable___lam__0___boxed(lean_object* v_m_77_){
_start:
{
uint8_t v_m_boxed_78_; lean_object* v_res_79_; 
v_m_boxed_78_ = lean_unbox(v_m_77_);
v_res_79_ = lp_plausible_Plausible_UInt8_shrinkable___lam__0(v_m_boxed_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt16_shrinkable___lam__0(uint16_t v_m_83_){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_84_ = ((lean_object*)(lp_plausible_Plausible_UInt16_shrinkable___lam__0___closed__0));
v___x_85_ = lean_uint16_to_nat(v_m_83_);
v___x_86_ = lp_plausible_Plausible_Nat_shrink(v___x_85_);
v___x_87_ = lean_box(0);
v___x_88_ = l_List_mapTR_loop___redArg(v___x_84_, v___x_86_, v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt16_shrinkable___lam__0___boxed(lean_object* v_m_89_){
_start:
{
uint16_t v_m_boxed_90_; lean_object* v_res_91_; 
v_m_boxed_90_ = lean_unbox(v_m_89_);
v_res_91_ = lp_plausible_Plausible_UInt16_shrinkable___lam__0(v_m_boxed_90_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt32_shrinkable___lam__0(uint32_t v_m_95_){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_96_ = ((lean_object*)(lp_plausible_Plausible_UInt32_shrinkable___lam__0___closed__0));
v___x_97_ = lean_uint32_to_nat(v_m_95_);
v___x_98_ = lp_plausible_Plausible_Nat_shrink(v___x_97_);
lean_dec(v___x_97_);
v___x_99_ = lean_box(0);
v___x_100_ = l_List_mapTR_loop___redArg(v___x_96_, v___x_98_, v___x_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt32_shrinkable___lam__0___boxed(lean_object* v_m_101_){
_start:
{
uint32_t v_m_boxed_102_; lean_object* v_res_103_; 
v_m_boxed_102_ = lean_unbox_uint32(v_m_101_);
lean_dec(v_m_101_);
v_res_103_ = lp_plausible_Plausible_UInt32_shrinkable___lam__0(v_m_boxed_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt64_shrinkable___lam__0(uint64_t v_m_107_){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_108_ = ((lean_object*)(lp_plausible_Plausible_UInt64_shrinkable___lam__0___closed__0));
v___x_109_ = lean_uint64_to_nat(v_m_107_);
v___x_110_ = lp_plausible_Plausible_Nat_shrink(v___x_109_);
lean_dec(v___x_109_);
v___x_111_ = lean_box(0);
v___x_112_ = l_List_mapTR_loop___redArg(v___x_108_, v___x_110_, v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_UInt64_shrinkable___lam__0___boxed(lean_object* v_m_113_){
_start:
{
uint64_t v_m_boxed_114_; lean_object* v_res_115_; 
v_m_boxed_114_ = lean_unbox_uint64(v_m_113_);
lean_dec_ref(v_m_113_);
v_res_115_ = lp_plausible_Plausible_UInt64_shrinkable___lam__0(v_m_boxed_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_USize_shrinkable___lam__0(size_t v_m_119_){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_120_ = ((lean_object*)(lp_plausible_Plausible_USize_shrinkable___lam__0___closed__0));
v___x_121_ = lean_usize_to_nat(v_m_119_);
v___x_122_ = lp_plausible_Plausible_Nat_shrink(v___x_121_);
lean_dec(v___x_121_);
v___x_123_ = lean_box(0);
v___x_124_ = l_List_mapTR_loop___redArg(v___x_120_, v___x_122_, v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_USize_shrinkable___lam__0___boxed(lean_object* v_m_125_){
_start:
{
size_t v_m_boxed_126_; lean_object* v_res_127_; 
v_m_boxed_126_ = lean_unbox_usize(v_m_125_);
lean_dec(v_m_125_);
v_res_127_ = lp_plausible_Plausible_USize_shrinkable___lam__0(v_m_boxed_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__0(lean_object* v_n_130_){
_start:
{
lean_object* v_int_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_int_131_ = lean_nat_to_int(v_n_130_);
v___x_132_ = lean_int_neg(v_int_131_);
v___x_133_ = lean_box(0);
v___x_134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_132_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v_int_131_);
lean_ctor_set(v___x_135_, 1, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__1(lean_object* v_converter_138_, lean_object* v_n_139_){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_140_ = lean_nat_abs(v_n_139_);
v___x_141_ = lp_plausible_Plausible_Nat_shrink(v___x_140_);
lean_dec(v___x_140_);
v___x_142_ = ((lean_object*)(lp_plausible_Plausible_Int_shrinkable___lam__1___closed__0));
v___x_143_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v_converter_138_, v___x_141_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Int_shrinkable___lam__1___boxed(lean_object* v_converter_144_, lean_object* v_n_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_plausible_Plausible_Int_shrinkable___lam__1(v_converter_144_, v_n_145_);
lean_dec(v_n_145_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0(uint8_t v_x_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lean_box(0);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed(lean_object* v_x_153_){
_start:
{
uint8_t v_x_7__boxed_154_; lean_object* v_res_155_; 
v_x_7__boxed_154_ = lean_unbox(v_x_153_);
v_res_155_ = lp_plausible_Plausible_Bool_shrinkable___lam__0(v_x_7__boxed_154_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Char_shrinkable___lam__0(uint32_t v_x_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lean_box(0);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Char_shrinkable___lam__0___boxed(lean_object* v_x_160_){
_start:
{
uint32_t v_x_7__boxed_161_; lean_object* v_res_162_; 
v_x_7__boxed_161_ = lean_unbox_uint32(v_x_160_);
lean_dec(v_x_160_);
v_res_162_ = lp_plausible_Plausible_Char_shrinkable___lam__0(v_x_7__boxed_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg___lam__0(lean_object* v_val_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_166_, 0, v_val_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg___lam__1(lean_object* v_inst_167_, lean_object* v___f_168_, lean_object* v_o_169_){
_start:
{
if (lean_obj_tag(v_o_169_) == 0)
{
lean_object* v___x_170_; 
lean_dec_ref(v___f_168_);
lean_dec_ref(v_inst_167_);
v___x_170_ = lean_box(0);
return v___x_170_;
}
else
{
lean_object* v_val_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v_val_171_ = lean_ctor_get(v_o_169_, 0);
lean_inc(v_val_171_);
lean_dec_ref_known(v_o_169_, 1);
v___x_172_ = lean_apply_1(v_inst_167_, v_val_171_);
v___x_173_ = lean_box(0);
v___x_174_ = l_List_mapTR_loop___redArg(v___f_168_, v___x_172_, v___x_173_);
return v___x_174_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable___redArg(lean_object* v_inst_176_){
_start:
{
lean_object* v___f_177_; lean_object* v___f_178_; 
v___f_177_ = ((lean_object*)(lp_plausible_Plausible_Option_shrinkable___redArg___closed__0));
v___f_178_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Option_shrinkable___redArg___lam__1), 3, 2);
lean_closure_set(v___f_178_, 0, v_inst_176_);
lean_closure_set(v___f_178_, 1, v___f_177_);
return v___f_178_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_shrinkable(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_plausible_Plausible_Option_shrinkable___redArg(v_inst_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__0(lean_object* v_snd_182_, lean_object* v_x_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_184_, 0, v_x_183_);
lean_ctor_set(v___x_184_, 1, v_snd_182_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__1(lean_object* v_fst_185_, lean_object* v_x_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v_fst_185_);
lean_ctor_set(v___x_187_, 1, v_x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2(lean_object* v_shrA_188_, lean_object* v_shrB_189_, lean_object* v_x_190_){
_start:
{
lean_object* v_fst_191_; lean_object* v_snd_192_; lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v_shrink1_197_; lean_object* v___x_198_; lean_object* v_shrink2_199_; lean_object* v___x_200_; 
v_fst_191_ = lean_ctor_get(v_x_190_, 0);
lean_inc_n(v_fst_191_, 2);
v_snd_192_ = lean_ctor_get(v_x_190_, 1);
lean_inc_n(v_snd_192_, 2);
lean_dec_ref(v_x_190_);
v___f_193_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Prod_shrinkable___redArg___lam__0), 2, 1);
lean_closure_set(v___f_193_, 0, v_snd_192_);
v___f_194_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Prod_shrinkable___redArg___lam__1), 2, 1);
lean_closure_set(v___f_194_, 0, v_fst_191_);
v___x_195_ = lean_apply_1(v_shrA_188_, v_fst_191_);
v___x_196_ = lean_box(0);
v_shrink1_197_ = l_List_mapTR_loop___redArg(v___f_193_, v___x_195_, v___x_196_);
v___x_198_ = lean_apply_1(v_shrB_189_, v_snd_192_);
v_shrink2_199_ = l_List_mapTR_loop___redArg(v___f_194_, v___x_198_, v___x_196_);
v___x_200_ = l_List_appendTR___redArg(v_shrink1_197_, v_shrink2_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg(lean_object* v_shrA_201_, lean_object* v_shrB_202_){
_start:
{
lean_object* v___f_203_; 
v___f_203_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_203_, 0, v_shrA_201_);
lean_closure_set(v___f_203_, 1, v_shrB_202_);
return v___f_203_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_shrinkable(lean_object* v_00_u03b1_204_, lean_object* v_00_u03b2_205_, lean_object* v_shrA_206_, lean_object* v_shrB_207_){
_start:
{
lean_object* v___f_208_; 
v___f_208_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_208_, 0, v_shrA_206_);
lean_closure_set(v___f_208_, 1, v_shrB_207_);
return v___f_208_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__0(lean_object* v_snd_209_, lean_object* v_x_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_211_, 0, v_x_210_);
lean_ctor_set(v___x_211_, 1, v_snd_209_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__1(lean_object* v_fst_212_, lean_object* v_x_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v_fst_212_);
lean_ctor_set(v___x_214_, 1, v_x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2(lean_object* v_shrA_215_, lean_object* v_shrB_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_fst_218_; lean_object* v_snd_219_; lean_object* v___f_220_; lean_object* v___f_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_shrink1_224_; lean_object* v___x_225_; lean_object* v_shrink2_226_; lean_object* v___x_227_; 
v_fst_218_ = lean_ctor_get(v_x_217_, 0);
lean_inc_n(v_fst_218_, 2);
v_snd_219_ = lean_ctor_get(v_x_217_, 1);
lean_inc_n(v_snd_219_, 2);
lean_dec_ref(v_x_217_);
v___f_220_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__0), 2, 1);
lean_closure_set(v___f_220_, 0, v_snd_219_);
v___f_221_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__1), 2, 1);
lean_closure_set(v___f_221_, 0, v_fst_218_);
v___x_222_ = lean_apply_1(v_shrA_215_, v_fst_218_);
v___x_223_ = lean_box(0);
v_shrink1_224_ = l_List_mapTR_loop___redArg(v___f_220_, v___x_222_, v___x_223_);
v___x_225_ = lean_apply_1(v_shrB_216_, v_snd_219_);
v_shrink2_226_ = l_List_mapTR_loop___redArg(v___f_221_, v___x_225_, v___x_223_);
v___x_227_ = l_List_appendTR___redArg(v_shrink1_224_, v_shrink2_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg(lean_object* v_shrA_228_, lean_object* v_shrB_229_){
_start:
{
lean_object* v___f_230_; 
v___f_230_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_230_, 0, v_shrA_228_);
lean_closure_set(v___f_230_, 1, v_shrB_229_);
return v___f_230_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sigma_shrinkable(lean_object* v_00_u03b1_231_, lean_object* v_00_u03b2_232_, lean_object* v_shrA_233_, lean_object* v_shrB_234_){
_start:
{
lean_object* v___f_235_; 
v___f_235_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_235_, 0, v_shrA_233_);
lean_closure_set(v___f_235_, 1, v_shrB_234_);
return v___f_235_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__0(lean_object* v_L_238_, lean_object* v_i_239_, lean_object* v_x_240_){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_241_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0));
lean_inc(v_L_238_);
v___x_242_ = l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_box(0), v_L_238_, v_L_238_, v_i_239_, v___x_241_);
lean_dec(v_L_238_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__0___boxed(lean_object* v_L_243_, lean_object* v_i_244_, lean_object* v_x_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_plausible_Plausible_List_shrinkable___redArg___lam__0(v_L_243_, v_i_244_, v_x_245_);
lean_dec(v_x_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__1(lean_object* v_a_x27_247_, lean_object* v_x_248_){
_start:
{
lean_inc(v_a_x27_247_);
return v_a_x27_247_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__1___boxed(lean_object* v_a_x27_249_, lean_object* v_x_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_plausible_Plausible_List_shrinkable___redArg___lam__1(v_a_x27_249_, v_x_250_);
lean_dec(v_x_250_);
lean_dec(v_a_x27_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__2(lean_object* v_L_252_, lean_object* v_i_253_, lean_object* v_a_x27_254_){
_start:
{
lean_object* v___f_255_; lean_object* v___x_256_; 
v___f_255_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_255_, 0, v_a_x27_254_);
v___x_256_ = l_List_modifyTR___redArg(v_L_252_, v_i_253_, v___f_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__3(lean_object* v_L_257_, lean_object* v_inst_258_, lean_object* v_i_259_, lean_object* v_a_260_){
_start:
{
lean_object* v___f_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
v___f_261_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_261_, 0, v_L_257_);
lean_closure_set(v___f_261_, 1, v_i_259_);
v___x_262_ = lean_apply_1(v_inst_258_, v_a_260_);
v___x_263_ = lean_box(0);
v___x_264_ = l_List_mapTR_loop___redArg(v___f_261_, v___x_262_, v___x_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__4(lean_object* v_inst_268_, lean_object* v_L_269_){
_start:
{
lean_object* v___f_270_; lean_object* v___f_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
lean_inc_n(v_L_269_, 3);
v___f_270_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_270_, 0, v_L_269_);
v___f_271_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__3), 4, 2);
lean_closure_set(v___f_271_, 0, v_L_269_);
lean_closure_set(v___f_271_, 1, v_inst_268_);
v___x_272_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__0));
v___x_273_ = l_List_mapIdx_go___redArg(v___f_270_, v_L_269_, v___x_272_);
v___x_274_ = l_List_mapIdx_go___redArg(v___f_271_, v_L_269_, v___x_272_);
v___x_275_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__1));
v___x_276_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v___x_275_, v___x_274_, v___x_272_);
v___x_277_ = l_List_appendTR___redArg(v___x_273_, v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable___redArg(lean_object* v_inst_278_){
_start:
{
lean_object* v___f_279_; 
v___f_279_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4), 2, 1);
lean_closure_set(v___f_279_, 0, v_inst_278_);
return v___f_279_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_shrinkable(lean_object* v_00_u03b1_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___f_282_; 
v___f_282_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4), 2, 1);
lean_closure_set(v___f_282_, 0, v_inst_281_);
return v___f_282_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0(lean_object* v_down_283_){
_start:
{
lean_inc(v_down_283_);
return v_down_283_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0___boxed(lean_object* v_down_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_plausible_Plausible_ULift_shrinkable___redArg___lam__0(v_down_284_);
lean_dec(v_down_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg___lam__1(lean_object* v_inst_286_, lean_object* v___f_287_, lean_object* v_u_288_){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_289_ = lean_apply_1(v_inst_286_, v_u_288_);
v___x_290_ = lean_box(0);
v___x_291_ = l_List_mapTR_loop___redArg(v___f_287_, v___x_289_, v___x_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable___redArg(lean_object* v_inst_293_){
_start:
{
lean_object* v___f_294_; lean_object* v___f_295_; 
v___f_294_ = ((lean_object*)(lp_plausible_Plausible_ULift_shrinkable___redArg___closed__0));
v___f_295_ = lean_alloc_closure((void*)(lp_plausible_Plausible_ULift_shrinkable___redArg___lam__1), 3, 2);
lean_closure_set(v___f_295_, 0, v_inst_293_);
lean_closure_set(v___f_295_, 1, v___f_294_);
return v___f_295_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_shrinkable(lean_object* v_00_u03b1_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_plausible_Plausible_ULift_shrinkable___redArg(v_inst_297_);
return v___x_298_;
}
}
LEAN_EXPORT uint32_t lp_plausible_Plausible_String_shrinkable___lam__0(uint32_t v_a_x27_299_, uint32_t v_x_300_){
_start:
{
return v_a_x27_299_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__0___boxed(lean_object* v_a_x27_301_, lean_object* v_x_302_){
_start:
{
uint32_t v_a_x27_boxed_303_; uint32_t v_x_91__boxed_304_; uint32_t v_res_305_; lean_object* v_r_306_; 
v_a_x27_boxed_303_ = lean_unbox_uint32(v_a_x27_301_);
lean_dec(v_a_x27_301_);
v_x_91__boxed_304_ = lean_unbox_uint32(v_x_302_);
lean_dec(v_x_302_);
v_res_305_ = lp_plausible_Plausible_String_shrinkable___lam__0(v_a_x27_boxed_303_, v_x_91__boxed_304_);
v_r_306_ = lean_box_uint32(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__1(lean_object* v___x_307_, lean_object* v_i_308_, uint32_t v_a_x27_309_){
_start:
{
lean_object* v___x_310_; lean_object* v___f_311_; lean_object* v___x_312_; 
v___x_310_ = lean_box_uint32(v_a_x27_309_);
v___f_311_ = lean_alloc_closure((void*)(lp_plausible_Plausible_String_shrinkable___lam__0___boxed), 2, 1);
lean_closure_set(v___f_311_, 0, v___x_310_);
v___x_312_ = l_List_modifyTR___redArg(v___x_307_, v_i_308_, v___f_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__1___boxed(lean_object* v___x_313_, lean_object* v_i_314_, lean_object* v_a_x27_315_){
_start:
{
uint32_t v_a_x27_boxed_316_; lean_object* v_res_317_; 
v_a_x27_boxed_316_ = lean_unbox_uint32(v_a_x27_315_);
lean_dec(v_a_x27_315_);
v_res_317_ = lp_plausible_Plausible_String_shrinkable___lam__1(v___x_313_, v_i_314_, v_a_x27_boxed_316_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__2(lean_object* v___x_318_, lean_object* v_i_319_, uint32_t v_a_320_){
_start:
{
lean_object* v___f_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v___f_321_ = lean_alloc_closure((void*)(lp_plausible_Plausible_String_shrinkable___lam__1___boxed), 3, 2);
lean_closure_set(v___f_321_, 0, v___x_318_);
lean_closure_set(v___f_321_, 1, v_i_319_);
v___x_322_ = lean_box(0);
v___x_323_ = l_List_mapTR_loop___redArg(v___f_321_, v___x_322_, v___x_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__2___boxed(lean_object* v___x_324_, lean_object* v_i_325_, lean_object* v_a_326_){
_start:
{
uint32_t v_a_boxed_327_; lean_object* v_res_328_; 
v_a_boxed_327_ = lean_unbox_uint32(v_a_326_);
lean_dec(v_a_326_);
v_res_328_ = lp_plausible_Plausible_String_shrinkable___lam__2(v___x_324_, v_i_325_, v_a_boxed_327_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__3(lean_object* v___x_331_, lean_object* v_i_332_, uint32_t v_x_333_){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = ((lean_object*)(lp_plausible_Plausible_String_shrinkable___lam__3___closed__0));
lean_inc(v___x_331_);
v___x_335_ = l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_box(0), v___x_331_, v___x_331_, v_i_332_, v___x_334_);
lean_dec(v___x_331_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__3___boxed(lean_object* v___x_336_, lean_object* v_i_337_, lean_object* v_x_338_){
_start:
{
uint32_t v_x_117__boxed_339_; lean_object* v_res_340_; 
v_x_117__boxed_339_ = lean_unbox_uint32(v_x_338_);
lean_dec(v_x_338_);
v_res_340_ = lp_plausible_Plausible_String_shrinkable___lam__3(v___x_336_, v_i_337_, v_x_117__boxed_339_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_String_shrinkable___lam__4(lean_object* v_s_345_){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___f_348_; lean_object* v___f_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_346_ = ((lean_object*)(lp_plausible_Plausible_String_shrinkable___lam__4___closed__0));
v___x_347_ = lean_string_data(v_s_345_);
lean_inc_n(v___x_347_, 3);
v___f_348_ = lean_alloc_closure((void*)(lp_plausible_Plausible_String_shrinkable___lam__2___boxed), 3, 1);
lean_closure_set(v___f_348_, 0, v___x_347_);
v___f_349_ = lean_alloc_closure((void*)(lp_plausible_Plausible_String_shrinkable___lam__3___boxed), 3, 1);
lean_closure_set(v___f_349_, 0, v___x_347_);
v___x_350_ = ((lean_object*)(lp_plausible_Plausible_String_shrinkable___lam__4___closed__1));
v___x_351_ = l_List_mapIdx_go___redArg(v___f_349_, v___x_347_, v___x_350_);
v___x_352_ = l_List_mapIdx_go___redArg(v___f_348_, v___x_347_, v___x_350_);
v___x_353_ = ((lean_object*)(lp_plausible_Plausible_String_shrinkable___lam__4___closed__2));
v___x_354_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v___x_353_, v___x_352_, v___x_350_);
v___x_355_ = l_List_appendTR___redArg(v___x_351_, v___x_354_);
v___x_356_ = lean_box(0);
v___x_357_ = l_List_mapTR_loop___redArg(v___x_346_, v___x_355_, v___x_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__0(lean_object* v_toList_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lean_array_mk(v_toList_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__2(lean_object* v___x_362_, lean_object* v_i_363_, lean_object* v_a_x27_364_){
_start:
{
lean_object* v___f_365_; lean_object* v___x_366_; 
v___f_365_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_365_, 0, v_a_x27_364_);
v___x_366_ = l_List_modifyTR___redArg(v___x_362_, v_i_363_, v___f_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__1(lean_object* v___x_367_, lean_object* v_inst_368_, lean_object* v_i_369_, lean_object* v_a_370_){
_start:
{
lean_object* v___f_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
v___f_371_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Array_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_371_, 0, v___x_367_);
lean_closure_set(v___f_371_, 1, v_i_369_);
v___x_372_ = lean_apply_1(v_inst_368_, v_a_370_);
v___x_373_ = lean_box(0);
v___x_374_ = l_List_mapTR_loop___redArg(v___f_371_, v___x_372_, v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__3(lean_object* v___x_375_, lean_object* v_i_376_, lean_object* v_x_377_){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0));
lean_inc(v___x_375_);
v___x_379_ = l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_box(0), v___x_375_, v___x_375_, v_i_376_, v___x_378_);
lean_dec(v___x_375_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__3___boxed(lean_object* v___x_380_, lean_object* v_i_381_, lean_object* v_x_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_plausible_Plausible_Array_shrinkable___redArg___lam__3(v___x_380_, v_i_381_, v_x_382_);
lean_dec(v_x_382_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg___lam__4(lean_object* v_inst_384_, lean_object* v___f_385_, lean_object* v_xs_386_){
_start:
{
lean_object* v___x_387_; lean_object* v___f_388_; lean_object* v___f_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_387_ = lean_array_to_list(v_xs_386_);
lean_inc_n(v___x_387_, 3);
v___f_388_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Array_shrinkable___redArg___lam__1), 4, 2);
lean_closure_set(v___f_388_, 0, v___x_387_);
lean_closure_set(v___f_388_, 1, v_inst_384_);
v___f_389_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Array_shrinkable___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_389_, 0, v___x_387_);
v___x_390_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__0));
v___x_391_ = l_List_mapIdx_go___redArg(v___f_389_, v___x_387_, v___x_390_);
v___x_392_ = l_List_mapIdx_go___redArg(v___f_388_, v___x_387_, v___x_390_);
v___x_393_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4___closed__1));
v___x_394_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v___x_393_, v___x_392_, v___x_390_);
v___x_395_ = l_List_appendTR___redArg(v___x_391_, v___x_394_);
v___x_396_ = lean_box(0);
v___x_397_ = l_List_mapTR_loop___redArg(v___f_385_, v___x_395_, v___x_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable___redArg(lean_object* v_inst_399_){
_start:
{
lean_object* v___f_400_; lean_object* v___f_401_; 
v___f_400_ = ((lean_object*)(lp_plausible_Plausible_Array_shrinkable___redArg___closed__0));
v___f_401_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Array_shrinkable___redArg___lam__4), 3, 2);
lean_closure_set(v___f_401_, 0, v_inst_399_);
lean_closure_set(v___f_401_, 1, v___f_400_);
return v___f_401_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_shrinkable(lean_object* v_00_u03b1_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_plausible_Plausible_Array_shrinkable___redArg(v_inst_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__0(lean_object* v_inst_405_, lean_object* v_x_406_){
_start:
{
lean_object* v___x_407_; uint8_t v___x_408_; 
lean_inc(v_x_406_);
v___x_407_ = lean_apply_1(v_inst_405_, v_x_406_);
v___x_408_ = lean_unbox(v___x_407_);
if (v___x_408_ == 0)
{
lean_object* v___x_409_; 
lean_dec(v_x_406_);
v___x_409_ = lean_box(0);
return v___x_409_;
}
else
{
lean_object* v___x_410_; 
v___x_410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_410_, 0, v_x_406_);
return v___x_410_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__1(lean_object* v_inst_411_, lean_object* v_filter_412_, lean_object* v_x_413_){
_start:
{
lean_object* v_candidates_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v_candidates_414_ = lean_apply_1(v_inst_411_, v_x_413_);
v___x_415_ = ((lean_object*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__0___closed__0));
v___x_416_ = l_List_filterMapTR_go___redArg(v_filter_412_, v_candidates_414_, v___x_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable___redArg(lean_object* v_inst_417_, lean_object* v_inst_418_){
_start:
{
lean_object* v_filter_419_; lean_object* v___f_420_; 
v_filter_419_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__0), 2, 1);
lean_closure_set(v_filter_419_, 0, v_inst_418_);
v___f_420_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Subtype_shrinkable___redArg___lam__1), 3, 2);
lean_closure_set(v___f_420_, 0, v_inst_417_);
lean_closure_set(v___f_420_, 1, v_filter_419_);
return v___f_420_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Subtype_shrinkable(lean_object* v_00_u03b1_421_, lean_object* v_00_u03b2_422_, lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_plausible_Plausible_Subtype_shrinkable___redArg(v_inst_423_, v_inst_424_);
return v___x_425_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Shrinkable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Shrinkable(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Shrinkable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Shrinkable(builtin);
}
#ifdef __cplusplus
}
#endif
