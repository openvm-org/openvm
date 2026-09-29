// Lean compiler output
// Module: Mathlib.Logic.Encodable.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Countable.Defs public import Mathlib.Data.Fin.Basic public import Mathlib.Data.Nat.Find public import Mathlib.Data.PNat.Equiv public import Mathlib.Logic.Equiv.Nat public import Mathlib.Order.Directed public import Mathlib.Order.RelIso.Basic
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Nat_testBit(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unpair(lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_equivSubtype(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Equiv_intEquivNat;
lean_object* lp_mathlib_Equiv_sigmaEquivProd(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_plift(lean_object*);
lean_object* lp_mathlib_Nat_pair(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit;
extern lean_object* lp_mathlib_Equiv_pnatEquivNat;
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableEqOfEncodable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableEqOfEncodable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse___redArg___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Encodable_ofLeftInverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Encodable_ofLeftInverse___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Encodable_ofLeftInverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_Encodable_ofLeftInverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_encodable___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Nat_encodable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_encodable___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_encodable___closed__0 = (const lean_object*)&lp_mathlib_Nat_encodable___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_encodable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Nat_encodable___closed__1 = (const lean_object*)&lp_mathlib_Nat_encodable___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_encodable___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat_encodable___closed__1_value),((lean_object*)&lp_mathlib_Nat_encodable___closed__0_value)}};
static const lean_object* lp_mathlib_Nat_encodable___closed__2 = (const lean_object*)&lp_mathlib_Nat_encodable___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_encodable = (const lean_object*)&lp_mathlib_Nat_encodable___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_IsEmpty_toEncodable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_IsEmpty_toEncodable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_IsEmpty_toEncodable___closed__0 = (const lean_object*)&lp_mathlib_IsEmpty_toEncodable___closed__0_value;
static const lean_closure_object lp_mathlib_IsEmpty_toEncodable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_IsEmpty_toEncodable___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_IsEmpty_toEncodable___closed__1 = (const lean_object*)&lp_mathlib_IsEmpty_toEncodable___closed__1_value;
static const lean_ctor_object lp_mathlib_IsEmpty_toEncodable___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_IsEmpty_toEncodable___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toEncodable___closed__1_value)}};
static const lean_object* lp_mathlib_IsEmpty_toEncodable___closed__2 = (const lean_object*)&lp_mathlib_IsEmpty_toEncodable___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__0(lean_object*);
static const lean_ctor_object lp_mathlib_PUnit_encodable___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PUnit_encodable___lam__1___closed__0 = (const lean_object*)&lp_mathlib_PUnit_encodable___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PUnit_encodable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_encodable___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_encodable___closed__0 = (const lean_object*)&lp_mathlib_PUnit_encodable___closed__0_value;
static const lean_closure_object lp_mathlib_PUnit_encodable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_encodable___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_encodable___closed__1 = (const lean_object*)&lp_mathlib_PUnit_encodable___closed__1_value;
static const lean_ctor_object lp_mathlib_PUnit_encodable___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_PUnit_encodable___closed__0_value),((lean_object*)&lp_mathlib_PUnit_encodable___closed__1_value)}};
static const lean_object* lp_mathlib_PUnit_encodable___closed__2 = (const lean_object*)&lp_mathlib_PUnit_encodable___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_PUnit_encodable = (const lean_object*)&lp_mathlib_PUnit_encodable___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__0(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Option_encodable___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Option_encodable___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Option_encodable___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decode_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decode_u2082(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableRangeEncode___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableRangeEncode___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableRangeEncode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableRangeEncode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Unique_encodable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Unique_encodable___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Unique_encodable___redArg___closed__0 = (const lean_object*)&lp_mathlib_Unique_encodable___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSum_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSum_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_encodable(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Bool_encodable___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_encodable___closed__0;
static lean_once_cell_t lp_mathlib_Bool_encodable___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_encodable___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Bool_encodable;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSigma___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSigma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSigma_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSigma_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_encodable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Encodable_Prod_encodable___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Encodable_Prod_encodable___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSubtype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSubtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSubtype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSubtype_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSubtype_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_encodable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fin_encodable___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_encodable___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_encodable(lean_object*);
static lean_once_cell_t lp_mathlib_Int_encodable___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_encodable___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_encodable;
static lean_once_cell_t lp_mathlib_PNat_encodable___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PNat_encodable___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PNat_encodable;
static lean_once_cell_t lp_mathlib_ULift_encodable___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_encodable___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ULift_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_encodable(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PLift_encodable___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PLift_encodable___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PLift_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PLift_encodable(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqULower___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqULower___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqULower(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqULower___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableULower___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableULower(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_equiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_down___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_down(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_up___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULower_up(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_good_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_good_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Quotient_rep___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableEqOfEncodable___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
lean_object* v_encode_4_; lean_object* v___x_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v_encode_4_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref_n(v_encode_4_, 2);
lean_dec_ref(v_inst_1_);
v___x_5_ = lean_apply_1(v_encode_4_, v_x_2_);
v___x_6_ = lean_apply_1(v_encode_4_, v_x_3_);
v___x_7_ = lean_nat_dec_eq(v___x_5_, v___x_6_);
lean_dec(v___x_6_);
lean_dec(v___x_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___redArg___boxed(lean_object* v_inst_8_, lean_object* v_x_9_, lean_object* v_x_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Encodable_decidableEqOfEncodable___redArg(v_inst_8_, v_x_9_, v_x_10_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableEqOfEncodable(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_, lean_object* v_x_15_, lean_object* v_x_16_){
_start:
{
uint8_t v___x_17_; 
v___x_17_ = lp_mathlib_Encodable_decidableEqOfEncodable___redArg(v_inst_14_, v_x_15_, v_x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___boxed(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_, lean_object* v_x_20_, lean_object* v_x_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_Encodable_decidableEqOfEncodable(v_00_u03b1_18_, v_inst_19_, v_x_20_, v_x_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg___lam__0(lean_object* v_inst_24_, lean_object* v_f_25_, lean_object* v_b_26_){
_start:
{
lean_object* v_encode_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v_encode_27_ = lean_ctor_get(v_inst_24_, 0);
lean_inc_ref(v_encode_27_);
lean_dec_ref(v_inst_24_);
v___x_28_ = lean_apply_1(v_f_25_, v_b_26_);
v___x_29_ = lean_apply_1(v_encode_27_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg___lam__1(lean_object* v_inst_30_, lean_object* v_finv_31_, lean_object* v_n_32_){
_start:
{
lean_object* v_decode_33_; lean_object* v___x_34_; 
v_decode_33_ = lean_ctor_get(v_inst_30_, 1);
lean_inc_ref(v_decode_33_);
lean_dec_ref(v_inst_30_);
v___x_34_ = lean_apply_1(v_decode_33_, v_n_32_);
if (lean_obj_tag(v___x_34_) == 0)
{
lean_object* v___x_35_; 
lean_dec_ref(v_finv_31_);
v___x_35_ = lean_box(0);
return v___x_35_;
}
else
{
lean_object* v_val_36_; lean_object* v___x_37_; 
v_val_36_ = lean_ctor_get(v___x_34_, 0);
lean_inc(v_val_36_);
lean_dec_ref_known(v___x_34_, 1);
v___x_37_ = lean_apply_1(v_finv_31_, v_val_36_);
return v___x_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection___redArg(lean_object* v_inst_38_, lean_object* v_f_39_, lean_object* v_finv_40_){
_start:
{
lean_object* v___f_41_; lean_object* v___f_42_; lean_object* v___x_43_; 
lean_inc_ref(v_inst_38_);
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_ofLeftInjection___redArg___lam__0), 3, 2);
lean_closure_set(v___f_41_, 0, v_inst_38_);
lean_closure_set(v___f_41_, 1, v_f_39_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_ofLeftInjection___redArg___lam__1), 3, 2);
lean_closure_set(v___f_42_, 0, v_inst_38_);
lean_closure_set(v___f_42_, 1, v_finv_40_);
v___x_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_43_, 0, v___f_41_);
lean_ctor_set(v___x_43_, 1, v___f_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInjection(lean_object* v_00_u03b1_44_, lean_object* v_00_u03b2_45_, lean_object* v_inst_46_, lean_object* v_f_47_, lean_object* v_finv_48_, lean_object* v_linv_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_Encodable_ofLeftInjection___redArg(v_inst_46_, v_f_47_, v_finv_48_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse___redArg___lam__0(lean_object* v_val_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_52_, 0, v_val_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse___redArg(lean_object* v_inst_54_, lean_object* v_f_55_, lean_object* v_finv_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___f_57_ = ((lean_object*)(lp_mathlib_Encodable_ofLeftInverse___redArg___closed__0));
v___x_58_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_58_, 0, lean_box(0));
lean_closure_set(v___x_58_, 1, lean_box(0));
lean_closure_set(v___x_58_, 2, lean_box(0));
lean_closure_set(v___x_58_, 3, v___f_57_);
lean_closure_set(v___x_58_, 4, v_finv_56_);
v___x_59_ = lp_mathlib_Encodable_ofLeftInjection___redArg(v_inst_54_, v_f_55_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofLeftInverse(lean_object* v_00_u03b1_60_, lean_object* v_00_u03b2_61_, lean_object* v_inst_62_, lean_object* v_f_63_, lean_object* v_finv_64_, lean_object* v_linv_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Encodable_ofLeftInverse___redArg(v_inst_62_, v_f_63_, v_finv_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg___lam__0(lean_object* v_e_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_toFun_69_; lean_object* v___x_70_; 
v_toFun_69_ = lean_ctor_get(v_e_67_, 0);
lean_inc(v_toFun_69_);
lean_dec_ref(v_e_67_);
v___x_70_ = lean_apply_1(v_toFun_69_, v___y_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg___lam__1(lean_object* v___x_71_, lean_object* v___y_72_){
_start:
{
lean_object* v_toFun_73_; lean_object* v___x_74_; 
v_toFun_73_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toFun_73_);
lean_dec_ref(v___x_71_);
v___x_74_ = lean_apply_1(v_toFun_73_, v___y_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object* v_inst_75_, lean_object* v_e_76_){
_start:
{
lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___f_79_; lean_object* v___x_80_; 
lean_inc_ref(v_e_76_);
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_ofEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_77_, 0, v_e_76_);
v___x_78_ = lp_mathlib_Equiv_symm___redArg(v_e_76_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_ofEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_79_, 0, v___x_78_);
v___x_80_ = lp_mathlib_Encodable_ofLeftInverse___redArg(v_inst_75_, v___f_77_, v___f_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_ofEquiv(lean_object* v_00_u03b2_81_, lean_object* v_00_u03b1_82_, lean_object* v_inst_83_, lean_object* v_e_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_83_, v_e_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_encodable___lam__0(lean_object* v_val_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_87_, 0, v_val_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__0(lean_object* v_a_94_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__0___boxed(lean_object* v_a_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_IsEmpty_toEncodable___lam__0(v_a_95_);
lean_dec(v_a_95_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__1(lean_object* v_x_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_box(0);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable___lam__1___boxed(lean_object* v_x_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_IsEmpty_toEncodable___lam__1(v_x_99_);
lean_dec(v_x_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toEncodable(lean_object* v_00_u03b1_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = ((lean_object*)(lp_mathlib_IsEmpty_toEncodable___closed__2));
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__0(lean_object* v_x_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_unsigned_to_nat(0u);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__1(lean_object* v_n_113_){
_start:
{
lean_object* v_zero_114_; uint8_t v_isZero_115_; 
v_zero_114_ = lean_unsigned_to_nat(0u);
v_isZero_115_ = lean_nat_dec_eq(v_n_113_, v_zero_114_);
if (v_isZero_115_ == 1)
{
lean_object* v___x_116_; 
v___x_116_ = ((lean_object*)(lp_mathlib_PUnit_encodable___lam__1___closed__0));
return v___x_116_;
}
else
{
lean_object* v___x_117_; 
v___x_117_ = lean_box(0);
return v___x_117_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_encodable___lam__1___boxed(lean_object* v_n_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_PUnit_encodable___lam__1(v_n_118_);
lean_dec(v_n_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__0(lean_object* v_h_126_, lean_object* v_o_127_){
_start:
{
if (lean_obj_tag(v_o_127_) == 0)
{
lean_object* v___x_128_; 
lean_dec_ref(v_h_126_);
v___x_128_ = lean_unsigned_to_nat(0u);
return v___x_128_;
}
else
{
lean_object* v_a_129_; lean_object* v_encode_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v_a_129_ = lean_ctor_get(v_o_127_, 0);
lean_inc(v_a_129_);
lean_dec_ref_known(v_o_127_, 1);
v_encode_130_ = lean_ctor_get(v_h_126_, 0);
lean_inc_ref(v_encode_130_);
lean_dec_ref(v_h_126_);
v___x_131_ = lean_apply_1(v_encode_130_, v_a_129_);
v___x_132_ = lean_unsigned_to_nat(1u);
v___x_133_ = lean_nat_add(v___x_131_, v___x_132_);
lean_dec(v___x_131_);
return v___x_133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__1(lean_object* v_h_136_, lean_object* v_n_137_){
_start:
{
lean_object* v_zero_138_; uint8_t v_isZero_139_; 
v_zero_138_ = lean_unsigned_to_nat(0u);
v_isZero_139_ = lean_nat_dec_eq(v_n_137_, v_zero_138_);
if (v_isZero_139_ == 1)
{
lean_object* v___x_140_; 
lean_dec_ref(v_h_136_);
v___x_140_ = ((lean_object*)(lp_mathlib_Option_encodable___redArg___lam__1___closed__0));
return v___x_140_;
}
else
{
lean_object* v_decode_141_; lean_object* v_one_142_; lean_object* v_m_143_; lean_object* v___x_144_; 
v_decode_141_ = lean_ctor_get(v_h_136_, 1);
lean_inc_ref(v_decode_141_);
lean_dec_ref(v_h_136_);
v_one_142_ = lean_unsigned_to_nat(1u);
v_m_143_ = lean_nat_sub(v_n_137_, v_one_142_);
v___x_144_ = lean_apply_1(v_decode_141_, v_m_143_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v___x_145_; 
v___x_145_ = lean_box(0);
return v___x_145_;
}
else
{
lean_object* v___x_146_; 
v___x_146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_146_, 0, v___x_144_);
return v___x_146_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg___lam__1___boxed(lean_object* v_h_147_, lean_object* v_n_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Option_encodable___redArg___lam__1(v_h_147_, v_n_148_);
lean_dec(v_n_148_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable___redArg(lean_object* v_h_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___f_152_; lean_object* v___x_153_; 
lean_inc_ref(v_h_150_);
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Option_encodable___redArg___lam__0), 2, 1);
lean_closure_set(v___f_151_, 0, v_h_150_);
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_Option_encodable___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_152_, 0, v_h_150_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v___f_151_);
lean_ctor_set(v___x_153_, 1, v___f_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_encodable(lean_object* v_00_u03b1_154_, lean_object* v_h_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_Option_encodable___redArg(v_h_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decode_u2082___redArg(lean_object* v_inst_157_, lean_object* v_n_158_){
_start:
{
lean_object* v_encode_159_; lean_object* v_decode_160_; lean_object* v___x_161_; 
v_encode_159_ = lean_ctor_get(v_inst_157_, 0);
lean_inc_ref(v_encode_159_);
v_decode_160_ = lean_ctor_get(v_inst_157_, 1);
lean_inc_ref(v_decode_160_);
lean_dec_ref(v_inst_157_);
lean_inc(v_n_158_);
v___x_161_ = lean_apply_1(v_decode_160_, v_n_158_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_dec_ref(v_encode_159_);
lean_dec(v_n_158_);
return v___x_161_;
}
else
{
lean_object* v_val_162_; lean_object* v___x_163_; uint8_t v___x_164_; 
v_val_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_val_162_);
v___x_163_ = lean_apply_1(v_encode_159_, v_val_162_);
v___x_164_ = lean_nat_dec_eq(v___x_163_, v_n_158_);
lean_dec(v_n_158_);
lean_dec(v___x_163_);
if (v___x_164_ == 0)
{
lean_object* v___x_165_; 
lean_dec_ref_known(v___x_161_, 1);
v___x_165_ = lean_box(0);
return v___x_165_;
}
else
{
return v___x_161_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decode_u2082(lean_object* v_00_u03b1_166_, lean_object* v_inst_167_, lean_object* v_n_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_Encodable_decode_u2082___redArg(v_inst_167_, v_n_168_);
return v___x_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableRangeEncode___redArg(lean_object* v_inst_170_, lean_object* v_x_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_Encodable_decode_u2082___redArg(v_inst_170_, v_x_171_);
if (lean_obj_tag(v___x_172_) == 0)
{
uint8_t v___x_173_; 
v___x_173_ = 0;
return v___x_173_;
}
else
{
uint8_t v___x_174_; 
lean_dec_ref_known(v___x_172_, 1);
v___x_174_ = 1;
return v___x_174_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableRangeEncode___redArg___boxed(lean_object* v_inst_175_, lean_object* v_x_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_mathlib_Encodable_decidableRangeEncode___redArg(v_inst_175_, v_x_176_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Encodable_decidableRangeEncode(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_x_181_){
_start:
{
uint8_t v___x_182_; 
v___x_182_ = lp_mathlib_Encodable_decidableRangeEncode___redArg(v_inst_180_, v_x_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decidableRangeEncode___boxed(lean_object* v_00_u03b1_183_, lean_object* v_inst_184_, lean_object* v_x_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib_Encodable_decidableRangeEncode(v_00_u03b1_183_, v_inst_184_, v_x_185_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg___lam__0(lean_object* v_inst_188_, lean_object* v_a_189_){
_start:
{
lean_object* v_encode_190_; lean_object* v___x_191_; 
v_encode_190_ = lean_ctor_get(v_inst_188_, 0);
lean_inc_ref(v_encode_190_);
lean_dec_ref(v_inst_188_);
v___x_191_ = lean_apply_1(v_encode_190_, v_a_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg___lam__1(lean_object* v_inst_192_, lean_object* v_n_193_){
_start:
{
lean_object* v___x_194_; lean_object* v_val_195_; 
v___x_194_ = lp_mathlib_Encodable_decode_u2082___redArg(v_inst_192_, v_n_193_);
v_val_195_ = lean_ctor_get(v___x_194_, 0);
lean_inc(v_val_195_);
lean_dec(v___x_194_);
return v_val_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode___redArg(lean_object* v_inst_196_){
_start:
{
lean_object* v___f_197_; lean_object* v___f_198_; lean_object* v___x_199_; 
lean_inc_ref(v_inst_196_);
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_equivRangeEncode___redArg___lam__0), 2, 1);
lean_closure_set(v___f_197_, 0, v_inst_196_);
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_equivRangeEncode___redArg___lam__1), 2, 1);
lean_closure_set(v___f_198_, 0, v_inst_196_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v___f_197_);
lean_ctor_set(v___x_199_, 1, v___f_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_equivRangeEncode(lean_object* v_00_u03b1_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___x_202_; 
v___x_202_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__0(lean_object* v_x_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lean_unsigned_to_nat(0u);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__0___boxed(lean_object* v_x_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Unique_encodable___redArg___lam__0(v_x_205_);
lean_dec(v_x_205_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__1(lean_object* v_inst_207_, lean_object* v_x_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_209_, 0, v_inst_207_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg___lam__1___boxed(lean_object* v_inst_210_, lean_object* v_x_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_Unique_encodable___redArg___lam__1(v_inst_210_, v_x_211_);
lean_dec(v_x_211_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable___redArg(lean_object* v_inst_214_){
_start:
{
lean_object* v___f_215_; lean_object* v___f_216_; lean_object* v___x_217_; 
v___f_215_ = ((lean_object*)(lp_mathlib_Unique_encodable___redArg___closed__0));
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_Unique_encodable___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_216_, 0, v_inst_214_);
v___x_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_217_, 0, v___f_215_);
lean_ctor_set(v___x_217_, 1, v___f_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Unique_encodable(lean_object* v_00_u03b1_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_Unique_encodable___redArg(v_inst_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSum___redArg(lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_x_223_){
_start:
{
if (lean_obj_tag(v_x_223_) == 0)
{
lean_object* v_val_224_; lean_object* v_encode_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
lean_dec_ref(v_inst_222_);
v_val_224_ = lean_ctor_get(v_x_223_, 0);
lean_inc(v_val_224_);
lean_dec_ref_known(v_x_223_, 1);
v_encode_225_ = lean_ctor_get(v_inst_221_, 0);
lean_inc_ref(v_encode_225_);
lean_dec_ref(v_inst_221_);
v___x_226_ = lean_unsigned_to_nat(2u);
v___x_227_ = lean_apply_1(v_encode_225_, v_val_224_);
v___x_228_ = lean_nat_mul(v___x_226_, v___x_227_);
lean_dec(v___x_227_);
return v___x_228_;
}
else
{
lean_object* v_val_229_; lean_object* v_encode_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
lean_dec_ref(v_inst_221_);
v_val_229_ = lean_ctor_get(v_x_223_, 0);
lean_inc(v_val_229_);
lean_dec_ref_known(v_x_223_, 1);
v_encode_230_ = lean_ctor_get(v_inst_222_, 0);
lean_inc_ref(v_encode_230_);
lean_dec_ref(v_inst_222_);
v___x_231_ = lean_unsigned_to_nat(2u);
v___x_232_ = lean_apply_1(v_encode_230_, v_val_229_);
v___x_233_ = lean_nat_mul(v___x_231_, v___x_232_);
lean_dec(v___x_232_);
v___x_234_ = lean_unsigned_to_nat(1u);
v___x_235_ = lean_nat_add(v___x_233_, v___x_234_);
lean_dec(v___x_233_);
return v___x_235_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSum(lean_object* v_00_u03b1_236_, lean_object* v_00_u03b2_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_x_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Encodable_encodeSum___redArg(v_inst_238_, v_inst_239_, v_x_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___redArg(lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_n_244_){
_start:
{
lean_object* v___x_245_; uint8_t v___x_246_; 
v___x_245_ = lean_unsigned_to_nat(0u);
v___x_246_ = l_Nat_testBit(v_n_244_, v___x_245_);
if (v___x_246_ == 0)
{
lean_object* v_decode_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
lean_dec_ref(v_inst_243_);
v_decode_247_ = lean_ctor_get(v_inst_242_, 1);
lean_inc_ref(v_decode_247_);
lean_dec_ref(v_inst_242_);
v___x_248_ = lean_unsigned_to_nat(1u);
v___x_249_ = lean_nat_shiftr(v_n_244_, v___x_248_);
v___x_250_ = lean_apply_1(v_decode_247_, v___x_249_);
if (lean_obj_tag(v___x_250_) == 0)
{
lean_object* v___x_251_; 
v___x_251_ = lean_box(0);
return v___x_251_;
}
else
{
lean_object* v_val_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_260_; 
v_val_252_ = lean_ctor_get(v___x_250_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_260_ == 0)
{
v___x_254_ = v___x_250_;
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_val_252_);
lean_dec(v___x_250_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; lean_object* v___x_258_; 
v___x_256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_256_, 0, v_val_252_);
if (v_isShared_255_ == 0)
{
lean_ctor_set(v___x_254_, 0, v___x_256_);
v___x_258_ = v___x_254_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v___x_256_);
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
else
{
lean_object* v_decode_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
lean_dec_ref(v_inst_242_);
v_decode_261_ = lean_ctor_get(v_inst_243_, 1);
lean_inc_ref(v_decode_261_);
lean_dec_ref(v_inst_243_);
v___x_262_ = lean_unsigned_to_nat(1u);
v___x_263_ = lean_nat_shiftr(v_n_244_, v___x_262_);
v___x_264_ = lean_apply_1(v_decode_261_, v___x_263_);
if (lean_obj_tag(v___x_264_) == 0)
{
lean_object* v___x_265_; 
v___x_265_ = lean_box(0);
return v___x_265_;
}
else
{
lean_object* v_val_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_274_; 
v_val_266_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_274_ == 0)
{
v___x_268_ = v___x_264_;
v_isShared_269_ = v_isSharedCheck_274_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_val_266_);
lean_dec(v___x_264_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_274_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_270_; lean_object* v___x_272_; 
v___x_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_270_, 0, v_val_266_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 0, v___x_270_);
v___x_272_ = v___x_268_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v___x_270_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___redArg___boxed(lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_n_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_Encodable_decodeSum___redArg(v_inst_275_, v_inst_276_, v_n_277_);
lean_dec(v_n_277_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_n_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_Encodable_decodeSum___redArg(v_inst_281_, v_inst_282_, v_n_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSum___boxed(lean_object* v_00_u03b1_285_, lean_object* v_00_u03b2_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_n_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_Encodable_decodeSum(v_00_u03b1_285_, v_00_u03b2_286_, v_inst_287_, v_inst_288_, v_n_289_);
lean_dec(v_n_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSum_match__1_splitter___redArg(lean_object* v_x_291_, lean_object* v_h__1_292_, lean_object* v_h__2_293_){
_start:
{
if (lean_obj_tag(v_x_291_) == 0)
{
lean_object* v_val_294_; lean_object* v___x_295_; 
lean_dec(v_h__2_293_);
v_val_294_ = lean_ctor_get(v_x_291_, 0);
lean_inc(v_val_294_);
lean_dec_ref_known(v_x_291_, 1);
v___x_295_ = lean_apply_1(v_h__1_292_, v_val_294_);
return v___x_295_;
}
else
{
lean_object* v_val_296_; lean_object* v___x_297_; 
lean_dec(v_h__1_292_);
v_val_296_ = lean_ctor_get(v_x_291_, 0);
lean_inc(v_val_296_);
lean_dec_ref_known(v_x_291_, 1);
v___x_297_ = lean_apply_1(v_h__2_293_, v_val_296_);
return v___x_297_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSum_match__1_splitter(lean_object* v_00_u03b1_298_, lean_object* v_00_u03b2_299_, lean_object* v_motive_300_, lean_object* v_x_301_, lean_object* v_h__1_302_, lean_object* v_h__2_303_){
_start:
{
if (lean_obj_tag(v_x_301_) == 0)
{
lean_object* v_val_304_; lean_object* v___x_305_; 
lean_dec(v_h__2_303_);
v_val_304_ = lean_ctor_get(v_x_301_, 0);
lean_inc(v_val_304_);
lean_dec_ref_known(v_x_301_, 1);
v___x_305_ = lean_apply_1(v_h__1_302_, v_val_304_);
return v___x_305_;
}
else
{
lean_object* v_val_306_; lean_object* v___x_307_; 
lean_dec(v_h__1_302_);
v_val_306_ = lean_ctor_get(v_x_301_, 0);
lean_inc(v_val_306_);
lean_dec_ref_known(v_x_301_, 1);
v___x_307_ = lean_apply_1(v_h__2_303_, v_val_306_);
return v___x_307_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_encodable___redArg(lean_object* v_inst_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
lean_inc_ref(v_inst_309_);
lean_inc_ref(v_inst_308_);
v___x_310_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodeSum), 5, 4);
lean_closure_set(v___x_310_, 0, lean_box(0));
lean_closure_set(v___x_310_, 1, lean_box(0));
lean_closure_set(v___x_310_, 2, v_inst_308_);
lean_closure_set(v___x_310_, 3, v_inst_309_);
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decodeSum___boxed), 5, 4);
lean_closure_set(v___x_311_, 0, lean_box(0));
lean_closure_set(v___x_311_, 1, lean_box(0));
lean_closure_set(v___x_311_, 2, v_inst_308_);
lean_closure_set(v___x_311_, 3, v_inst_309_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_310_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_encodable(lean_object* v_00_u03b1_313_, lean_object* v_00_u03b2_314_, lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_Sum_encodable___redArg(v_inst_315_, v_inst_316_);
return v___x_317_;
}
}
static lean_object* _init_lp_mathlib_Bool_encodable___closed__0(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = ((lean_object*)(lp_mathlib_PUnit_encodable));
v___x_319_ = lp_mathlib_Sum_encodable___redArg(v___x_318_, v___x_318_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Bool_encodable___closed__1(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_320_ = lp_mathlib_Equiv_boolEquivPUnitSumPUnit;
v___x_321_ = lean_obj_once(&lp_mathlib_Bool_encodable___closed__0, &lp_mathlib_Bool_encodable___closed__0_once, _init_lp_mathlib_Bool_encodable___closed__0);
v___x_322_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_321_, v___x_320_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib_Bool_encodable(void){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lean_obj_once(&lp_mathlib_Bool_encodable___closed__1, &lp_mathlib_Bool_encodable___closed__1_once, _init_lp_mathlib_Bool_encodable___closed__1);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___redArg(uint8_t v_x_324_, lean_object* v_x_325_, lean_object* v_h__1_326_, lean_object* v_h__2_327_){
_start:
{
if (v_x_324_ == 0)
{
lean_object* v___x_328_; 
lean_dec(v_h__2_327_);
v___x_328_ = lean_apply_1(v_h__1_326_, v_x_325_);
return v___x_328_;
}
else
{
lean_object* v___x_329_; lean_object* v___x_330_; 
lean_dec(v_h__1_326_);
v___x_329_ = lean_box(v_x_324_);
v___x_330_ = lean_apply_3(v_h__2_327_, v___x_329_, v_x_325_, lean_box(0));
return v___x_330_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___redArg___boxed(lean_object* v_x_331_, lean_object* v_x_332_, lean_object* v_h__1_333_, lean_object* v_h__2_334_){
_start:
{
uint8_t v_x_16__boxed_335_; lean_object* v_res_336_; 
v_x_16__boxed_335_ = lean_unbox(v_x_331_);
v_res_336_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___redArg(v_x_16__boxed_335_, v_x_332_, v_h__1_333_, v_h__2_334_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter(lean_object* v_motive_337_, uint8_t v_x_338_, lean_object* v_x_339_, lean_object* v_h__1_340_, lean_object* v_h__2_341_){
_start:
{
if (v_x_338_ == 0)
{
lean_object* v___x_342_; 
lean_dec(v_h__2_341_);
v___x_342_ = lean_apply_1(v_h__1_340_, v_x_339_);
return v___x_342_;
}
else
{
lean_object* v___x_343_; lean_object* v___x_344_; 
lean_dec(v_h__1_340_);
v___x_343_ = lean_box(v_x_338_);
v___x_344_ = lean_apply_3(v_h__2_341_, v___x_343_, v_x_339_, lean_box(0));
return v___x_344_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter___boxed(lean_object* v_motive_345_, lean_object* v_x_346_, lean_object* v_x_347_, lean_object* v_h__1_348_, lean_object* v_h__2_349_){
_start:
{
uint8_t v_x_28__boxed_350_; lean_object* v_res_351_; 
v_x_28__boxed_350_ = lean_unbox(v_x_346_);
v_res_351_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decodeSum_match__1_splitter(v_motive_345_, v_x_28__boxed_350_, v_x_347_, v_h__1_348_, v_h__2_349_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSigma___redArg(lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_x_354_){
_start:
{
lean_object* v_fst_355_; lean_object* v_snd_356_; lean_object* v_encode_357_; lean_object* v___x_358_; lean_object* v_encode_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
v_fst_355_ = lean_ctor_get(v_x_354_, 0);
lean_inc_n(v_fst_355_, 2);
v_snd_356_ = lean_ctor_get(v_x_354_, 1);
lean_inc(v_snd_356_);
lean_dec_ref(v_x_354_);
v_encode_357_ = lean_ctor_get(v_inst_352_, 0);
lean_inc_ref(v_encode_357_);
lean_dec_ref(v_inst_352_);
v___x_358_ = lean_apply_1(v_inst_353_, v_fst_355_);
v_encode_359_ = lean_ctor_get(v___x_358_, 0);
lean_inc_ref(v_encode_359_);
lean_dec_ref(v___x_358_);
v___x_360_ = lean_apply_1(v_encode_357_, v_fst_355_);
v___x_361_ = lean_apply_1(v_encode_359_, v_snd_356_);
v___x_362_ = lp_mathlib_Nat_pair(v___x_360_, v___x_361_);
lean_dec(v___x_361_);
lean_dec(v___x_360_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSigma(lean_object* v_00_u03b1_363_, lean_object* v_00_u03b3_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_x_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_mathlib_Encodable_encodeSigma___redArg(v_inst_365_, v_inst_366_, v_x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___redArg(lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_n_371_){
_start:
{
lean_object* v___x_372_; lean_object* v_fst_373_; lean_object* v_snd_374_; lean_object* v_decode_375_; lean_object* v___x_376_; 
v___x_372_ = lp_mathlib_Nat_unpair(v_n_371_);
v_fst_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc(v_fst_373_);
v_snd_374_ = lean_ctor_get(v___x_372_, 1);
lean_inc(v_snd_374_);
lean_dec_ref(v___x_372_);
v_decode_375_ = lean_ctor_get(v_inst_369_, 1);
lean_inc_ref(v_decode_375_);
lean_dec_ref(v_inst_369_);
v___x_376_ = lean_apply_1(v_decode_375_, v_fst_373_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v___x_377_; 
lean_dec(v_snd_374_);
lean_dec_ref(v_inst_370_);
v___x_377_ = lean_box(0);
return v___x_377_;
}
else
{
lean_object* v_val_378_; lean_object* v___x_379_; lean_object* v_decode_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_397_; 
v_val_378_ = lean_ctor_get(v___x_376_, 0);
lean_inc_n(v_val_378_, 2);
lean_dec_ref_known(v___x_376_, 1);
v___x_379_ = lean_apply_1(v_inst_370_, v_val_378_);
v_decode_380_ = lean_ctor_get(v___x_379_, 1);
v_isSharedCheck_397_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_397_ == 0)
{
lean_object* v_unused_398_; 
v_unused_398_ = lean_ctor_get(v___x_379_, 0);
lean_dec(v_unused_398_);
v___x_382_ = v___x_379_;
v_isShared_383_ = v_isSharedCheck_397_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_decode_380_);
lean_dec(v___x_379_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_397_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_384_; 
v___x_384_ = lean_apply_1(v_decode_380_, v_snd_374_);
if (lean_obj_tag(v___x_384_) == 0)
{
lean_object* v___x_385_; 
lean_del_object(v___x_382_);
lean_dec(v_val_378_);
v___x_385_ = lean_box(0);
return v___x_385_;
}
else
{
lean_object* v_val_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_396_; 
v_val_386_ = lean_ctor_get(v___x_384_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_396_ == 0)
{
v___x_388_ = v___x_384_;
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_val_386_);
lean_dec(v___x_384_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_391_; 
if (v_isShared_383_ == 0)
{
lean_ctor_set(v___x_382_, 1, v_val_386_);
lean_ctor_set(v___x_382_, 0, v_val_378_);
v___x_391_ = v___x_382_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v_val_378_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v_val_386_);
v___x_391_ = v_reuseFailAlloc_395_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
lean_object* v___x_393_; 
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_391_);
v___x_393_ = v___x_388_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v___x_391_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___redArg___boxed(lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_n_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_mathlib_Encodable_decodeSigma___redArg(v_inst_399_, v_inst_400_, v_n_401_);
lean_dec(v_n_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b3_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_n_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_Encodable_decodeSigma___redArg(v_inst_405_, v_inst_406_, v_n_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSigma___boxed(lean_object* v_00_u03b1_409_, lean_object* v_00_u03b3_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_n_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_Encodable_decodeSigma(v_00_u03b1_409_, v_00_u03b3_410_, v_inst_411_, v_inst_412_, v_n_413_);
lean_dec(v_n_413_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSigma_match__1_splitter___redArg(lean_object* v_x_415_, lean_object* v_h__1_416_){
_start:
{
lean_object* v_fst_417_; lean_object* v_snd_418_; lean_object* v___x_419_; 
v_fst_417_ = lean_ctor_get(v_x_415_, 0);
lean_inc(v_fst_417_);
v_snd_418_ = lean_ctor_get(v_x_415_, 1);
lean_inc(v_snd_418_);
lean_dec_ref(v_x_415_);
v___x_419_ = lean_apply_2(v_h__1_416_, v_fst_417_, v_snd_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSigma_match__1_splitter(lean_object* v_00_u03b1_420_, lean_object* v_00_u03b3_421_, lean_object* v_motive_422_, lean_object* v_x_423_, lean_object* v_h__1_424_){
_start:
{
lean_object* v_fst_425_; lean_object* v_snd_426_; lean_object* v___x_427_; 
v_fst_425_ = lean_ctor_get(v_x_423_, 0);
lean_inc(v_fst_425_);
v_snd_426_ = lean_ctor_get(v_x_423_, 1);
lean_inc(v_snd_426_);
lean_dec_ref(v_x_423_);
v___x_427_ = lean_apply_2(v_h__1_424_, v_fst_425_, v_snd_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_encodable___redArg(lean_object* v_inst_428_, lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; 
lean_inc_ref(v_inst_429_);
lean_inc_ref(v_inst_428_);
v___x_430_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodeSigma), 5, 4);
lean_closure_set(v___x_430_, 0, lean_box(0));
lean_closure_set(v___x_430_, 1, lean_box(0));
lean_closure_set(v___x_430_, 2, v_inst_428_);
lean_closure_set(v___x_430_, 3, v_inst_429_);
v___x_431_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decodeSigma___boxed), 5, 4);
lean_closure_set(v___x_431_, 0, lean_box(0));
lean_closure_set(v___x_431_, 1, lean_box(0));
lean_closure_set(v___x_431_, 2, v_inst_428_);
lean_closure_set(v___x_431_, 3, v_inst_429_);
v___x_432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_430_);
lean_ctor_set(v___x_432_, 1, v___x_431_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_encodable(lean_object* v_00_u03b1_433_, lean_object* v_00_u03b3_434_, lean_object* v_inst_435_, lean_object* v_inst_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lp_mathlib_Sigma_encodable___redArg(v_inst_435_, v_inst_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___lam__0(lean_object* v_inst_438_, lean_object* v_a_439_){
_start:
{
lean_inc_ref(v_inst_438_);
return v_inst_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg___lam__0___boxed(lean_object* v_inst_440_, lean_object* v_a_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_Encodable_Prod_encodable___redArg___lam__0(v_inst_440_, v_a_441_);
lean_dec(v_a_441_);
lean_dec_ref(v_inst_440_);
return v_res_442_;
}
}
static lean_object* _init_lp_mathlib_Encodable_Prod_encodable___redArg___closed__0(void){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_mathlib_Equiv_sigmaEquivProd(lean_box(0), lean_box(0));
return v___x_443_;
}
}
static lean_object* _init_lp_mathlib_Encodable_Prod_encodable___redArg___closed__1(void){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_444_ = lean_obj_once(&lp_mathlib_Encodable_Prod_encodable___redArg___closed__0, &lp_mathlib_Encodable_Prod_encodable___redArg___closed__0_once, _init_lp_mathlib_Encodable_Prod_encodable___redArg___closed__0);
v___x_445_ = lp_mathlib_Equiv_symm___redArg(v___x_444_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable___redArg(lean_object* v_inst_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v___f_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_Prod_encodable___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_448_, 0, v_inst_447_);
v___x_449_ = lp_mathlib_Sigma_encodable___redArg(v_inst_446_, v___f_448_);
v___x_450_ = lean_obj_once(&lp_mathlib_Encodable_Prod_encodable___redArg___closed__1, &lp_mathlib_Encodable_Prod_encodable___redArg___closed__1_once, _init_lp_mathlib_Encodable_Prod_encodable___redArg___closed__1);
v___x_451_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_449_, v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_Prod_encodable(lean_object* v_00_u03b1_452_, lean_object* v_00_u03b2_453_, lean_object* v_inst_454_, lean_object* v_inst_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_Encodable_Prod_encodable___redArg(v_inst_454_, v_inst_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSubtype___redArg(lean_object* v_encA_457_, lean_object* v_x_458_){
_start:
{
lean_object* v_encode_459_; lean_object* v___x_460_; 
v_encode_459_ = lean_ctor_get(v_encA_457_, 0);
lean_inc_ref(v_encode_459_);
lean_dec_ref(v_encA_457_);
v___x_460_ = lean_apply_1(v_encode_459_, v_x_458_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encodeSubtype(lean_object* v_00_u03b1_461_, lean_object* v_P_462_, lean_object* v_encA_463_, lean_object* v_x_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_Encodable_encodeSubtype___redArg(v_encA_463_, v_x_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSubtype___redArg(lean_object* v_encA_466_, lean_object* v_decP_467_, lean_object* v_v_468_){
_start:
{
lean_object* v_decode_469_; lean_object* v___x_470_; 
v_decode_469_ = lean_ctor_get(v_encA_466_, 1);
lean_inc_ref(v_decode_469_);
lean_dec_ref(v_encA_466_);
v___x_470_ = lean_apply_1(v_decode_469_, v_v_468_);
if (lean_obj_tag(v___x_470_) == 0)
{
lean_object* v___x_471_; 
lean_dec_ref(v_decP_467_);
v___x_471_ = lean_box(0);
return v___x_471_;
}
else
{
lean_object* v_val_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_482_; 
v_val_472_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_482_ == 0)
{
v___x_474_ = v___x_470_;
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_val_472_);
lean_dec(v___x_470_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_476_; uint8_t v___x_477_; 
lean_inc(v_val_472_);
v___x_476_ = lean_apply_1(v_decP_467_, v_val_472_);
v___x_477_ = lean_unbox(v___x_476_);
if (v___x_477_ == 0)
{
lean_object* v___x_478_; 
lean_del_object(v___x_474_);
lean_dec(v_val_472_);
v___x_478_ = lean_box(0);
return v___x_478_;
}
else
{
lean_object* v___x_480_; 
if (v_isShared_475_ == 0)
{
v___x_480_ = v___x_474_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_val_472_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_decodeSubtype(lean_object* v_00_u03b1_483_, lean_object* v_P_484_, lean_object* v_encA_485_, lean_object* v_decP_486_, lean_object* v_v_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_mathlib_Encodable_decodeSubtype___redArg(v_encA_485_, v_decP_486_, v_v_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSubtype_match__1_splitter___redArg(lean_object* v_x_489_, lean_object* v_h__1_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lean_apply_2(v_h__1_490_, v_x_489_, lean_box(0));
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_encodeSubtype_match__1_splitter(lean_object* v_00_u03b1_492_, lean_object* v_P_493_, lean_object* v_motive_494_, lean_object* v_x_495_, lean_object* v_h__1_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lean_apply_2(v_h__1_496_, v_x_495_, lean_box(0));
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_encodable___redArg(lean_object* v_encA_498_, lean_object* v_decP_499_){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; 
lean_inc_ref(v_encA_498_);
v___x_500_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_encodeSubtype), 4, 3);
lean_closure_set(v___x_500_, 0, lean_box(0));
lean_closure_set(v___x_500_, 1, lean_box(0));
lean_closure_set(v___x_500_, 2, v_encA_498_);
v___x_501_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decodeSubtype), 5, 4);
lean_closure_set(v___x_501_, 0, lean_box(0));
lean_closure_set(v___x_501_, 1, lean_box(0));
lean_closure_set(v___x_501_, 2, v_encA_498_);
lean_closure_set(v___x_501_, 3, v_decP_499_);
v___x_502_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_encodable(lean_object* v_00_u03b1_503_, lean_object* v_P_504_, lean_object* v_encA_505_, lean_object* v_decP_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lp_mathlib_Subtype_encodable___redArg(v_encA_505_, v_decP_506_);
return v___x_507_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fin_encodable___lam__0(lean_object* v_n_508_, lean_object* v_a_509_){
_start:
{
uint8_t v___x_510_; 
v___x_510_ = lean_nat_dec_lt(v_a_509_, v_n_508_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_encodable___lam__0___boxed(lean_object* v_n_511_, lean_object* v_a_512_){
_start:
{
uint8_t v_res_513_; lean_object* v_r_514_; 
v_res_513_ = lp_mathlib_Fin_encodable___lam__0(v_n_511_, v_a_512_);
lean_dec(v_a_512_);
lean_dec(v_n_511_);
v_r_514_ = lean_box(v_res_513_);
return v_r_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_encodable(lean_object* v_n_515_){
_start:
{
lean_object* v___f_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
lean_inc(v_n_515_);
v___f_516_ = lean_alloc_closure((void*)(lp_mathlib_Fin_encodable___lam__0___boxed), 2, 1);
lean_closure_set(v___f_516_, 0, v_n_515_);
v___x_517_ = ((lean_object*)(lp_mathlib_Nat_encodable));
v___x_518_ = lp_mathlib_Subtype_encodable___redArg(v___x_517_, v___f_516_);
v___x_519_ = lp_mathlib_Fin_equivSubtype(v_n_515_);
lean_dec(v_n_515_);
v___x_520_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_518_, v___x_519_);
return v___x_520_;
}
}
static lean_object* _init_lp_mathlib_Int_encodable___closed__0(void){
_start:
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; 
v___x_521_ = lp_mathlib_Equiv_intEquivNat;
v___x_522_ = ((lean_object*)(lp_mathlib_Nat_encodable));
v___x_523_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_522_, v___x_521_);
return v___x_523_;
}
}
static lean_object* _init_lp_mathlib_Int_encodable(void){
_start:
{
lean_object* v___x_524_; 
v___x_524_ = lean_obj_once(&lp_mathlib_Int_encodable___closed__0, &lp_mathlib_Int_encodable___closed__0_once, _init_lp_mathlib_Int_encodable___closed__0);
return v___x_524_;
}
}
static lean_object* _init_lp_mathlib_PNat_encodable___closed__0(void){
_start:
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_525_ = lp_mathlib_Equiv_pnatEquivNat;
v___x_526_ = ((lean_object*)(lp_mathlib_Nat_encodable));
v___x_527_ = lp_mathlib_Encodable_ofEquiv___redArg(v___x_526_, v___x_525_);
return v___x_527_;
}
}
static lean_object* _init_lp_mathlib_PNat_encodable(void){
_start:
{
lean_object* v___x_528_; 
v___x_528_ = lean_obj_once(&lp_mathlib_PNat_encodable___closed__0, &lp_mathlib_PNat_encodable___closed__0_once, _init_lp_mathlib_PNat_encodable___closed__0);
return v___x_528_;
}
}
static lean_object* _init_lp_mathlib_ULift_encodable___redArg___closed__0(void){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_encodable___redArg(lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_531_ = lean_obj_once(&lp_mathlib_ULift_encodable___redArg___closed__0, &lp_mathlib_ULift_encodable___redArg___closed__0_once, _init_lp_mathlib_ULift_encodable___redArg___closed__0);
v___x_532_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_530_, v___x_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_encodable(lean_object* v_00_u03b1_533_, lean_object* v_inst_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = lp_mathlib_ULift_encodable___redArg(v_inst_534_);
return v___x_535_;
}
}
static lean_object* _init_lp_mathlib_PLift_encodable___redArg___closed__0(void){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_encodable___redArg(lean_object* v_inst_537_){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_538_ = lean_obj_once(&lp_mathlib_PLift_encodable___redArg___closed__0, &lp_mathlib_PLift_encodable___redArg___closed__0_once, _init_lp_mathlib_PLift_encodable___redArg___closed__0);
v___x_539_ = lp_mathlib_Encodable_ofEquiv___redArg(v_inst_537_, v___x_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PLift_encodable(lean_object* v_00_u03b1_540_, lean_object* v_inst_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_PLift_encodable___redArg(v_inst_541_);
return v___x_542_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqULower___redArg(lean_object* v_inst_543_, lean_object* v_a_544_, lean_object* v_b_545_){
_start:
{
lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; uint8_t v___x_549_; 
v___x_546_ = ((lean_object*)(lp_mathlib_Nat_encodable));
v___x_547_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableRangeEncode___boxed), 3, 2);
lean_closure_set(v___x_547_, 0, lean_box(0));
lean_closure_set(v___x_547_, 1, v_inst_543_);
v___x_548_ = lp_mathlib_Subtype_encodable___redArg(v___x_546_, v___x_547_);
v___x_549_ = lp_mathlib_Encodable_decidableEqOfEncodable___redArg(v___x_548_, v_a_544_, v_b_545_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqULower___redArg___boxed(lean_object* v_inst_550_, lean_object* v_a_551_, lean_object* v_b_552_){
_start:
{
uint8_t v_res_553_; lean_object* v_r_554_; 
v_res_553_ = lp_mathlib_instDecidableEqULower___redArg(v_inst_550_, v_a_551_, v_b_552_);
v_r_554_ = lean_box(v_res_553_);
return v_r_554_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqULower(lean_object* v_00_u03b1_555_, lean_object* v_inst_556_, lean_object* v_a_557_, lean_object* v_b_558_){
_start:
{
uint8_t v___x_559_; 
v___x_559_ = lp_mathlib_instDecidableEqULower___redArg(v_inst_556_, v_a_557_, v_b_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqULower___boxed(lean_object* v_00_u03b1_560_, lean_object* v_inst_561_, lean_object* v_a_562_, lean_object* v_b_563_){
_start:
{
uint8_t v_res_564_; lean_object* v_r_565_; 
v_res_564_ = lp_mathlib_instDecidableEqULower(v_00_u03b1_560_, v_inst_561_, v_a_562_, v_b_563_);
v_r_565_ = lean_box(v_res_564_);
return v_r_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableULower___redArg(lean_object* v_inst_566_){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_567_ = ((lean_object*)(lp_mathlib_Nat_encodable));
v___x_568_ = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableRangeEncode___boxed), 3, 2);
lean_closure_set(v___x_568_, 0, lean_box(0));
lean_closure_set(v___x_568_, 1, v_inst_566_);
v___x_569_ = lp_mathlib_Subtype_encodable___redArg(v___x_567_, v___x_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableULower(lean_object* v_00_u03b1_570_, lean_object* v_inst_571_){
_start:
{
lean_object* v___x_572_; 
v___x_572_ = lp_mathlib_instEncodableULower___redArg(v_inst_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_equiv___redArg(lean_object* v_inst_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_equiv(lean_object* v_00_u03b1_575_, lean_object* v_inst_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_down___redArg(lean_object* v_inst_578_, lean_object* v_a_579_){
_start:
{
lean_object* v___x_580_; lean_object* v_toFun_581_; lean_object* v___x_582_; 
v___x_580_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_578_);
v_toFun_581_ = lean_ctor_get(v___x_580_, 0);
lean_inc(v_toFun_581_);
lean_dec_ref(v___x_580_);
v___x_582_ = lean_apply_1(v_toFun_581_, v_a_579_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_down(lean_object* v_00_u03b1_583_, lean_object* v_inst_584_, lean_object* v_a_585_){
_start:
{
lean_object* v___x_586_; 
v___x_586_ = lp_mathlib_ULower_down___redArg(v_inst_584_, v_a_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_instInhabited___redArg(lean_object* v_inst_587_, lean_object* v_inst_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_mathlib_ULower_down___redArg(v_inst_587_, v_inst_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_instInhabited(lean_object* v_00_u03b1_590_, lean_object* v_inst_591_, lean_object* v_inst_592_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lp_mathlib_ULower_down___redArg(v_inst_591_, v_inst_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_up___redArg(lean_object* v_inst_594_, lean_object* v_a_595_){
_start:
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v_toFun_598_; lean_object* v___x_599_; 
v___x_596_ = lp_mathlib_Encodable_equivRangeEncode___redArg(v_inst_594_);
v___x_597_ = lp_mathlib_Equiv_symm___redArg(v___x_596_);
v_toFun_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_toFun_598_);
lean_dec_ref(v___x_597_);
v___x_599_ = lean_apply_1(v_toFun_598_, v_a_595_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULower_up(lean_object* v_00_u03b1_600_, lean_object* v_inst_601_, lean_object* v_a_602_){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = lp_mathlib_ULower_up___redArg(v_inst_601_, v_a_602_);
return v___x_603_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___redArg(lean_object* v_inst_604_, lean_object* v_a_605_){
_start:
{
lean_object* v___x_606_; uint8_t v___x_607_; 
v___x_606_ = lean_apply_1(v_inst_604_, v_a_605_);
v___x_607_ = lean_unbox(v___x_606_);
return v___x_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___redArg___boxed(lean_object* v_inst_608_, lean_object* v_a_609_){
_start:
{
uint8_t v_res_610_; lean_object* v_r_611_; 
v_res_610_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___redArg(v_inst_608_, v_a_609_);
v_r_611_ = lean_box(v_res_610_);
return v_r_611_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1(lean_object* v_00_u03b1_612_, lean_object* v_p_613_, lean_object* v_inst_614_, lean_object* v_a_615_){
_start:
{
lean_object* v___x_616_; uint8_t v___x_617_; 
v___x_616_ = lean_apply_1(v_inst_614_, v_a_615_);
v___x_617_ = lean_unbox(v___x_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1___boxed(lean_object* v_00_u03b1_618_, lean_object* v_p_619_, lean_object* v_inst_620_, lean_object* v_a_621_){
_start:
{
uint8_t v_res_622_; lean_object* v_r_623_; 
v_res_622_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___aux__1(v_00_u03b1_618_, v_p_619_, v_inst_620_, v_a_621_);
v_r_623_ = lean_box(v_res_622_);
return v_r_623_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg(lean_object* v_inst_624_, lean_object* v_x_625_){
_start:
{
if (lean_obj_tag(v_x_625_) == 0)
{
uint8_t v___x_626_; 
lean_dec_ref(v_inst_624_);
v___x_626_ = 0;
return v___x_626_;
}
else
{
lean_object* v_val_627_; lean_object* v___x_628_; uint8_t v___x_629_; 
v_val_627_ = lean_ctor_get(v_x_625_, 0);
lean_inc(v_val_627_);
lean_dec_ref_known(v_x_625_, 1);
v___x_628_ = lean_apply_1(v_inst_624_, v_val_627_);
v___x_629_ = lean_unbox(v___x_628_);
return v___x_629_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg___boxed(lean_object* v_inst_630_, lean_object* v_x_631_){
_start:
{
uint8_t v_res_632_; lean_object* v_r_633_; 
v_res_632_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg(v_inst_630_, v_x_631_);
v_r_633_ = lean_box(v_res_632_);
return v_r_633_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good(lean_object* v_00_u03b1_634_, lean_object* v_p_635_, lean_object* v_inst_636_, lean_object* v_x_637_){
_start:
{
uint8_t v___x_638_; 
v___x_638_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg(v_inst_636_, v_x_637_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___boxed(lean_object* v_00_u03b1_639_, lean_object* v_p_640_, lean_object* v_inst_641_, lean_object* v_x_642_){
_start:
{
uint8_t v_res_643_; lean_object* v_r_644_; 
v_res_643_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good(v_00_u03b1_639_, v_p_640_, v_inst_641_, v_x_642_);
v_r_644_ = lean_box(v_res_643_);
return v_r_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_good_match__1_splitter___redArg(lean_object* v_x_645_, lean_object* v_h__1_646_, lean_object* v_h__2_647_){
_start:
{
if (lean_obj_tag(v_x_645_) == 0)
{
lean_object* v___x_648_; lean_object* v___x_649_; 
lean_dec(v_h__1_646_);
v___x_648_ = lean_box(0);
v___x_649_ = lean_apply_1(v_h__2_647_, v___x_648_);
return v___x_649_;
}
else
{
lean_object* v_val_650_; lean_object* v___x_651_; 
lean_dec(v_h__2_647_);
v_val_650_ = lean_ctor_get(v_x_645_, 0);
lean_inc(v_val_650_);
lean_dec_ref_known(v_x_645_, 1);
v___x_651_ = lean_apply_1(v_h__1_646_, v_val_650_);
return v___x_651_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_good_match__1_splitter(lean_object* v_00_u03b1_652_, lean_object* v_motive_653_, lean_object* v_x_654_, lean_object* v_h__1_655_, lean_object* v_h__2_656_){
_start:
{
if (lean_obj_tag(v_x_654_) == 0)
{
lean_object* v___x_657_; lean_object* v___x_658_; 
lean_dec(v_h__1_655_);
v___x_657_ = lean_box(0);
v___x_658_ = lean_apply_1(v_h__2_656_, v___x_657_);
return v___x_658_;
}
else
{
lean_object* v_val_659_; lean_object* v___x_660_; 
lean_dec(v_h__2_656_);
v_val_659_ = lean_ctor_get(v_x_654_, 0);
lean_inc(v_val_659_);
lean_dec_ref_known(v_x_654_, 1);
v___x_660_ = lean_apply_1(v_h__1_655_, v_val_659_);
return v___x_660_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0(lean_object* v_decode_661_, lean_object* v_inst_662_, lean_object* v_a_663_){
_start:
{
lean_object* v___x_664_; uint8_t v___x_665_; 
v___x_664_ = lean_apply_1(v_decode_661_, v_a_663_);
v___x_665_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_decidable__good___redArg(v_inst_662_, v___x_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0___boxed(lean_object* v_decode_666_, lean_object* v_inst_667_, lean_object* v_a_668_){
_start:
{
uint8_t v_res_669_; lean_object* v_r_670_; 
v_res_669_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0(v_decode_666_, v_inst_667_, v_a_668_);
v_r_670_ = lean_box(v_res_669_);
return v_r_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(lean_object* v_inst_671_, lean_object* v_inst_672_){
_start:
{
lean_object* v_decode_673_; lean_object* v___f_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v_val_677_; 
v_decode_673_ = lean_ctor_get(v_inst_671_, 1);
lean_inc_ref_n(v_decode_673_, 2);
lean_dec_ref(v_inst_671_);
v___f_674_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_674_, 0, v_decode_673_);
lean_closure_set(v___f_674_, 1, v_inst_672_);
v___x_675_ = lp_mathlib_Nat_findX___redArg(v___f_674_);
v___x_676_ = lean_apply_1(v_decode_673_, v___x_675_);
v_val_677_ = lean_ctor_get(v___x_676_, 0);
lean_inc(v_val_677_);
lean_dec(v___x_676_);
return v_val_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX(lean_object* v_00_u03b1_678_, lean_object* v_p_679_, lean_object* v_inst_680_, lean_object* v_inst_681_, lean_object* v_h_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(v_inst_680_, v_inst_681_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_choose___redArg(lean_object* v_inst_684_, lean_object* v_inst_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(v_inst_684_, v_inst_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_choose(lean_object* v_00_u03b1_687_, lean_object* v_p_688_, lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_h_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(v_inst_689_, v_inst_690_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___redArg(lean_object* v_inst_693_){
_start:
{
lean_object* v_encode_694_; 
v_encode_694_ = lean_ctor_get(v_inst_693_, 0);
lean_inc_ref(v_encode_694_);
return v_encode_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___redArg___boxed(lean_object* v_inst_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_mathlib_Encodable_encode_x27___redArg(v_inst_695_);
lean_dec_ref(v_inst_695_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27(lean_object* v_00_u03b1_697_, lean_object* v_inst_698_){
_start:
{
lean_object* v_encode_699_; 
v_encode_699_ = lean_ctor_get(v_inst_698_, 0);
lean_inc_ref(v_encode_699_);
return v_encode_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Encodable_encode_x27___boxed(lean_object* v_00_u03b1_700_, lean_object* v_inst_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_mathlib_Encodable_encode_x27(v_00_u03b1_700_, v_inst_701_);
lean_dec_ref(v_inst_701_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___redArg(lean_object* v_x_703_, lean_object* v_h__1_704_, lean_object* v_h__2_705_){
_start:
{
lean_object* v_zero_706_; uint8_t v_isZero_707_; 
v_zero_706_ = lean_unsigned_to_nat(0u);
v_isZero_707_ = lean_nat_dec_eq(v_x_703_, v_zero_706_);
if (v_isZero_707_ == 1)
{
lean_object* v___x_708_; lean_object* v___x_709_; 
lean_dec(v_h__2_705_);
v___x_708_ = lean_box(0);
v___x_709_ = lean_apply_1(v_h__1_704_, v___x_708_);
return v___x_709_;
}
else
{
lean_object* v_one_710_; lean_object* v_n_711_; lean_object* v___x_712_; 
lean_dec(v_h__1_704_);
v_one_710_ = lean_unsigned_to_nat(1u);
v_n_711_ = lean_nat_sub(v_x_703_, v_one_710_);
v___x_712_ = lean_apply_1(v_h__2_705_, v_n_711_);
return v___x_712_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___redArg___boxed(lean_object* v_x_713_, lean_object* v_h__1_714_, lean_object* v_h__2_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___redArg(v_x_713_, v_h__1_714_, v_h__2_715_);
lean_dec(v_x_713_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter(lean_object* v_motive_717_, lean_object* v_x_718_, lean_object* v_h__1_719_, lean_object* v_h__2_720_){
_start:
{
lean_object* v_zero_721_; uint8_t v_isZero_722_; 
v_zero_721_ = lean_unsigned_to_nat(0u);
v_isZero_722_ = lean_nat_dec_eq(v_x_718_, v_zero_721_);
if (v_isZero_722_ == 1)
{
lean_object* v___x_723_; lean_object* v___x_724_; 
lean_dec(v_h__2_720_);
v___x_723_ = lean_box(0);
v___x_724_ = lean_apply_1(v_h__1_719_, v___x_723_);
return v___x_724_;
}
else
{
lean_object* v_one_725_; lean_object* v_n_726_; lean_object* v___x_727_; 
lean_dec(v_h__1_719_);
v_one_725_ = lean_unsigned_to_nat(1u);
v_n_726_ = lean_nat_sub(v_x_718_, v_one_725_);
v___x_727_ = lean_apply_1(v_h__2_720_, v_n_726_);
return v___x_727_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter___boxed(lean_object* v_motive_728_, lean_object* v_x_729_, lean_object* v_h__1_730_, lean_object* v_h__2_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__3_splitter(v_motive_728_, v_x_729_, v_h__1_730_, v_h__2_731_);
lean_dec(v_x_729_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__1_splitter___redArg(lean_object* v_x_733_, lean_object* v_h__1_734_, lean_object* v_h__2_735_){
_start:
{
if (lean_obj_tag(v_x_733_) == 0)
{
lean_object* v___x_736_; lean_object* v___x_737_; 
lean_dec(v_h__2_735_);
v___x_736_ = lean_box(0);
v___x_737_ = lean_apply_1(v_h__1_734_, v___x_736_);
return v___x_737_;
}
else
{
lean_object* v_val_738_; lean_object* v___x_739_; 
lean_dec(v_h__1_734_);
v_val_738_ = lean_ctor_get(v_x_733_, 0);
lean_inc(v_val_738_);
lean_dec_ref_known(v_x_733_, 1);
v___x_739_ = lean_apply_1(v_h__2_735_, v_val_738_);
return v___x_739_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Directed_sequence_match__1_splitter(lean_object* v_00_u03b1_740_, lean_object* v_motive_741_, lean_object* v_x_742_, lean_object* v_h__1_743_, lean_object* v_h__2_744_){
_start:
{
if (lean_obj_tag(v_x_742_) == 0)
{
lean_object* v___x_745_; lean_object* v___x_746_; 
lean_dec(v_h__2_744_);
v___x_745_ = lean_box(0);
v___x_746_ = lean_apply_1(v_h__1_743_, v___x_745_);
return v___x_746_;
}
else
{
lean_object* v_val_747_; lean_object* v___x_748_; 
lean_dec(v_h__1_743_);
v_val_747_ = lean_ctor_get(v_x_742_, 0);
lean_inc(v_val_747_);
lean_dec_ref_known(v_x_742_, 1);
v___x_748_ = lean_apply_1(v_h__2_744_, v_val_747_);
return v___x_748_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Quotient_rep___redArg___lam__0(lean_object* v_inst_749_, lean_object* v_q_750_, lean_object* v_a_751_){
_start:
{
lean_object* v___x_752_; uint8_t v___x_753_; 
v___x_752_ = lean_apply_2(v_inst_749_, v_a_751_, v_q_750_);
v___x_753_ = lean_unbox(v___x_752_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep___redArg___lam__0___boxed(lean_object* v_inst_754_, lean_object* v_q_755_, lean_object* v_a_756_){
_start:
{
uint8_t v_res_757_; lean_object* v_r_758_; 
v_res_757_ = lp_mathlib_Quotient_rep___redArg___lam__0(v_inst_754_, v_q_755_, v_a_756_);
v_r_758_ = lean_box(v_res_757_);
return v_r_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep___redArg(lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_q_761_){
_start:
{
lean_object* v___f_762_; lean_object* v___x_763_; 
v___f_762_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_rep___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_762_, 0, v_inst_759_);
lean_closure_set(v___f_762_, 1, v_q_761_);
v___x_763_ = lp_mathlib___private_Mathlib_Logic_Encodable_Basic_0__Encodable_chooseX___redArg(v_inst_760_, v___f_762_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_rep(lean_object* v_00_u03b1_764_, lean_object* v_s_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_q_768_){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lp_mathlib_Quotient_rep___redArg(v_inst_766_, v_inst_767_, v_q_768_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg___lam__0(lean_object* v_inst_770_, lean_object* v_inst_771_, lean_object* v_q_772_){
_start:
{
lean_object* v_encode_773_; lean_object* v___x_774_; lean_object* v___x_775_; 
v_encode_773_ = lean_ctor_get(v_inst_770_, 0);
lean_inc_ref(v_encode_773_);
v___x_774_ = lp_mathlib_Quotient_rep___redArg(v_inst_771_, v_inst_770_, v_q_772_);
v___x_775_ = lean_apply_1(v_encode_773_, v___x_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg___lam__1(lean_object* v_inst_776_, lean_object* v_n_777_){
_start:
{
lean_object* v_decode_778_; lean_object* v___x_779_; 
v_decode_778_ = lean_ctor_get(v_inst_776_, 1);
lean_inc_ref(v_decode_778_);
lean_dec_ref(v_inst_776_);
v___x_779_ = lean_apply_1(v_decode_778_, v_n_777_);
if (lean_obj_tag(v___x_779_) == 0)
{
lean_object* v___x_780_; 
v___x_780_ = lean_box(0);
return v___x_780_;
}
else
{
lean_object* v_val_781_; lean_object* v___x_783_; uint8_t v_isShared_784_; uint8_t v_isSharedCheck_788_; 
v_val_781_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_788_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_788_ == 0)
{
v___x_783_ = v___x_779_;
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
else
{
lean_inc(v_val_781_);
lean_dec(v___x_779_);
v___x_783_ = lean_box(0);
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
v_resetjp_782_:
{
lean_object* v___x_786_; 
if (v_isShared_784_ == 0)
{
v___x_786_ = v___x_783_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_val_781_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient___redArg(lean_object* v_inst_789_, lean_object* v_inst_790_){
_start:
{
lean_object* v___f_791_; lean_object* v___f_792_; lean_object* v___x_793_; 
lean_inc_ref(v_inst_790_);
v___f_791_ = lean_alloc_closure((void*)(lp_mathlib_encodableQuotient___redArg___lam__0), 3, 2);
lean_closure_set(v___f_791_, 0, v_inst_790_);
lean_closure_set(v___f_791_, 1, v_inst_789_);
v___f_792_ = lean_alloc_closure((void*)(lp_mathlib_encodableQuotient___redArg___lam__1), 2, 1);
lean_closure_set(v___f_792_, 0, v_inst_790_);
v___x_793_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_793_, 0, v___f_791_);
lean_ctor_set(v___x_793_, 1, v___f_792_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodableQuotient(lean_object* v_00_u03b1_794_, lean_object* v_s_795_, lean_object* v_inst_796_, lean_object* v_inst_797_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_mathlib_encodableQuotient___redArg(v_inst_796_, v_inst_797_);
return v___x_798_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Countable_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Countable_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Bool_encodable = _init_lp_mathlib_Bool_encodable();
lean_mark_persistent(lp_mathlib_Bool_encodable);
lp_mathlib_Int_encodable = _init_lp_mathlib_Int_encodable();
lean_mark_persistent(lp_mathlib_Int_encodable);
lp_mathlib_PNat_encodable = _init_lp_mathlib_PNat_encodable();
lean_mark_persistent(lp_mathlib_PNat_encodable);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Countable_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Countable_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
