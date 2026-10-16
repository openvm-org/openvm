// Lean compiler output
// Module: Aesop.Rule
// Imports: public import Init public meta import Init public import Aesop.Rule.Basic
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
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t);
extern double lp_aesop_Aesop_Percent_hundred;
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
uint8_t lean_float_decLt(double, double);
double lean_float_sub(double, double);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Int_repr(lean_object*);
extern double lp_aesop_Aesop_instInhabitedPercent_default;
lean_object* lp_aesop_Aesop_instInhabitedRule_default___redArg(lean_object*);
extern double lp_aesop_Aesop_Percent_fifty;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedRuleName_default;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormRuleInfo_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormRuleInfo;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdNormRuleInfo___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdNormRuleInfo___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdNormRuleInfo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdNormRuleInfo___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdNormRuleInfo___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdNormRuleInfo___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdNormRuleInfo = (const lean_object*)&lp_aesop_Aesop_instOrdNormRuleInfo___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTNormRuleInfo;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLENormRuleInfo;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14_value;
static const lean_string_object lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToStringNormRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToStringNormRule___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToStringNormRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToStringNormRule = (const lean_object*)&lp_aesop_Aesop_instToStringNormRule___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_defaultNormPenalty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_defaultNormPenalty___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_defaultNormPenalty;
static lean_once_cell_t lp_aesop_Aesop_defaultSimpRulePriority___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_defaultSimpRulePriority___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_defaultSimpRulePriority;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedSafety_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedSafety;
static const lean_string_object lp_aesop_Aesop_Safety_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "almostSafe"};
static const lean_object* lp_aesop_Aesop_Safety_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Safety_instToString___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_instToString___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_instToString___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Safety_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Safety_instToString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Safety_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_Safety_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Safety_instToString = (const lean_object*)&lp_aesop_Aesop_Safety_instToString___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedSafeRuleInfo;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdSafeRuleInfo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdSafeRuleInfo___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdSafeRuleInfo___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdSafeRuleInfo = (const lean_object*)&lp_aesop_Aesop_instOrdSafeRuleInfo___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTSafeRuleInfo;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLESafeRuleInfo;
static const lean_string_object lp_aesop_Aesop_instToStringSafeRule___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_aesop_Aesop_instToStringSafeRule___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringSafeRule___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringSafeRule___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToStringSafeRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToStringSafeRule___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToStringSafeRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringSafeRule___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToStringSafeRule = (const lean_object*)&lp_aesop_Aesop_instToStringSafeRule___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_defaultSafePenalty;
LEAN_EXPORT double lp_aesop_Aesop_instInhabitedUnsafeRuleInfo_default;
LEAN_EXPORT double lp_aesop_Aesop_instInhabitedUnsafeRuleInfo;
static lean_once_cell_t lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdUnsafeRuleInfo___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdUnsafeRuleInfo___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdUnsafeRuleInfo___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdUnsafeRuleInfo = (const lean_object*)&lp_aesop_Aesop_instOrdUnsafeRuleInfo___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLTUnsafeRuleInfo;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instLEUnsafeRuleInfo;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringUnsafeRule___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToStringUnsafeRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToStringUnsafeRule___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToStringUnsafeRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringUnsafeRule___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToStringUnsafeRule = (const lean_object*)&lp_aesop_Aesop_instToStringUnsafeRule___closed__0_value;
LEAN_EXPORT double lp_aesop_Aesop_defaultSuccessProbability;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_safe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_safe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_unsafe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_unsafe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRegularRule_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRegularRule;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqRegularRule_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqRegularRule_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqRegularRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqRegularRule_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqRegularRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqRegularRule___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqRegularRule = (const lean_object*)&lp_aesop_Aesop_instBEqRegularRule___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_instToFormat___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_RegularRule_instToFormat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RegularRule_instToFormat___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RegularRule_instToFormat___closed__0 = (const lean_object*)&lp_aesop_Aesop_RegularRule_instToFormat___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RegularRule_instToFormat = (const lean_object*)&lp_aesop_Aesop_RegularRule_instToFormat___closed__0_value;
LEAN_EXPORT double lp_aesop_Aesop_RegularRule_successProbability(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_successProbability___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RegularRule_isSafe(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_isSafe___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RegularRule_isUnsafe(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_isUnsafe___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_withRule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_withRule(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_name___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_indexingMode(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_indexingMode___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_tac(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_tac___boxed(lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpRule_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpRule;
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormSimpRule_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSimpRule_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_NormSimpRule_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_NormSimpRule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_NormSimpRule_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_NormSimpRule_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_NormSimpRule_instBEq = (const lean_object*)&lp_aesop_Aesop_NormSimpRule_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_NormSimpRule_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSimpRule_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_NormSimpRule_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_NormSimpRule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_NormSimpRule_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_NormSimpRule_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_NormSimpRule_instHashable = (const lean_object*)&lp_aesop_Aesop_NormSimpRule_instHashable___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedLocalNormSimpRule = (const lean_object*)&lp_aesop_Aesop_instInhabitedLocalNormSimpRule_default___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_LocalNormSimpRule_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_LocalNormSimpRule_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_LocalNormSimpRule_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_LocalNormSimpRule_instBEq = (const lean_object*)&lp_aesop_Aesop_LocalNormSimpRule_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_LocalNormSimpRule_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_LocalNormSimpRule_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_LocalNormSimpRule_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_LocalNormSimpRule_instHashable = (const lean_object*)&lp_aesop_Aesop_LocalNormSimpRule_instHashable___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__0;
static lean_once_cell_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__1;
static lean_once_cell_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__2;
static lean_once_cell_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__3;
static lean_once_cell_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_LocalNormSimpRule_name___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_name___boxed(lean_object*);
static const lean_ctor_object lp_aesop_Aesop_instInhabitedUnfoldRule_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedUnfoldRule_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedUnfoldRule_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedUnfoldRule_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedUnfoldRule_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedUnfoldRule = (const lean_object*)&lp_aesop_Aesop_instInhabitedUnfoldRule_default___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnfoldRule_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnfoldRule_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnfoldRule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnfoldRule_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnfoldRule_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnfoldRule_instBEq = (const lean_object*)&lp_aesop_Aesop_UnfoldRule_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_UnfoldRule_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnfoldRule_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnfoldRule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnfoldRule_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnfoldRule_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnfoldRule_instHashable = (const lean_object*)&lp_aesop_Aesop_UnfoldRule_instHashable___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_UnfoldRule_name___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_UnfoldRule_name___closed__0;
static lean_once_cell_t lp_aesop_Aesop_UnfoldRule_name___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_UnfoldRule_name___closed__1;
static lean_once_cell_t lp_aesop_Aesop_UnfoldRule_name___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_UnfoldRule_name___closed__2;
static lean_once_cell_t lp_aesop_Aesop_UnfoldRule_name___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_UnfoldRule_name___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_name___boxed(lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(0u);
v___x_2_ = lean_nat_to_int(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormRuleInfo_default(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0, &lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormRuleInfo(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_aesop_Aesop_instInhabitedNormRuleInfo_default;
return v___x_4_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdNormRuleInfo___lam__0(lean_object* v_i_5_, lean_object* v_j_6_){
_start:
{
uint8_t v___x_7_; 
v___x_7_ = lean_int_dec_lt(v_i_5_, v_j_6_);
if (v___x_7_ == 0)
{
uint8_t v___x_8_; 
v___x_8_ = lean_int_dec_eq(v_i_5_, v_j_6_);
if (v___x_8_ == 0)
{
uint8_t v___x_9_; 
v___x_9_ = 2;
return v___x_9_;
}
else
{
uint8_t v___x_10_; 
v___x_10_ = 1;
return v___x_10_;
}
}
else
{
uint8_t v___x_11_; 
v___x_11_ = 0;
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdNormRuleInfo___lam__0___boxed(lean_object* v_i_12_, lean_object* v_j_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_aesop_Aesop_instOrdNormRuleInfo___lam__0(v_i_12_, v_j_13_);
lean_dec(v_j_13_);
lean_dec(v_i_12_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTNormRuleInfo(void){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLENormRuleInfo(void){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_box(0);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringNormRule___lam__0(lean_object* v_r_36_){
_start:
{
lean_object* v_name_37_; lean_object* v_extra_38_; lean_object* v_name_39_; uint8_t v_builder_40_; uint8_t v_phase_41_; uint8_t v_scope_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___y_49_; lean_object* v___y_50_; lean_object* v___y_51_; lean_object* v___y_59_; lean_object* v___y_60_; lean_object* v___y_61_; lean_object* v___y_67_; 
v_name_37_ = lean_ctor_get(v_r_36_, 0);
lean_inc_ref(v_name_37_);
v_extra_38_ = lean_ctor_get(v_r_36_, 3);
lean_inc(v_extra_38_);
lean_dec_ref(v_r_36_);
v_name_39_ = lean_ctor_get(v_name_37_, 0);
lean_inc(v_name_39_);
v_builder_40_ = lean_ctor_get_uint8(v_name_37_, sizeof(void*)*1 + 8);
v_phase_41_ = lean_ctor_get_uint8(v_name_37_, sizeof(void*)*1 + 9);
v_scope_42_ = lean_ctor_get_uint8(v_name_37_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_37_);
v___x_43_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0));
v___x_44_ = l_Int_repr(v_extra_38_);
lean_dec(v_extra_38_);
v___x_45_ = lean_string_append(v___x_43_, v___x_44_);
lean_dec_ref(v___x_44_);
v___x_46_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1));
v___x_47_ = lean_string_append(v___x_45_, v___x_46_);
switch(v_phase_41_)
{
case 0:
{
lean_object* v___x_78_; 
v___x_78_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13));
v___y_67_ = v___x_78_;
goto v___jp_66_;
}
case 1:
{
lean_object* v___x_79_; 
v___x_79_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_67_ = v___x_79_;
goto v___jp_66_;
}
default: 
{
lean_object* v___x_80_; 
v___x_80_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15));
v___y_67_ = v___x_80_;
goto v___jp_66_;
}
}
v___jp_48_:
{
lean_object* v___x_52_; lean_object* v___x_53_; uint8_t v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_52_ = lean_string_append(v___y_50_, v___y_51_);
v___x_53_ = lean_string_append(v___x_52_, v___y_49_);
v___x_54_ = 1;
v___x_55_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_39_, v___x_54_);
v___x_56_ = lean_string_append(v___x_53_, v___x_55_);
lean_dec_ref(v___x_55_);
v___x_57_ = lean_string_append(v___x_47_, v___x_56_);
lean_dec_ref(v___x_56_);
return v___x_57_;
}
v___jp_58_:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_string_append(v___y_60_, v___y_61_);
v___x_63_ = lean_string_append(v___x_62_, v___y_59_);
if (v_scope_42_ == 0)
{
lean_object* v___x_64_; 
v___x_64_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2));
v___y_49_ = v___y_59_;
v___y_50_ = v___x_63_;
v___y_51_ = v___x_64_;
goto v___jp_48_;
}
else
{
lean_object* v___x_65_; 
v___x_65_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3));
v___y_49_ = v___y_59_;
v___y_50_ = v___x_63_;
v___y_51_ = v___x_65_;
goto v___jp_48_;
}
}
v___jp_66_:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4));
lean_inc_ref(v___y_67_);
v___x_69_ = lean_string_append(v___y_67_, v___x_68_);
switch(v_builder_40_)
{
case 0:
{
lean_object* v___x_70_; 
v___x_70_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_70_;
goto v___jp_58_;
}
case 1:
{
lean_object* v___x_71_; 
v___x_71_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_71_;
goto v___jp_58_;
}
case 2:
{
lean_object* v___x_72_; 
v___x_72_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_72_;
goto v___jp_58_;
}
case 3:
{
lean_object* v___x_73_; 
v___x_73_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_73_;
goto v___jp_58_;
}
case 4:
{
lean_object* v___x_74_; 
v___x_74_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_74_;
goto v___jp_58_;
}
case 5:
{
lean_object* v___x_75_; 
v___x_75_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_75_;
goto v___jp_58_;
}
case 6:
{
lean_object* v___x_76_; 
v___x_76_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_76_;
goto v___jp_58_;
}
default: 
{
lean_object* v___x_77_; 
v___x_77_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12));
v___y_59_ = v___x_68_;
v___y_60_ = v___x_69_;
v___y_61_ = v___x_77_;
goto v___jp_58_;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_defaultNormPenalty___closed__0(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lean_unsigned_to_nat(1u);
v___x_84_ = lean_nat_to_int(v___x_83_);
return v___x_84_;
}
}
static lean_object* _init_lp_aesop_Aesop_defaultNormPenalty(void){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_obj_once(&lp_aesop_Aesop_defaultNormPenalty___closed__0, &lp_aesop_Aesop_defaultNormPenalty___closed__0_once, _init_lp_aesop_Aesop_defaultNormPenalty___closed__0);
return v___x_85_;
}
}
static lean_object* _init_lp_aesop_Aesop_defaultSimpRulePriority___closed__0(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(1000u);
v___x_87_ = lean_nat_to_int(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_aesop_Aesop_defaultSimpRulePriority(void){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lean_obj_once(&lp_aesop_Aesop_defaultSimpRulePriority___closed__0, &lp_aesop_Aesop_defaultSimpRulePriority___closed__0_once, _init_lp_aesop_Aesop_defaultSimpRulePriority___closed__0);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorIdx(uint8_t v_x_89_){
_start:
{
if (v_x_89_ == 0)
{
lean_object* v___x_90_; 
v___x_90_ = lean_unsigned_to_nat(0u);
return v___x_90_;
}
else
{
lean_object* v___x_91_; 
v___x_91_ = lean_unsigned_to_nat(1u);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorIdx___boxed(lean_object* v_x_92_){
_start:
{
uint8_t v_x_boxed_93_; lean_object* v_res_94_; 
v_x_boxed_93_ = lean_unbox(v_x_92_);
v_res_94_ = lp_aesop_Aesop_Safety_ctorIdx(v_x_boxed_93_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___redArg(lean_object* v_k_95_){
_start:
{
lean_inc(v_k_95_);
return v_k_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___redArg___boxed(lean_object* v_k_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_aesop_Aesop_Safety_ctorElim___redArg(v_k_96_);
lean_dec(v_k_96_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim(lean_object* v_motive_98_, lean_object* v_ctorIdx_99_, uint8_t v_t_100_, lean_object* v_h_101_, lean_object* v_k_102_){
_start:
{
lean_inc(v_k_102_);
return v_k_102_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_ctorElim___boxed(lean_object* v_motive_103_, lean_object* v_ctorIdx_104_, lean_object* v_t_105_, lean_object* v_h_106_, lean_object* v_k_107_){
_start:
{
uint8_t v_t_boxed_108_; lean_object* v_res_109_; 
v_t_boxed_108_ = lean_unbox(v_t_105_);
v_res_109_ = lp_aesop_Aesop_Safety_ctorElim(v_motive_103_, v_ctorIdx_104_, v_t_boxed_108_, v_h_106_, v_k_107_);
lean_dec(v_k_107_);
lean_dec(v_ctorIdx_104_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___redArg(lean_object* v_safe_110_){
_start:
{
lean_inc(v_safe_110_);
return v_safe_110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___redArg___boxed(lean_object* v_safe_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_aesop_Aesop_Safety_safe_elim___redArg(v_safe_111_);
lean_dec(v_safe_111_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim(lean_object* v_motive_113_, uint8_t v_t_114_, lean_object* v_h_115_, lean_object* v_safe_116_){
_start:
{
lean_inc(v_safe_116_);
return v_safe_116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_safe_elim___boxed(lean_object* v_motive_117_, lean_object* v_t_118_, lean_object* v_h_119_, lean_object* v_safe_120_){
_start:
{
uint8_t v_t_boxed_121_; lean_object* v_res_122_; 
v_t_boxed_121_ = lean_unbox(v_t_118_);
v_res_122_ = lp_aesop_Aesop_Safety_safe_elim(v_motive_117_, v_t_boxed_121_, v_h_119_, v_safe_120_);
lean_dec(v_safe_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___redArg(lean_object* v_almostSafe_123_){
_start:
{
lean_inc(v_almostSafe_123_);
return v_almostSafe_123_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___redArg___boxed(lean_object* v_almostSafe_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_aesop_Aesop_Safety_almostSafe_elim___redArg(v_almostSafe_124_);
lean_dec(v_almostSafe_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim(lean_object* v_motive_126_, uint8_t v_t_127_, lean_object* v_h_128_, lean_object* v_almostSafe_129_){
_start:
{
lean_inc(v_almostSafe_129_);
return v_almostSafe_129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_almostSafe_elim___boxed(lean_object* v_motive_130_, lean_object* v_t_131_, lean_object* v_h_132_, lean_object* v_almostSafe_133_){
_start:
{
uint8_t v_t_boxed_134_; lean_object* v_res_135_; 
v_t_boxed_134_ = lean_unbox(v_t_131_);
v_res_135_ = lp_aesop_Aesop_Safety_almostSafe_elim(v_motive_130_, v_t_boxed_134_, v_h_132_, v_almostSafe_133_);
lean_dec(v_almostSafe_133_);
return v_res_135_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedSafety_default(void){
_start:
{
uint8_t v___x_136_; 
v___x_136_ = 0;
return v___x_136_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedSafety(void){
_start:
{
uint8_t v___x_137_; 
v___x_137_ = 0;
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_instToString___lam__0(uint8_t v_x_139_){
_start:
{
if (v_x_139_ == 0)
{
lean_object* v___x_140_; 
v___x_140_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
return v___x_140_;
}
else
{
lean_object* v___x_141_; 
v___x_141_ = ((lean_object*)(lp_aesop_Aesop_Safety_instToString___lam__0___closed__0));
return v___x_141_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Safety_instToString___lam__0___boxed(lean_object* v_x_142_){
_start:
{
uint8_t v_x_25__boxed_143_; lean_object* v_res_144_; 
v_x_25__boxed_143_ = lean_unbox(v_x_142_);
v_res_144_ = lp_aesop_Aesop_Safety_instToString___lam__0(v_x_25__boxed_143_);
return v_res_144_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0(void){
_start:
{
uint8_t v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_147_ = 0;
v___x_148_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0, &lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedNormRuleInfo_default___closed__0);
v___x_149_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set_uint8(v___x_149_, sizeof(void*)*1, v___x_147_);
return v___x_149_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo_default(void){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0, &lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo_default___closed__0);
return v___x_150_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo(void){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
return v___x_151_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0(lean_object* v_i_152_, lean_object* v_j_153_){
_start:
{
lean_object* v_penalty_154_; lean_object* v_penalty_155_; uint8_t v___x_156_; 
v_penalty_154_ = lean_ctor_get(v_i_152_, 0);
v_penalty_155_ = lean_ctor_get(v_j_153_, 0);
v___x_156_ = lean_int_dec_lt(v_penalty_154_, v_penalty_155_);
if (v___x_156_ == 0)
{
uint8_t v___x_157_; 
v___x_157_ = lean_int_dec_eq(v_penalty_154_, v_penalty_155_);
if (v___x_157_ == 0)
{
uint8_t v___x_158_; 
v___x_158_ = 2;
return v___x_158_;
}
else
{
uint8_t v___x_159_; 
v___x_159_ = 1;
return v___x_159_;
}
}
else
{
uint8_t v___x_160_; 
v___x_160_ = 0;
return v___x_160_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0___boxed(lean_object* v_i_161_, lean_object* v_j_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_aesop_Aesop_instOrdSafeRuleInfo___lam__0(v_i_161_, v_j_162_);
lean_dec_ref(v_j_162_);
lean_dec_ref(v_i_161_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTSafeRuleInfo(void){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_box(0);
return v___x_167_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLESafeRuleInfo(void){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_box(0);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringSafeRule___lam__0(lean_object* v_r_170_){
_start:
{
lean_object* v_name_171_; lean_object* v_extra_172_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_186_; lean_object* v___y_187_; lean_object* v___y_188_; lean_object* v___y_189_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v_penalty_209_; uint8_t v_safety_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___y_217_; 
v_name_171_ = lean_ctor_get(v_r_170_, 0);
lean_inc_ref(v_name_171_);
v_extra_172_ = lean_ctor_get(v_r_170_, 3);
lean_inc(v_extra_172_);
lean_dec_ref(v_r_170_);
v_penalty_209_ = lean_ctor_get(v_extra_172_, 0);
lean_inc(v_penalty_209_);
v_safety_210_ = lean_ctor_get_uint8(v_extra_172_, sizeof(void*)*1);
lean_dec(v_extra_172_);
v___x_211_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0));
v___x_212_ = l_Int_repr(v_penalty_209_);
lean_dec(v_penalty_209_);
v___x_213_ = lean_string_append(v___x_211_, v___x_212_);
lean_dec_ref(v___x_212_);
v___x_214_ = ((lean_object*)(lp_aesop_Aesop_instToStringSafeRule___lam__0___closed__0));
v___x_215_ = lean_string_append(v___x_213_, v___x_214_);
if (v_safety_210_ == 0)
{
lean_object* v___x_225_; 
v___x_225_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_217_ = v___x_225_;
goto v___jp_216_;
}
else
{
lean_object* v___x_226_; 
v___x_226_ = ((lean_object*)(lp_aesop_Aesop_Safety_instToString___lam__0___closed__0));
v___y_217_ = v___x_226_;
goto v___jp_216_;
}
v___jp_173_:
{
lean_object* v_name_178_; lean_object* v___x_179_; lean_object* v___x_180_; uint8_t v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v_name_178_ = lean_ctor_get(v_name_171_, 0);
lean_inc(v_name_178_);
lean_dec_ref(v_name_171_);
v___x_179_ = lean_string_append(v___y_176_, v___y_177_);
v___x_180_ = lean_string_append(v___x_179_, v___y_175_);
v___x_181_ = 1;
v___x_182_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_178_, v___x_181_);
v___x_183_ = lean_string_append(v___x_180_, v___x_182_);
lean_dec_ref(v___x_182_);
v___x_184_ = lean_string_append(v___y_174_, v___x_183_);
lean_dec_ref(v___x_183_);
return v___x_184_;
}
v___jp_185_:
{
uint8_t v_scope_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v_scope_190_ = lean_ctor_get_uint8(v_name_171_, sizeof(void*)*1 + 10);
v___x_191_ = lean_string_append(v___y_188_, v___y_189_);
v___x_192_ = lean_string_append(v___x_191_, v___y_187_);
if (v_scope_190_ == 0)
{
lean_object* v___x_193_; 
v___x_193_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2));
v___y_174_ = v___y_186_;
v___y_175_ = v___y_187_;
v___y_176_ = v___x_192_;
v___y_177_ = v___x_193_;
goto v___jp_173_;
}
else
{
lean_object* v___x_194_; 
v___x_194_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3));
v___y_174_ = v___y_186_;
v___y_175_ = v___y_187_;
v___y_176_ = v___x_192_;
v___y_177_ = v___x_194_;
goto v___jp_173_;
}
}
v___jp_195_:
{
uint8_t v_builder_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v_builder_198_ = lean_ctor_get_uint8(v_name_171_, sizeof(void*)*1 + 8);
v___x_199_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4));
lean_inc_ref(v___y_197_);
v___x_200_ = lean_string_append(v___y_197_, v___x_199_);
switch(v_builder_198_)
{
case 0:
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_201_;
goto v___jp_185_;
}
case 1:
{
lean_object* v___x_202_; 
v___x_202_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_202_;
goto v___jp_185_;
}
case 2:
{
lean_object* v___x_203_; 
v___x_203_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_203_;
goto v___jp_185_;
}
case 3:
{
lean_object* v___x_204_; 
v___x_204_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_204_;
goto v___jp_185_;
}
case 4:
{
lean_object* v___x_205_; 
v___x_205_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_205_;
goto v___jp_185_;
}
case 5:
{
lean_object* v___x_206_; 
v___x_206_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_206_;
goto v___jp_185_;
}
case 6:
{
lean_object* v___x_207_; 
v___x_207_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_207_;
goto v___jp_185_;
}
default: 
{
lean_object* v___x_208_; 
v___x_208_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12));
v___y_186_ = v___y_196_;
v___y_187_ = v___x_199_;
v___y_188_ = v___x_200_;
v___y_189_ = v___x_208_;
goto v___jp_185_;
}
}
}
v___jp_216_:
{
uint8_t v_phase_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v_phase_218_ = lean_ctor_get_uint8(v_name_171_, sizeof(void*)*1 + 9);
v___x_219_ = lean_string_append(v___x_215_, v___y_217_);
v___x_220_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1));
v___x_221_ = lean_string_append(v___x_219_, v___x_220_);
switch(v_phase_218_)
{
case 0:
{
lean_object* v___x_222_; 
v___x_222_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13));
v___y_196_ = v___x_221_;
v___y_197_ = v___x_222_;
goto v___jp_195_;
}
case 1:
{
lean_object* v___x_223_; 
v___x_223_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_196_ = v___x_221_;
v___y_197_ = v___x_223_;
goto v___jp_195_;
}
default: 
{
lean_object* v___x_224_; 
v___x_224_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15));
v___y_196_ = v___x_221_;
v___y_197_ = v___x_224_;
goto v___jp_195_;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_defaultSafePenalty(void){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_obj_once(&lp_aesop_Aesop_defaultNormPenalty___closed__0, &lp_aesop_Aesop_defaultNormPenalty___closed__0_once, _init_lp_aesop_Aesop_defaultNormPenalty___closed__0);
return v___x_229_;
}
}
static double _init_lp_aesop_Aesop_instInhabitedUnsafeRuleInfo_default(void){
_start:
{
double v___x_230_; 
v___x_230_ = lp_aesop_Aesop_instInhabitedPercent_default;
return v___x_230_;
}
}
static double _init_lp_aesop_Aesop_instInhabitedUnsafeRuleInfo(void){
_start:
{
double v___x_231_; 
v___x_231_ = lp_aesop_Aesop_instInhabitedPercent_default;
return v___x_231_;
}
}
static double _init_lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0(void){
_start:
{
lean_object* v___x_232_; uint8_t v___x_233_; lean_object* v___x_234_; double v___x_235_; 
v___x_232_ = lean_unsigned_to_nat(5u);
v___x_233_ = 1;
v___x_234_ = lean_unsigned_to_nat(1u);
v___x_235_ = l_Float_ofScientific(v___x_234_, v___x_233_, v___x_232_);
return v___x_235_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0(double v_i_236_, double v_j_237_){
_start:
{
uint8_t v___y_239_; uint8_t v___x_244_; 
v___x_244_ = lean_float_decLt(v_i_236_, v_j_237_);
if (v___x_244_ == 0)
{
double v___x_245_; double v___x_246_; uint8_t v___x_247_; 
v___x_245_ = lean_float_sub(v_i_236_, v_j_237_);
v___x_246_ = lean_float_once(&lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0, &lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0_once, _init_lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___closed__0);
v___x_247_ = lean_float_decLt(v___x_245_, v___x_246_);
v___y_239_ = v___x_247_;
goto v___jp_238_;
}
else
{
double v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; double v___x_251_; uint8_t v___x_252_; 
v___x_248_ = lean_float_sub(v_j_237_, v_i_236_);
v___x_249_ = lean_unsigned_to_nat(1u);
v___x_250_ = lean_unsigned_to_nat(5u);
v___x_251_ = l_Float_ofScientific(v___x_249_, v___x_244_, v___x_250_);
v___x_252_ = lean_float_decLt(v___x_248_, v___x_251_);
v___y_239_ = v___x_252_;
goto v___jp_238_;
}
v___jp_238_:
{
if (v___y_239_ == 0)
{
uint8_t v___x_240_; 
v___x_240_ = lean_float_decLt(v_j_237_, v_i_236_);
if (v___x_240_ == 0)
{
uint8_t v___x_241_; 
v___x_241_ = 2;
return v___x_241_;
}
else
{
uint8_t v___x_242_; 
v___x_242_ = 0;
return v___x_242_;
}
}
else
{
uint8_t v___x_243_; 
v___x_243_ = 1;
return v___x_243_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0___boxed(lean_object* v_i_253_, lean_object* v_j_254_){
_start:
{
double v_i_boxed_255_; double v_j_boxed_256_; uint8_t v_res_257_; lean_object* v_r_258_; 
v_i_boxed_255_ = lean_unbox_float(v_i_253_);
lean_dec_ref(v_i_253_);
v_j_boxed_256_ = lean_unbox_float(v_j_254_);
lean_dec_ref(v_j_254_);
v_res_257_ = lp_aesop_Aesop_instOrdUnsafeRuleInfo___lam__0(v_i_boxed_255_, v_j_boxed_256_);
v_r_258_ = lean_box(v_res_257_);
return v_r_258_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLTUnsafeRuleInfo(void){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_box(0);
return v___x_261_;
}
}
static lean_object* _init_lp_aesop_Aesop_instLEUnsafeRuleInfo(void){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lean_box(0);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringUnsafeRule___lam__0(lean_object* v_r_263_){
_start:
{
lean_object* v_name_264_; lean_object* v_extra_265_; lean_object* v_name_266_; uint8_t v_builder_267_; uint8_t v_phase_268_; uint8_t v_scope_269_; lean_object* v___x_270_; double v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___y_277_; lean_object* v___y_278_; lean_object* v___y_279_; lean_object* v___y_287_; lean_object* v___y_288_; lean_object* v___y_289_; lean_object* v___y_295_; 
v_name_264_ = lean_ctor_get(v_r_263_, 0);
lean_inc_ref(v_name_264_);
v_extra_265_ = lean_ctor_get(v_r_263_, 3);
lean_inc(v_extra_265_);
lean_dec_ref(v_r_263_);
v_name_266_ = lean_ctor_get(v_name_264_, 0);
lean_inc(v_name_266_);
v_builder_267_ = lean_ctor_get_uint8(v_name_264_, sizeof(void*)*1 + 8);
v_phase_268_ = lean_ctor_get_uint8(v_name_264_, sizeof(void*)*1 + 9);
v_scope_269_ = lean_ctor_get_uint8(v_name_264_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_264_);
v___x_270_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0));
v___x_271_ = lean_unbox_float(v_extra_265_);
lean_dec(v_extra_265_);
v___x_272_ = lp_aesop_Aesop_Percent_toHumanString(v___x_271_);
v___x_273_ = lean_string_append(v___x_270_, v___x_272_);
lean_dec_ref(v___x_272_);
v___x_274_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1));
v___x_275_ = lean_string_append(v___x_273_, v___x_274_);
switch(v_phase_268_)
{
case 0:
{
lean_object* v___x_306_; 
v___x_306_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13));
v___y_295_ = v___x_306_;
goto v___jp_294_;
}
case 1:
{
lean_object* v___x_307_; 
v___x_307_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_295_ = v___x_307_;
goto v___jp_294_;
}
default: 
{
lean_object* v___x_308_; 
v___x_308_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15));
v___y_295_ = v___x_308_;
goto v___jp_294_;
}
}
v___jp_276_:
{
lean_object* v___x_280_; lean_object* v___x_281_; uint8_t v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_280_ = lean_string_append(v___y_277_, v___y_279_);
v___x_281_ = lean_string_append(v___x_280_, v___y_278_);
v___x_282_ = 1;
v___x_283_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_266_, v___x_282_);
v___x_284_ = lean_string_append(v___x_281_, v___x_283_);
lean_dec_ref(v___x_283_);
v___x_285_ = lean_string_append(v___x_275_, v___x_284_);
lean_dec_ref(v___x_284_);
return v___x_285_;
}
v___jp_286_:
{
lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_290_ = lean_string_append(v___y_288_, v___y_289_);
v___x_291_ = lean_string_append(v___x_290_, v___y_287_);
if (v_scope_269_ == 0)
{
lean_object* v___x_292_; 
v___x_292_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2));
v___y_277_ = v___x_291_;
v___y_278_ = v___y_287_;
v___y_279_ = v___x_292_;
goto v___jp_276_;
}
else
{
lean_object* v___x_293_; 
v___x_293_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3));
v___y_277_ = v___x_291_;
v___y_278_ = v___y_287_;
v___y_279_ = v___x_293_;
goto v___jp_276_;
}
}
v___jp_294_:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4));
lean_inc_ref(v___y_295_);
v___x_297_ = lean_string_append(v___y_295_, v___x_296_);
switch(v_builder_267_)
{
case 0:
{
lean_object* v___x_298_; 
v___x_298_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_298_;
goto v___jp_286_;
}
case 1:
{
lean_object* v___x_299_; 
v___x_299_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_299_;
goto v___jp_286_;
}
case 2:
{
lean_object* v___x_300_; 
v___x_300_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_300_;
goto v___jp_286_;
}
case 3:
{
lean_object* v___x_301_; 
v___x_301_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_301_;
goto v___jp_286_;
}
case 4:
{
lean_object* v___x_302_; 
v___x_302_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_302_;
goto v___jp_286_;
}
case 5:
{
lean_object* v___x_303_; 
v___x_303_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_303_;
goto v___jp_286_;
}
case 6:
{
lean_object* v___x_304_; 
v___x_304_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_304_;
goto v___jp_286_;
}
default: 
{
lean_object* v___x_305_; 
v___x_305_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12));
v___y_287_ = v___x_296_;
v___y_288_ = v___x_297_;
v___y_289_ = v___x_305_;
goto v___jp_286_;
}
}
}
}
}
static double _init_lp_aesop_Aesop_defaultSuccessProbability(void){
_start:
{
double v___x_311_; 
v___x_311_ = lp_aesop_Aesop_Percent_fifty;
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorIdx(lean_object* v_x_312_){
_start:
{
if (lean_obj_tag(v_x_312_) == 0)
{
lean_object* v___x_313_; 
v___x_313_ = lean_unsigned_to_nat(0u);
return v___x_313_;
}
else
{
lean_object* v___x_314_; 
v___x_314_ = lean_unsigned_to_nat(1u);
return v___x_314_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorIdx___boxed(lean_object* v_x_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_aesop_Aesop_RegularRule_ctorIdx(v_x_315_);
lean_dec_ref(v_x_315_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim___redArg(lean_object* v_t_317_, lean_object* v_k_318_){
_start:
{
lean_object* v_r_319_; lean_object* v___x_320_; 
v_r_319_ = lean_ctor_get(v_t_317_, 0);
lean_inc_ref(v_r_319_);
lean_dec_ref(v_t_317_);
v___x_320_ = lean_apply_1(v_k_318_, v_r_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim(lean_object* v_motive_321_, lean_object* v_ctorIdx_322_, lean_object* v_t_323_, lean_object* v_h_324_, lean_object* v_k_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_aesop_Aesop_RegularRule_ctorElim___redArg(v_t_323_, v_k_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_ctorElim___boxed(lean_object* v_motive_327_, lean_object* v_ctorIdx_328_, lean_object* v_t_329_, lean_object* v_h_330_, lean_object* v_k_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_aesop_Aesop_RegularRule_ctorElim(v_motive_327_, v_ctorIdx_328_, v_t_329_, v_h_330_, v_k_331_);
lean_dec(v_ctorIdx_328_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_safe_elim___redArg(lean_object* v_t_333_, lean_object* v_safe_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_aesop_Aesop_RegularRule_ctorElim___redArg(v_t_333_, v_safe_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_safe_elim(lean_object* v_motive_336_, lean_object* v_t_337_, lean_object* v_h_338_, lean_object* v_safe_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_aesop_Aesop_RegularRule_ctorElim___redArg(v_t_337_, v_safe_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_unsafe_elim___redArg(lean_object* v_t_341_, lean_object* v_unsafe_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_aesop_Aesop_RegularRule_ctorElim___redArg(v_t_341_, v_unsafe_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_unsafe_elim(lean_object* v_motive_344_, lean_object* v_t_345_, lean_object* v_h_346_, lean_object* v_unsafe_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_aesop_Aesop_RegularRule_ctorElim___redArg(v_t_345_, v_unsafe_347_);
return v___x_348_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_349_ = lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
v___x_350_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v___x_349_);
return v___x_350_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_351_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0, &lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedRegularRule_default___closed__0);
v___x_352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
return v___x_352_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRegularRule_default(void){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1, &lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedRegularRule_default___closed__1);
return v___x_353_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRegularRule(void){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lp_aesop_Aesop_instInhabitedRegularRule_default;
return v___x_354_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqRegularRule_beq(lean_object* v_x_355_, lean_object* v_x_356_){
_start:
{
if (lean_obj_tag(v_x_355_) == 0)
{
if (lean_obj_tag(v_x_356_) == 0)
{
lean_object* v_r_357_; lean_object* v_r_358_; lean_object* v_name_359_; lean_object* v_name_360_; lean_object* v_name_361_; uint8_t v_builder_362_; uint8_t v_phase_363_; uint8_t v_scope_364_; uint64_t v_hash_365_; lean_object* v_name_366_; uint8_t v_builder_367_; uint8_t v_phase_368_; uint8_t v_scope_369_; uint64_t v_hash_370_; uint8_t v___y_372_; uint8_t v___x_376_; 
v_r_357_ = lean_ctor_get(v_x_355_, 0);
v_r_358_ = lean_ctor_get(v_x_356_, 0);
v_name_359_ = lean_ctor_get(v_r_357_, 0);
v_name_360_ = lean_ctor_get(v_r_358_, 0);
v_name_361_ = lean_ctor_get(v_name_359_, 0);
v_builder_362_ = lean_ctor_get_uint8(v_name_359_, sizeof(void*)*1 + 8);
v_phase_363_ = lean_ctor_get_uint8(v_name_359_, sizeof(void*)*1 + 9);
v_scope_364_ = lean_ctor_get_uint8(v_name_359_, sizeof(void*)*1 + 10);
v_hash_365_ = lean_ctor_get_uint64(v_name_359_, sizeof(void*)*1);
v_name_366_ = lean_ctor_get(v_name_360_, 0);
v_builder_367_ = lean_ctor_get_uint8(v_name_360_, sizeof(void*)*1 + 8);
v_phase_368_ = lean_ctor_get_uint8(v_name_360_, sizeof(void*)*1 + 9);
v_scope_369_ = lean_ctor_get_uint8(v_name_360_, sizeof(void*)*1 + 10);
v_hash_370_ = lean_ctor_get_uint64(v_name_360_, sizeof(void*)*1);
v___x_376_ = lean_uint64_dec_eq(v_hash_365_, v_hash_370_);
if (v___x_376_ == 0)
{
v___y_372_ = v___x_376_;
goto v___jp_371_;
}
else
{
uint8_t v___x_377_; 
v___x_377_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_362_, v_builder_367_);
v___y_372_ = v___x_377_;
goto v___jp_371_;
}
v___jp_371_:
{
if (v___y_372_ == 0)
{
return v___y_372_;
}
else
{
uint8_t v___x_373_; 
v___x_373_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_363_, v_phase_368_);
if (v___x_373_ == 0)
{
return v___x_373_;
}
else
{
uint8_t v___x_374_; 
v___x_374_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_364_, v_scope_369_);
if (v___x_374_ == 0)
{
return v___x_374_;
}
else
{
uint8_t v___x_375_; 
v___x_375_ = lean_name_eq(v_name_361_, v_name_366_);
return v___x_375_;
}
}
}
}
}
else
{
uint8_t v___x_378_; 
v___x_378_ = 0;
return v___x_378_;
}
}
else
{
if (lean_obj_tag(v_x_356_) == 1)
{
lean_object* v_r_379_; lean_object* v_r_380_; lean_object* v_name_381_; lean_object* v_name_382_; lean_object* v_name_383_; uint8_t v_builder_384_; uint8_t v_phase_385_; uint8_t v_scope_386_; uint64_t v_hash_387_; lean_object* v_name_388_; uint8_t v_builder_389_; uint8_t v_phase_390_; uint8_t v_scope_391_; uint64_t v_hash_392_; uint8_t v___y_394_; uint8_t v___x_398_; 
v_r_379_ = lean_ctor_get(v_x_355_, 0);
v_r_380_ = lean_ctor_get(v_x_356_, 0);
v_name_381_ = lean_ctor_get(v_r_379_, 0);
v_name_382_ = lean_ctor_get(v_r_380_, 0);
v_name_383_ = lean_ctor_get(v_name_381_, 0);
v_builder_384_ = lean_ctor_get_uint8(v_name_381_, sizeof(void*)*1 + 8);
v_phase_385_ = lean_ctor_get_uint8(v_name_381_, sizeof(void*)*1 + 9);
v_scope_386_ = lean_ctor_get_uint8(v_name_381_, sizeof(void*)*1 + 10);
v_hash_387_ = lean_ctor_get_uint64(v_name_381_, sizeof(void*)*1);
v_name_388_ = lean_ctor_get(v_name_382_, 0);
v_builder_389_ = lean_ctor_get_uint8(v_name_382_, sizeof(void*)*1 + 8);
v_phase_390_ = lean_ctor_get_uint8(v_name_382_, sizeof(void*)*1 + 9);
v_scope_391_ = lean_ctor_get_uint8(v_name_382_, sizeof(void*)*1 + 10);
v_hash_392_ = lean_ctor_get_uint64(v_name_382_, sizeof(void*)*1);
v___x_398_ = lean_uint64_dec_eq(v_hash_387_, v_hash_392_);
if (v___x_398_ == 0)
{
v___y_394_ = v___x_398_;
goto v___jp_393_;
}
else
{
uint8_t v___x_399_; 
v___x_399_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_384_, v_builder_389_);
v___y_394_ = v___x_399_;
goto v___jp_393_;
}
v___jp_393_:
{
if (v___y_394_ == 0)
{
return v___y_394_;
}
else
{
uint8_t v___x_395_; 
v___x_395_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_385_, v_phase_390_);
if (v___x_395_ == 0)
{
return v___x_395_;
}
else
{
uint8_t v___x_396_; 
v___x_396_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_386_, v_scope_391_);
if (v___x_396_ == 0)
{
return v___x_396_;
}
else
{
uint8_t v___x_397_; 
v___x_397_ = lean_name_eq(v_name_383_, v_name_388_);
return v___x_397_;
}
}
}
}
}
else
{
uint8_t v___x_400_; 
v___x_400_ = 0;
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqRegularRule_beq___boxed(lean_object* v_x_401_, lean_object* v_x_402_){
_start:
{
uint8_t v_res_403_; lean_object* v_r_404_; 
v_res_403_ = lp_aesop_Aesop_instBEqRegularRule_beq(v_x_401_, v_x_402_);
lean_dec_ref(v_x_402_);
lean_dec_ref(v_x_401_);
v_r_404_ = lean_box(v_res_403_);
return v_r_404_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_instToFormat___lam__0(lean_object* v_x_407_){
_start:
{
if (lean_obj_tag(v_x_407_) == 0)
{
lean_object* v_r_408_; lean_object* v___x_410_; uint8_t v_isShared_411_; uint8_t v_isSharedCheck_471_; 
v_r_408_ = lean_ctor_get(v_x_407_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v_x_407_);
if (v_isSharedCheck_471_ == 0)
{
v___x_410_ = v_x_407_;
v_isShared_411_ = v_isSharedCheck_471_;
goto v_resetjp_409_;
}
else
{
lean_inc(v_r_408_);
lean_dec(v_x_407_);
v___x_410_ = lean_box(0);
v_isShared_411_ = v_isSharedCheck_471_;
goto v_resetjp_409_;
}
v_resetjp_409_:
{
lean_object* v_name_412_; lean_object* v_extra_413_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_418_; lean_object* v___y_430_; lean_object* v___y_431_; lean_object* v___y_432_; lean_object* v___y_433_; lean_object* v___y_440_; lean_object* v___y_441_; lean_object* v_penalty_453_; uint8_t v_safety_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___y_461_; 
v_name_412_ = lean_ctor_get(v_r_408_, 0);
lean_inc_ref(v_name_412_);
v_extra_413_ = lean_ctor_get(v_r_408_, 3);
lean_inc(v_extra_413_);
lean_dec_ref(v_r_408_);
v_penalty_453_ = lean_ctor_get(v_extra_413_, 0);
lean_inc(v_penalty_453_);
v_safety_454_ = lean_ctor_get_uint8(v_extra_413_, sizeof(void*)*1);
lean_dec(v_extra_413_);
v___x_455_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0));
v___x_456_ = l_Int_repr(v_penalty_453_);
lean_dec(v_penalty_453_);
v___x_457_ = lean_string_append(v___x_455_, v___x_456_);
lean_dec_ref(v___x_456_);
v___x_458_ = ((lean_object*)(lp_aesop_Aesop_instToStringSafeRule___lam__0___closed__0));
v___x_459_ = lean_string_append(v___x_457_, v___x_458_);
if (v_safety_454_ == 0)
{
lean_object* v___x_469_; 
v___x_469_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_461_ = v___x_469_;
goto v___jp_460_;
}
else
{
lean_object* v___x_470_; 
v___x_470_ = ((lean_object*)(lp_aesop_Aesop_Safety_instToString___lam__0___closed__0));
v___y_461_ = v___x_470_;
goto v___jp_460_;
}
v___jp_414_:
{
lean_object* v_name_419_; lean_object* v___x_420_; lean_object* v___x_421_; uint8_t v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_427_; 
v_name_419_ = lean_ctor_get(v_name_412_, 0);
lean_inc(v_name_419_);
lean_dec_ref(v_name_412_);
v___x_420_ = lean_string_append(v___y_417_, v___y_418_);
v___x_421_ = lean_string_append(v___x_420_, v___y_416_);
v___x_422_ = 1;
v___x_423_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_419_, v___x_422_);
v___x_424_ = lean_string_append(v___x_421_, v___x_423_);
lean_dec_ref(v___x_423_);
v___x_425_ = lean_string_append(v___y_415_, v___x_424_);
lean_dec_ref(v___x_424_);
if (v_isShared_411_ == 0)
{
lean_ctor_set_tag(v___x_410_, 3);
lean_ctor_set(v___x_410_, 0, v___x_425_);
v___x_427_ = v___x_410_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_425_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
v___jp_429_:
{
uint8_t v_scope_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v_scope_434_ = lean_ctor_get_uint8(v_name_412_, sizeof(void*)*1 + 10);
v___x_435_ = lean_string_append(v___y_432_, v___y_433_);
v___x_436_ = lean_string_append(v___x_435_, v___y_431_);
if (v_scope_434_ == 0)
{
lean_object* v___x_437_; 
v___x_437_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2));
v___y_415_ = v___y_430_;
v___y_416_ = v___y_431_;
v___y_417_ = v___x_436_;
v___y_418_ = v___x_437_;
goto v___jp_414_;
}
else
{
lean_object* v___x_438_; 
v___x_438_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3));
v___y_415_ = v___y_430_;
v___y_416_ = v___y_431_;
v___y_417_ = v___x_436_;
v___y_418_ = v___x_438_;
goto v___jp_414_;
}
}
v___jp_439_:
{
uint8_t v_builder_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
v_builder_442_ = lean_ctor_get_uint8(v_name_412_, sizeof(void*)*1 + 8);
v___x_443_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4));
lean_inc_ref(v___y_441_);
v___x_444_ = lean_string_append(v___y_441_, v___x_443_);
switch(v_builder_442_)
{
case 0:
{
lean_object* v___x_445_; 
v___x_445_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_445_;
goto v___jp_429_;
}
case 1:
{
lean_object* v___x_446_; 
v___x_446_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_446_;
goto v___jp_429_;
}
case 2:
{
lean_object* v___x_447_; 
v___x_447_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_447_;
goto v___jp_429_;
}
case 3:
{
lean_object* v___x_448_; 
v___x_448_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_448_;
goto v___jp_429_;
}
case 4:
{
lean_object* v___x_449_; 
v___x_449_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_449_;
goto v___jp_429_;
}
case 5:
{
lean_object* v___x_450_; 
v___x_450_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_450_;
goto v___jp_429_;
}
case 6:
{
lean_object* v___x_451_; 
v___x_451_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_451_;
goto v___jp_429_;
}
default: 
{
lean_object* v___x_452_; 
v___x_452_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12));
v___y_430_ = v___y_440_;
v___y_431_ = v___x_443_;
v___y_432_ = v___x_444_;
v___y_433_ = v___x_452_;
goto v___jp_429_;
}
}
}
v___jp_460_:
{
uint8_t v_phase_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v_phase_462_ = lean_ctor_get_uint8(v_name_412_, sizeof(void*)*1 + 9);
v___x_463_ = lean_string_append(v___x_459_, v___y_461_);
v___x_464_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1));
v___x_465_ = lean_string_append(v___x_463_, v___x_464_);
switch(v_phase_462_)
{
case 0:
{
lean_object* v___x_466_; 
v___x_466_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13));
v___y_440_ = v___x_465_;
v___y_441_ = v___x_466_;
goto v___jp_439_;
}
case 1:
{
lean_object* v___x_467_; 
v___x_467_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_440_ = v___x_465_;
v___y_441_ = v___x_467_;
goto v___jp_439_;
}
default: 
{
lean_object* v___x_468_; 
v___x_468_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15));
v___y_440_ = v___x_465_;
v___y_441_ = v___x_468_;
goto v___jp_439_;
}
}
}
}
}
else
{
lean_object* v_r_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_524_; 
v_r_472_ = lean_ctor_get(v_x_407_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v_x_407_);
if (v_isSharedCheck_524_ == 0)
{
v___x_474_ = v_x_407_;
v_isShared_475_ = v_isSharedCheck_524_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_r_472_);
lean_dec(v_x_407_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_524_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v_name_476_; lean_object* v_extra_477_; lean_object* v_name_478_; uint8_t v_builder_479_; uint8_t v_phase_480_; uint8_t v_scope_481_; lean_object* v___x_482_; double v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___y_489_; lean_object* v___y_490_; lean_object* v___y_491_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; lean_object* v___y_510_; 
v_name_476_ = lean_ctor_get(v_r_472_, 0);
lean_inc_ref(v_name_476_);
v_extra_477_ = lean_ctor_get(v_r_472_, 3);
lean_inc(v_extra_477_);
lean_dec_ref(v_r_472_);
v_name_478_ = lean_ctor_get(v_name_476_, 0);
lean_inc(v_name_478_);
v_builder_479_ = lean_ctor_get_uint8(v_name_476_, sizeof(void*)*1 + 8);
v_phase_480_ = lean_ctor_get_uint8(v_name_476_, sizeof(void*)*1 + 9);
v_scope_481_ = lean_ctor_get_uint8(v_name_476_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_476_);
v___x_482_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__0));
v___x_483_ = lean_unbox_float(v_extra_477_);
lean_dec(v_extra_477_);
v___x_484_ = lp_aesop_Aesop_Percent_toHumanString(v___x_483_);
v___x_485_ = lean_string_append(v___x_482_, v___x_484_);
lean_dec_ref(v___x_484_);
v___x_486_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__1));
v___x_487_ = lean_string_append(v___x_485_, v___x_486_);
switch(v_phase_480_)
{
case 0:
{
lean_object* v___x_521_; 
v___x_521_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__13));
v___y_510_ = v___x_521_;
goto v___jp_509_;
}
case 1:
{
lean_object* v___x_522_; 
v___x_522_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__14));
v___y_510_ = v___x_522_;
goto v___jp_509_;
}
default: 
{
lean_object* v___x_523_; 
v___x_523_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__15));
v___y_510_ = v___x_523_;
goto v___jp_509_;
}
}
v___jp_488_:
{
lean_object* v___x_492_; lean_object* v___x_493_; uint8_t v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_499_; 
v___x_492_ = lean_string_append(v___y_490_, v___y_491_);
v___x_493_ = lean_string_append(v___x_492_, v___y_489_);
v___x_494_ = 1;
v___x_495_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_478_, v___x_494_);
v___x_496_ = lean_string_append(v___x_493_, v___x_495_);
lean_dec_ref(v___x_495_);
v___x_497_ = lean_string_append(v___x_487_, v___x_496_);
lean_dec_ref(v___x_496_);
if (v_isShared_475_ == 0)
{
lean_ctor_set_tag(v___x_474_, 3);
lean_ctor_set(v___x_474_, 0, v___x_497_);
v___x_499_ = v___x_474_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_497_);
v___x_499_ = v_reuseFailAlloc_500_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
return v___x_499_;
}
}
v___jp_501_:
{
lean_object* v___x_505_; lean_object* v___x_506_; 
v___x_505_ = lean_string_append(v___y_503_, v___y_504_);
v___x_506_ = lean_string_append(v___x_505_, v___y_502_);
if (v_scope_481_ == 0)
{
lean_object* v___x_507_; 
v___x_507_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__2));
v___y_489_ = v___y_502_;
v___y_490_ = v___x_506_;
v___y_491_ = v___x_507_;
goto v___jp_488_;
}
else
{
lean_object* v___x_508_; 
v___x_508_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__3));
v___y_489_ = v___y_502_;
v___y_490_ = v___x_506_;
v___y_491_ = v___x_508_;
goto v___jp_488_;
}
}
v___jp_509_:
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__4));
lean_inc_ref(v___y_510_);
v___x_512_ = lean_string_append(v___y_510_, v___x_511_);
switch(v_builder_479_)
{
case 0:
{
lean_object* v___x_513_; 
v___x_513_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__5));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_513_;
goto v___jp_501_;
}
case 1:
{
lean_object* v___x_514_; 
v___x_514_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__6));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_514_;
goto v___jp_501_;
}
case 2:
{
lean_object* v___x_515_; 
v___x_515_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__7));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_515_;
goto v___jp_501_;
}
case 3:
{
lean_object* v___x_516_; 
v___x_516_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__8));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_516_;
goto v___jp_501_;
}
case 4:
{
lean_object* v___x_517_; 
v___x_517_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__9));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_517_;
goto v___jp_501_;
}
case 5:
{
lean_object* v___x_518_; 
v___x_518_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__10));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_518_;
goto v___jp_501_;
}
case 6:
{
lean_object* v___x_519_; 
v___x_519_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__11));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_519_;
goto v___jp_501_;
}
default: 
{
lean_object* v___x_520_; 
v___x_520_ = ((lean_object*)(lp_aesop_Aesop_instToStringNormRule___lam__0___closed__12));
v___y_502_ = v___x_511_;
v___y_503_ = v___x_512_;
v___y_504_ = v___x_520_;
goto v___jp_501_;
}
}
}
}
}
}
}
LEAN_EXPORT double lp_aesop_Aesop_RegularRule_successProbability(lean_object* v_x_527_){
_start:
{
if (lean_obj_tag(v_x_527_) == 0)
{
double v___x_528_; 
v___x_528_ = lp_aesop_Aesop_Percent_hundred;
return v___x_528_;
}
else
{
lean_object* v_r_529_; lean_object* v_extra_530_; double v___x_531_; 
v_r_529_ = lean_ctor_get(v_x_527_, 0);
v_extra_530_ = lean_ctor_get(v_r_529_, 3);
v___x_531_ = lean_unbox_float(v_extra_530_);
return v___x_531_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_successProbability___boxed(lean_object* v_x_532_){
_start:
{
double v_res_533_; lean_object* v_r_534_; 
v_res_533_ = lp_aesop_Aesop_RegularRule_successProbability(v_x_532_);
lean_dec_ref(v_x_532_);
v_r_534_ = lean_box_float(v_res_533_);
return v_r_534_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RegularRule_isSafe(lean_object* v_x_535_){
_start:
{
if (lean_obj_tag(v_x_535_) == 0)
{
uint8_t v___x_536_; 
v___x_536_ = 1;
return v___x_536_;
}
else
{
uint8_t v___x_537_; 
v___x_537_ = 0;
return v___x_537_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_isSafe___boxed(lean_object* v_x_538_){
_start:
{
uint8_t v_res_539_; lean_object* v_r_540_; 
v_res_539_ = lp_aesop_Aesop_RegularRule_isSafe(v_x_538_);
lean_dec_ref(v_x_538_);
v_r_540_ = lean_box(v_res_539_);
return v_r_540_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RegularRule_isUnsafe(lean_object* v_x_541_){
_start:
{
if (lean_obj_tag(v_x_541_) == 0)
{
uint8_t v___x_542_; 
v___x_542_ = 0;
return v___x_542_;
}
else
{
uint8_t v___x_543_; 
v___x_543_ = 1;
return v___x_543_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_isUnsafe___boxed(lean_object* v_x_544_){
_start:
{
uint8_t v_res_545_; lean_object* v_r_546_; 
v_res_545_ = lp_aesop_Aesop_RegularRule_isUnsafe(v_x_544_);
lean_dec_ref(v_x_544_);
v_r_546_ = lean_box(v_res_545_);
return v_r_546_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_withRule___redArg(lean_object* v_f_547_, lean_object* v_x_548_){
_start:
{
lean_object* v_r_549_; lean_object* v___x_550_; 
v_r_549_ = lean_ctor_get(v_x_548_, 0);
lean_inc_ref(v_r_549_);
lean_dec_ref(v_x_548_);
v___x_550_ = lean_apply_2(v_f_547_, lean_box(0), v_r_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_withRule(lean_object* v_00_u03b2_551_, lean_object* v_f_552_, lean_object* v_x_553_){
_start:
{
lean_object* v_r_554_; lean_object* v___x_555_; 
v_r_554_ = lean_ctor_get(v_x_553_, 0);
lean_inc_ref(v_r_554_);
lean_dec_ref(v_x_553_);
v___x_555_ = lean_apply_2(v_f_552_, lean_box(0), v_r_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_name(lean_object* v_r_556_){
_start:
{
lean_object* v_r_557_; lean_object* v_name_558_; 
v_r_557_ = lean_ctor_get(v_r_556_, 0);
v_name_558_ = lean_ctor_get(v_r_557_, 0);
lean_inc_ref(v_name_558_);
return v_name_558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_name___boxed(lean_object* v_r_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_aesop_Aesop_RegularRule_name(v_r_559_);
lean_dec_ref(v_r_559_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_indexingMode(lean_object* v_r_561_){
_start:
{
lean_object* v_r_562_; lean_object* v_indexingMode_563_; 
v_r_562_ = lean_ctor_get(v_r_561_, 0);
v_indexingMode_563_ = lean_ctor_get(v_r_562_, 1);
lean_inc(v_indexingMode_563_);
return v_indexingMode_563_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_indexingMode___boxed(lean_object* v_r_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_aesop_Aesop_RegularRule_indexingMode(v_r_564_);
lean_dec_ref(v_r_564_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_tac(lean_object* v_r_566_){
_start:
{
lean_object* v_r_567_; lean_object* v_tac_568_; 
v_r_567_ = lean_ctor_get(v_r_566_, 0);
v_tac_568_ = lean_ctor_get(v_r_567_, 4);
lean_inc(v_tac_568_);
return v_tac_568_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RegularRule_tac___boxed(lean_object* v_r_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_aesop_Aesop_RegularRule_tac(v_r_569_);
lean_dec_ref(v_r_569_);
return v_res_570_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1(void){
_start:
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; 
v___x_573_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__0));
v___x_574_ = lp_aesop_Aesop_instInhabitedRuleName_default;
v___x_575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_575_, 0, v___x_574_);
lean_ctor_set(v___x_575_, 1, v___x_573_);
return v___x_575_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpRule_default(void){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1, &lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedNormSimpRule_default___closed__1);
return v___x_576_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpRule(void){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_aesop_Aesop_instInhabitedNormSimpRule_default;
return v___x_577_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_NormSimpRule_instBEq___lam__0(lean_object* v_r_578_, lean_object* v_s_579_){
_start:
{
lean_object* v_name_580_; lean_object* v_name_581_; lean_object* v_name_582_; uint8_t v_builder_583_; uint8_t v_phase_584_; uint8_t v_scope_585_; uint64_t v_hash_586_; lean_object* v_name_587_; uint8_t v_builder_588_; uint8_t v_phase_589_; uint8_t v_scope_590_; uint64_t v_hash_591_; uint8_t v___x_592_; 
v_name_580_ = lean_ctor_get(v_r_578_, 0);
v_name_581_ = lean_ctor_get(v_s_579_, 0);
v_name_582_ = lean_ctor_get(v_name_580_, 0);
v_builder_583_ = lean_ctor_get_uint8(v_name_580_, sizeof(void*)*1 + 8);
v_phase_584_ = lean_ctor_get_uint8(v_name_580_, sizeof(void*)*1 + 9);
v_scope_585_ = lean_ctor_get_uint8(v_name_580_, sizeof(void*)*1 + 10);
v_hash_586_ = lean_ctor_get_uint64(v_name_580_, sizeof(void*)*1);
v_name_587_ = lean_ctor_get(v_name_581_, 0);
v_builder_588_ = lean_ctor_get_uint8(v_name_581_, sizeof(void*)*1 + 8);
v_phase_589_ = lean_ctor_get_uint8(v_name_581_, sizeof(void*)*1 + 9);
v_scope_590_ = lean_ctor_get_uint8(v_name_581_, sizeof(void*)*1 + 10);
v_hash_591_ = lean_ctor_get_uint64(v_name_581_, sizeof(void*)*1);
v___x_592_ = lean_uint64_dec_eq(v_hash_586_, v_hash_591_);
if (v___x_592_ == 0)
{
return v___x_592_;
}
else
{
uint8_t v___x_593_; 
v___x_593_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_583_, v_builder_588_);
if (v___x_593_ == 0)
{
return v___x_593_;
}
else
{
uint8_t v___x_594_; 
v___x_594_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_584_, v_phase_589_);
if (v___x_594_ == 0)
{
return v___x_594_;
}
else
{
uint8_t v___x_595_; 
v___x_595_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_585_, v_scope_590_);
if (v___x_595_ == 0)
{
return v___x_595_;
}
else
{
uint8_t v___x_596_; 
v___x_596_ = lean_name_eq(v_name_582_, v_name_587_);
return v___x_596_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSimpRule_instBEq___lam__0___boxed(lean_object* v_r_597_, lean_object* v_s_598_){
_start:
{
uint8_t v_res_599_; lean_object* v_r_600_; 
v_res_599_ = lp_aesop_Aesop_NormSimpRule_instBEq___lam__0(v_r_597_, v_s_598_);
lean_dec_ref(v_s_598_);
lean_dec_ref(v_r_597_);
v_r_600_ = lean_box(v_res_599_);
return v_r_600_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_NormSimpRule_instHashable___lam__0(lean_object* v_r_603_){
_start:
{
lean_object* v_name_604_; uint64_t v_hash_605_; 
v_name_604_ = lean_ctor_get(v_r_603_, 0);
v_hash_605_ = lean_ctor_get_uint64(v_name_604_, sizeof(void*)*1);
return v_hash_605_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_NormSimpRule_instHashable___lam__0___boxed(lean_object* v_r_606_){
_start:
{
uint64_t v_res_607_; lean_object* v_r_608_; 
v_res_607_ = lp_aesop_Aesop_NormSimpRule_instHashable___lam__0(v_r_606_);
lean_dec_ref(v_r_606_);
v_r_608_ = lean_box_uint64(v_res_607_);
return v_r_608_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0(lean_object* v_r_u2081_616_, lean_object* v_r_u2082_617_){
_start:
{
lean_object* v_id_618_; lean_object* v_id_619_; uint8_t v___x_620_; 
v_id_618_ = lean_ctor_get(v_r_u2081_616_, 0);
v_id_619_ = lean_ctor_get(v_r_u2082_617_, 0);
v___x_620_ = lean_name_eq(v_id_618_, v_id_619_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0___boxed(lean_object* v_r_u2081_621_, lean_object* v_r_u2082_622_){
_start:
{
uint8_t v_res_623_; lean_object* v_r_624_; 
v_res_623_ = lp_aesop_Aesop_LocalNormSimpRule_instBEq___lam__0(v_r_u2081_621_, v_r_u2082_622_);
lean_dec_ref(v_r_u2082_622_);
lean_dec_ref(v_r_u2081_621_);
v_r_624_ = lean_box(v_res_623_);
return v_r_624_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0(lean_object* v_r_627_){
_start:
{
lean_object* v_id_628_; 
v_id_628_ = lean_ctor_get(v_r_627_, 0);
if (lean_obj_tag(v_id_628_) == 0)
{
uint64_t v___x_629_; 
v___x_629_ = 1723ULL;
return v___x_629_;
}
else
{
uint64_t v_hash_630_; 
v_hash_630_ = lean_ctor_get_uint64(v_id_628_, sizeof(void*)*2);
return v_hash_630_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0___boxed(lean_object* v_r_631_){
_start:
{
uint64_t v_res_632_; lean_object* v_r_633_; 
v_res_632_ = lp_aesop_Aesop_LocalNormSimpRule_instHashable___lam__0(v_r_631_);
lean_dec_ref(v_r_631_);
v_r_633_ = lean_box_uint64(v_res_632_);
return v_r_633_;
}
}
static uint64_t _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__0(void){
_start:
{
uint8_t v___x_636_; uint64_t v___x_637_; 
v___x_636_ = 5;
v___x_637_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_636_);
return v___x_637_;
}
}
static uint64_t _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__1(void){
_start:
{
uint8_t v___x_638_; uint64_t v___x_639_; 
v___x_638_ = 0;
v___x_639_ = lp_aesop_Aesop_instHashablePhaseName_hash(v___x_638_);
return v___x_639_;
}
}
static uint64_t _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__2(void){
_start:
{
uint8_t v___x_640_; uint64_t v___x_641_; 
v___x_640_ = 1;
v___x_641_ = lp_aesop_Aesop_instHashableScopeName_hash(v___x_640_);
return v___x_641_;
}
}
static uint64_t _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__3(void){
_start:
{
uint64_t v___x_642_; uint64_t v___x_643_; uint64_t v___x_644_; 
v___x_642_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__2, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__2_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__2);
v___x_643_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__1, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__1_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__1);
v___x_644_ = lean_uint64_mix_hash(v___x_643_, v___x_642_);
return v___x_644_;
}
}
static uint64_t _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__4(void){
_start:
{
uint64_t v___x_645_; uint64_t v___x_646_; uint64_t v___x_647_; 
v___x_645_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__3, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__3_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__3);
v___x_646_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__0, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__0_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__0);
v___x_647_ = lean_uint64_mix_hash(v___x_646_, v___x_645_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_name(lean_object* v_r_648_){
_start:
{
lean_object* v_id_649_; uint8_t v___x_650_; uint8_t v___x_651_; uint8_t v___x_652_; uint64_t v___y_654_; 
v_id_649_ = lean_ctor_get(v_r_648_, 0);
v___x_650_ = 5;
v___x_651_ = 0;
v___x_652_ = 1;
if (lean_obj_tag(v_id_649_) == 0)
{
uint64_t v___x_658_; 
v___x_658_ = 1723ULL;
v___y_654_ = v___x_658_;
goto v___jp_653_;
}
else
{
uint64_t v_hash_659_; 
v_hash_659_ = lean_ctor_get_uint64(v_id_649_, sizeof(void*)*2);
v___y_654_ = v_hash_659_;
goto v___jp_653_;
}
v___jp_653_:
{
uint64_t v___x_655_; uint64_t v___x_656_; lean_object* v___x_657_; 
v___x_655_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__4, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__4_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__4);
v___x_656_ = lean_uint64_mix_hash(v___y_654_, v___x_655_);
lean_inc(v_id_649_);
v___x_657_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_657_, 0, v_id_649_);
lean_ctor_set_uint8(v___x_657_, sizeof(void*)*1 + 8, v___x_650_);
lean_ctor_set_uint8(v___x_657_, sizeof(void*)*1 + 9, v___x_651_);
lean_ctor_set_uint8(v___x_657_, sizeof(void*)*1 + 10, v___x_652_);
lean_ctor_set_uint64(v___x_657_, sizeof(void*)*1, v___x_656_);
return v___x_657_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalNormSimpRule_name___boxed(lean_object* v_r_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_aesop_Aesop_LocalNormSimpRule_name(v_r_660_);
lean_dec_ref(v_r_660_);
return v_res_661_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnfoldRule_instBEq___lam__0(lean_object* v_r_667_, lean_object* v_s_668_){
_start:
{
lean_object* v_decl_669_; lean_object* v_decl_670_; uint8_t v___x_671_; 
v_decl_669_ = lean_ctor_get(v_r_667_, 0);
v_decl_670_ = lean_ctor_get(v_s_668_, 0);
v___x_671_ = lean_name_eq(v_decl_669_, v_decl_670_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_instBEq___lam__0___boxed(lean_object* v_r_672_, lean_object* v_s_673_){
_start:
{
uint8_t v_res_674_; lean_object* v_r_675_; 
v_res_674_ = lp_aesop_Aesop_UnfoldRule_instBEq___lam__0(v_r_672_, v_s_673_);
lean_dec_ref(v_s_673_);
lean_dec_ref(v_r_672_);
v_r_675_ = lean_box(v_res_674_);
return v_r_675_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_UnfoldRule_instHashable___lam__0(lean_object* v_r_678_){
_start:
{
lean_object* v_decl_679_; 
v_decl_679_ = lean_ctor_get(v_r_678_, 0);
if (lean_obj_tag(v_decl_679_) == 0)
{
uint64_t v___x_680_; 
v___x_680_ = 1723ULL;
return v___x_680_;
}
else
{
uint64_t v_hash_681_; 
v_hash_681_ = lean_ctor_get_uint64(v_decl_679_, sizeof(void*)*2);
return v_hash_681_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_instHashable___lam__0___boxed(lean_object* v_r_682_){
_start:
{
uint64_t v_res_683_; lean_object* v_r_684_; 
v_res_683_ = lp_aesop_Aesop_UnfoldRule_instHashable___lam__0(v_r_682_);
lean_dec_ref(v_r_682_);
v_r_684_ = lean_box_uint64(v_res_683_);
return v_r_684_;
}
}
static uint64_t _init_lp_aesop_Aesop_UnfoldRule_name___closed__0(void){
_start:
{
uint8_t v___x_687_; uint64_t v___x_688_; 
v___x_687_ = 7;
v___x_688_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_687_);
return v___x_688_;
}
}
static uint64_t _init_lp_aesop_Aesop_UnfoldRule_name___closed__1(void){
_start:
{
uint8_t v___x_689_; uint64_t v___x_690_; 
v___x_689_ = 0;
v___x_690_ = lp_aesop_Aesop_instHashableScopeName_hash(v___x_689_);
return v___x_690_;
}
}
static uint64_t _init_lp_aesop_Aesop_UnfoldRule_name___closed__2(void){
_start:
{
uint64_t v___x_691_; uint64_t v___x_692_; uint64_t v___x_693_; 
v___x_691_ = lean_uint64_once(&lp_aesop_Aesop_UnfoldRule_name___closed__1, &lp_aesop_Aesop_UnfoldRule_name___closed__1_once, _init_lp_aesop_Aesop_UnfoldRule_name___closed__1);
v___x_692_ = lean_uint64_once(&lp_aesop_Aesop_LocalNormSimpRule_name___closed__1, &lp_aesop_Aesop_LocalNormSimpRule_name___closed__1_once, _init_lp_aesop_Aesop_LocalNormSimpRule_name___closed__1);
v___x_693_ = lean_uint64_mix_hash(v___x_692_, v___x_691_);
return v___x_693_;
}
}
static uint64_t _init_lp_aesop_Aesop_UnfoldRule_name___closed__3(void){
_start:
{
uint64_t v___x_694_; uint64_t v___x_695_; uint64_t v___x_696_; 
v___x_694_ = lean_uint64_once(&lp_aesop_Aesop_UnfoldRule_name___closed__2, &lp_aesop_Aesop_UnfoldRule_name___closed__2_once, _init_lp_aesop_Aesop_UnfoldRule_name___closed__2);
v___x_695_ = lean_uint64_once(&lp_aesop_Aesop_UnfoldRule_name___closed__0, &lp_aesop_Aesop_UnfoldRule_name___closed__0_once, _init_lp_aesop_Aesop_UnfoldRule_name___closed__0);
v___x_696_ = lean_uint64_mix_hash(v___x_695_, v___x_694_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_name(lean_object* v_r_697_){
_start:
{
lean_object* v_decl_698_; uint8_t v___x_699_; uint8_t v___x_700_; uint8_t v___x_701_; uint64_t v___y_703_; 
v_decl_698_ = lean_ctor_get(v_r_697_, 0);
v___x_699_ = 7;
v___x_700_ = 0;
v___x_701_ = 0;
if (lean_obj_tag(v_decl_698_) == 0)
{
uint64_t v___x_707_; 
v___x_707_ = 1723ULL;
v___y_703_ = v___x_707_;
goto v___jp_702_;
}
else
{
uint64_t v_hash_708_; 
v_hash_708_ = lean_ctor_get_uint64(v_decl_698_, sizeof(void*)*2);
v___y_703_ = v_hash_708_;
goto v___jp_702_;
}
v___jp_702_:
{
uint64_t v___x_704_; uint64_t v___x_705_; lean_object* v___x_706_; 
v___x_704_ = lean_uint64_once(&lp_aesop_Aesop_UnfoldRule_name___closed__3, &lp_aesop_Aesop_UnfoldRule_name___closed__3_once, _init_lp_aesop_Aesop_UnfoldRule_name___closed__3);
v___x_705_ = lean_uint64_mix_hash(v___y_703_, v___x_704_);
lean_inc(v_decl_698_);
v___x_706_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_706_, 0, v_decl_698_);
lean_ctor_set_uint8(v___x_706_, sizeof(void*)*1 + 8, v___x_699_);
lean_ctor_set_uint8(v___x_706_, sizeof(void*)*1 + 9, v___x_700_);
lean_ctor_set_uint8(v___x_706_, sizeof(void*)*1 + 10, v___x_701_);
lean_ctor_set_uint64(v___x_706_, sizeof(void*)*1, v___x_705_);
return v___x_706_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnfoldRule_name___boxed(lean_object* v_r_709_){
_start:
{
lean_object* v_res_710_; 
v_res_710_ = lp_aesop_Aesop_UnfoldRule_name(v_r_709_);
lean_dec_ref(v_r_709_);
return v_res_710_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Rule(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedNormRuleInfo_default = _init_lp_aesop_Aesop_instInhabitedNormRuleInfo_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormRuleInfo_default);
lp_aesop_Aesop_instInhabitedNormRuleInfo = _init_lp_aesop_Aesop_instInhabitedNormRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormRuleInfo);
lp_aesop_Aesop_instLTNormRuleInfo = _init_lp_aesop_Aesop_instLTNormRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLTNormRuleInfo);
lp_aesop_Aesop_instLENormRuleInfo = _init_lp_aesop_Aesop_instLENormRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLENormRuleInfo);
lp_aesop_Aesop_defaultNormPenalty = _init_lp_aesop_Aesop_defaultNormPenalty();
lean_mark_persistent(lp_aesop_Aesop_defaultNormPenalty);
lp_aesop_Aesop_defaultSimpRulePriority = _init_lp_aesop_Aesop_defaultSimpRulePriority();
lean_mark_persistent(lp_aesop_Aesop_defaultSimpRulePriority);
lp_aesop_Aesop_instInhabitedSafety_default = _init_lp_aesop_Aesop_instInhabitedSafety_default();
lp_aesop_Aesop_instInhabitedSafety = _init_lp_aesop_Aesop_instInhabitedSafety();
lp_aesop_Aesop_instInhabitedSafeRuleInfo_default = _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedSafeRuleInfo_default);
lp_aesop_Aesop_instInhabitedSafeRuleInfo = _init_lp_aesop_Aesop_instInhabitedSafeRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedSafeRuleInfo);
lp_aesop_Aesop_instLTSafeRuleInfo = _init_lp_aesop_Aesop_instLTSafeRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLTSafeRuleInfo);
lp_aesop_Aesop_instLESafeRuleInfo = _init_lp_aesop_Aesop_instLESafeRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLESafeRuleInfo);
lp_aesop_Aesop_defaultSafePenalty = _init_lp_aesop_Aesop_defaultSafePenalty();
lean_mark_persistent(lp_aesop_Aesop_defaultSafePenalty);
lp_aesop_Aesop_instInhabitedUnsafeRuleInfo_default = _init_lp_aesop_Aesop_instInhabitedUnsafeRuleInfo_default();
lp_aesop_Aesop_instInhabitedUnsafeRuleInfo = _init_lp_aesop_Aesop_instInhabitedUnsafeRuleInfo();
lp_aesop_Aesop_instLTUnsafeRuleInfo = _init_lp_aesop_Aesop_instLTUnsafeRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLTUnsafeRuleInfo);
lp_aesop_Aesop_instLEUnsafeRuleInfo = _init_lp_aesop_Aesop_instLEUnsafeRuleInfo();
lean_mark_persistent(lp_aesop_Aesop_instLEUnsafeRuleInfo);
lp_aesop_Aesop_defaultSuccessProbability = _init_lp_aesop_Aesop_defaultSuccessProbability();
lp_aesop_Aesop_instInhabitedRegularRule_default = _init_lp_aesop_Aesop_instInhabitedRegularRule_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRegularRule_default);
lp_aesop_Aesop_instInhabitedRegularRule = _init_lp_aesop_Aesop_instInhabitedRegularRule();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRegularRule);
lp_aesop_Aesop_instInhabitedNormSimpRule_default = _init_lp_aesop_Aesop_instInhabitedNormSimpRule_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpRule_default);
lp_aesop_Aesop_instInhabitedNormSimpRule = _init_lp_aesop_Aesop_instInhabitedNormSimpRule();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpRule);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Rule(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Rule(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Rule(builtin);
}
#ifdef __cplusplus
}
#endif
