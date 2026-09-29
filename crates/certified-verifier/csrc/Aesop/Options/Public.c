// Lean compiler output
// Module: Aesop.Options.Public
// Imports: public import Init public meta import Init public import Lean.Data.Options
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
uint8_t l_Lean_Meta_instBEqTransparencyMode_beq(uint8_t, uint8_t);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Meta_instReprTransparencyMode_repr(uint8_t, lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedStrategy_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedStrategy;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqStrategy_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqStrategy_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqStrategy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqStrategy_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqStrategy___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqStrategy___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqStrategy = (const lean_object*)&lp_aesop_Aesop_instBEqStrategy___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instReprStrategy_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Aesop.Strategy.bestFirst"};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instReprStrategy_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__1 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instReprStrategy_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Aesop.Strategy.depthFirst"};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__2 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instReprStrategy_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__3 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instReprStrategy_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Aesop.Strategy.breadthFirst"};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__4 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_instReprStrategy_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__5 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy_repr___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_instReprStrategy_repr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__6;
static lean_once_cell_t lp_aesop_Aesop_instReprStrategy_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprStrategy_repr___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprStrategy_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprStrategy_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instReprStrategy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instReprStrategy_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprStrategy___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprStrategy___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instReprStrategy = (const lean_object*)&lp_aesop_Aesop_instReprStrategy___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedOptions_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 16, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(30) << 1) | 1)),((lean_object*)(((size_t)(200) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 2, 0, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_instInhabitedOptions_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedOptions_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedOptions = (const lean_object*)&lp_aesop_Aesop_instInhabitedOptions_default___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqOptions_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqOptions_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqOptions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqOptions_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqOptions___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqOptions___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqOptions = (const lean_object*)&lp_aesop_Aesop_instBEqOptions___closed__0_value;
static const lean_string_object lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__0 = (const lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__0_value)}};
static const lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__1 = (const lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__1_value;
static const lean_string_object lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__2 = (const lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__2_value)}};
static const lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__3 = (const lean_object*)&lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Nat_cast___at___00Aesop_instReprOptions_repr_spec__1(lean_object*);
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "strategy"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__9_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "maxRuleApplicationDepth"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__11_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "maxRuleApplications"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__13 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "maxGoals"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__16 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__17_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "maxNormIterations"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__18 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__19 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "maxSafePrefixRuleApplications"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__21 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__21_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__21_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__22 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "applyHypsTransparency"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__24 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__24_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__24_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__25 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__25_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "assumptionTransparency"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__27 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__27_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__27_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__28 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__28_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "destructProductsTransparency"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__30 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__30_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__30_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__31 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__31_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "introsTransparency\?"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__33 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__33_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__33_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__34 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__34_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "terminal"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__35 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__35_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__35_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__36 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__36_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "warnOnNonterminal"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__37 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__37_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__37_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__38 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__38_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "traceScript"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__39 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__39_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__39_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__40 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__40_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "enableSimp"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__42 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__42_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__42_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__43 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__43_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "useSimpAll"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__45 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__45_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__45_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__46 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__46_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "useDefaultSimpSet"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__47 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__47_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__47_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__48 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__48_value;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "enableUnfold"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__49 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__49_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__49_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__50 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__50_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51;
static const lean_string_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__52 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__52_value;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53;
static lean_once_cell_t lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__55 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__55_value;
static const lean_ctor_object lp_aesop_Aesop_instReprOptions_repr___redArg___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__52_value)}};
static const lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg___closed__56 = (const lean_object*)&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__56_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instReprOptions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instReprOptions_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instReprOptions___closed__0 = (const lean_object*)&lp_aesop_Aesop_instReprOptions___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instReprOptions = (const lean_object*)&lp_aesop_Aesop_instReprOptions___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "dev"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "dynamicStructuring"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 53, 184, 139, 133, 143, 235, 166)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(227, 220, 103, 132, 212, 7, 30, 123)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = "(aesop) Only for use by Aesop developers. Enables dynamic script structuring."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(108, 239, 216, 196, 143, 139, 148, 28)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(33, 143, 239, 215, 248, 96, 19, 207)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_dev_dynamicStructuring;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "optimizedDynamicStructuring"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 53, 184, 139, 133, 143, 235, 166)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(32, 190, 8, 244, 224, 5, 46, 34)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 138, .m_capacity = 138, .m_length = 137, .m_data = "(aesop) Only for use by Aesop developers. Uses static structuring instead of dynamic structuring if no metavariables appear in the proof."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(108, 239, 216, 196, 143, 139, 148, 28)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(178, 234, 51, 149, 70, 195, 108, 112)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_dev_optimizedDynamicStructuring;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "generateScript"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 53, 184, 139, 133, 143, 235, 166)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(159, 183, 79, 78, 86, 29, 205, 84)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 89, .m_capacity = 89, .m_length = 88, .m_data = "(aesop) Only for use by Aesop developers. Generates a script even if none was requested."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(108, 239, 216, 196, 143, 139, 148, 28)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(93, 132, 165, 228, 144, 123, 12, 66)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "statefulForward"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 53, 184, 139, 133, 143, 235, 166)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(225, 170, 211, 51, 245, 2, 239, 252)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 93, .m_capacity = 93, .m_length = 92, .m_data = "(aesop) Only for use by Aesop developers. Enables the new stateful forward reasoning engine."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(108, 239, 216, 196, 143, 139, 148, 28)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(35, 67, 224, 52, 27, 143, 109, 235)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_dev_statefulForward;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "warn"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "applyIff"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(93, 197, 109, 228, 67, 224, 106, 67)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(111, 54, 108, 165, 194, 173, 46, 59)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 87, .m_data = "(aesop) Warn when apply builder is applied to a rule with conclusion of the form A ↔ B."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(55, 230, 33, 19, 84, 61, 243, 182)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(13, 35, 85, 204, 128, 223, 129, 29)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_warn_applyIff;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "nonterminal"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(93, 197, 109, 228, 67, 224, 106, 67)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(55, 130, 166, 116, 34, 219, 30, 190)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = "(aesop) Warn when `aesop` does not close the goal, i.e. is used as a non-terminal tactic."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(55, 230, 33, 19, 84, 61, 243, 182)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(69, 47, 3, 111, 196, 84, 128, 100)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_warn_nonterminal;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "collectStats"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(206, 147, 139, 233, 144, 112, 97, 30)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 106, .m_capacity = 106, .m_length = 105, .m_data = "(aesop) Collect statistics about Aesop invocations. Use #aesop_stats to display the collected statistics."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(244, 34, 30, 52, 165, 199, 210, 78)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_collectStats;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "stats"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "file"};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(250, 184, 88, 2, 24, 26, 95, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(145, 133, 36, 19, 93, 20, 142, 0)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_string_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 134, .m_capacity = 134, .m_length = 133, .m_data = "(aesop) Write statistics about Aesop invocations to the given file in JSONL format. Each invocation adds one JSON record to the file."};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 12, 85, 226, 117, 27, 136, 102)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__0_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(32, 198, 239, 182, 89, 33, 91, 230)}};
static const lean_ctor_object lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(35, 3, 130, 135, 231, 89, 109, 103)}};
static const lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_ = (const lean_object*)&lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_aesop_stats_file;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorIdx(uint8_t v_x_1_){
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
uint8_t v_x_boxed_6_; lean_object* v_res_7_; 
v_x_boxed_6_ = lean_unbox(v_x_5_);
v_res_7_ = lp_aesop_Aesop_Strategy_ctorIdx(v_x_boxed_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___redArg(lean_object* v_k_8_){
_start:
{
lean_inc(v_k_8_);
return v_k_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___redArg___boxed(lean_object* v_k_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop_Aesop_Strategy_ctorElim___redArg(v_k_9_);
lean_dec(v_k_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, uint8_t v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_inc(v_k_15_);
return v_k_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
uint8_t v_t_boxed_21_; lean_object* v_res_22_; 
v_t_boxed_21_ = lean_unbox(v_t_18_);
v_res_22_ = lp_aesop_Aesop_Strategy_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_boxed_21_, v_h_19_, v_k_20_);
lean_dec(v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___redArg(lean_object* v_bestFirst_23_){
_start:
{
lean_inc(v_bestFirst_23_);
return v_bestFirst_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___redArg___boxed(lean_object* v_bestFirst_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_Strategy_bestFirst_elim___redArg(v_bestFirst_24_);
lean_dec(v_bestFirst_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim(lean_object* v_motive_26_, uint8_t v_t_27_, lean_object* v_h_28_, lean_object* v_bestFirst_29_){
_start:
{
lean_inc(v_bestFirst_29_);
return v_bestFirst_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_bestFirst_elim___boxed(lean_object* v_motive_30_, lean_object* v_t_31_, lean_object* v_h_32_, lean_object* v_bestFirst_33_){
_start:
{
uint8_t v_t_boxed_34_; lean_object* v_res_35_; 
v_t_boxed_34_ = lean_unbox(v_t_31_);
v_res_35_ = lp_aesop_Aesop_Strategy_bestFirst_elim(v_motive_30_, v_t_boxed_34_, v_h_32_, v_bestFirst_33_);
lean_dec(v_bestFirst_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___redArg(lean_object* v_depthFirst_36_){
_start:
{
lean_inc(v_depthFirst_36_);
return v_depthFirst_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___redArg___boxed(lean_object* v_depthFirst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Aesop_Strategy_depthFirst_elim___redArg(v_depthFirst_37_);
lean_dec(v_depthFirst_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim(lean_object* v_motive_39_, uint8_t v_t_40_, lean_object* v_h_41_, lean_object* v_depthFirst_42_){
_start:
{
lean_inc(v_depthFirst_42_);
return v_depthFirst_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_depthFirst_elim___boxed(lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_depthFirst_46_){
_start:
{
uint8_t v_t_boxed_47_; lean_object* v_res_48_; 
v_t_boxed_47_ = lean_unbox(v_t_44_);
v_res_48_ = lp_aesop_Aesop_Strategy_depthFirst_elim(v_motive_43_, v_t_boxed_47_, v_h_45_, v_depthFirst_46_);
lean_dec(v_depthFirst_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___redArg(lean_object* v_breadthFirst_49_){
_start:
{
lean_inc(v_breadthFirst_49_);
return v_breadthFirst_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___redArg___boxed(lean_object* v_breadthFirst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_aesop_Aesop_Strategy_breadthFirst_elim___redArg(v_breadthFirst_50_);
lean_dec(v_breadthFirst_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim(lean_object* v_motive_52_, uint8_t v_t_53_, lean_object* v_h_54_, lean_object* v_breadthFirst_55_){
_start:
{
lean_inc(v_breadthFirst_55_);
return v_breadthFirst_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Strategy_breadthFirst_elim___boxed(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_breadthFirst_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_aesop_Aesop_Strategy_breadthFirst_elim(v_motive_56_, v_t_boxed_60_, v_h_58_, v_breadthFirst_59_);
lean_dec(v_breadthFirst_59_);
return v_res_61_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedStrategy_default(void){
_start:
{
uint8_t v___x_62_; 
v___x_62_ = 0;
return v___x_62_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedStrategy(void){
_start:
{
uint8_t v___x_63_; 
v___x_63_ = 0;
return v___x_63_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqStrategy_beq(uint8_t v_x_64_, uint8_t v_y_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_66_ = lp_aesop_Aesop_Strategy_ctorIdx(v_x_64_);
v___x_67_ = lp_aesop_Aesop_Strategy_ctorIdx(v_y_65_);
v___x_68_ = lean_nat_dec_eq(v___x_66_, v___x_67_);
lean_dec(v___x_67_);
lean_dec(v___x_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqStrategy_beq___boxed(lean_object* v_x_69_, lean_object* v_y_70_){
_start:
{
uint8_t v_x_17__boxed_71_; uint8_t v_y_18__boxed_72_; uint8_t v_res_73_; lean_object* v_r_74_; 
v_x_17__boxed_71_ = lean_unbox(v_x_69_);
v_y_18__boxed_72_ = lean_unbox(v_y_70_);
v_res_73_ = lp_aesop_Aesop_instBEqStrategy_beq(v_x_17__boxed_71_, v_y_18__boxed_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprStrategy_repr___closed__6(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(2u);
v___x_87_ = lean_nat_to_int(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprStrategy_repr___closed__7(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_nat_to_int(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprStrategy_repr(uint8_t v_x_90_, lean_object* v_prec_91_){
_start:
{
lean_object* v___y_93_; lean_object* v___y_100_; lean_object* v___y_107_; 
switch(v_x_90_)
{
case 0:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = lean_unsigned_to_nat(1024u);
v___x_114_ = lean_nat_dec_le(v___x_113_, v_prec_91_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; 
v___x_115_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__6, &lp_aesop_Aesop_instReprStrategy_repr___closed__6_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__6);
v___y_93_ = v___x_115_;
goto v___jp_92_;
}
else
{
lean_object* v___x_116_; 
v___x_116_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__7, &lp_aesop_Aesop_instReprStrategy_repr___closed__7_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__7);
v___y_93_ = v___x_116_;
goto v___jp_92_;
}
}
case 1:
{
lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_117_ = lean_unsigned_to_nat(1024u);
v___x_118_ = lean_nat_dec_le(v___x_117_, v_prec_91_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; 
v___x_119_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__6, &lp_aesop_Aesop_instReprStrategy_repr___closed__6_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__6);
v___y_100_ = v___x_119_;
goto v___jp_99_;
}
else
{
lean_object* v___x_120_; 
v___x_120_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__7, &lp_aesop_Aesop_instReprStrategy_repr___closed__7_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__7);
v___y_100_ = v___x_120_;
goto v___jp_99_;
}
}
default: 
{
lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_121_ = lean_unsigned_to_nat(1024u);
v___x_122_ = lean_nat_dec_le(v___x_121_, v_prec_91_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; 
v___x_123_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__6, &lp_aesop_Aesop_instReprStrategy_repr___closed__6_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__6);
v___y_107_ = v___x_123_;
goto v___jp_106_;
}
else
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_aesop_Aesop_instReprStrategy_repr___closed__7, &lp_aesop_Aesop_instReprStrategy_repr___closed__7_once, _init_lp_aesop_Aesop_instReprStrategy_repr___closed__7);
v___y_107_ = v___x_124_;
goto v___jp_106_;
}
}
}
v___jp_92_:
{
lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_94_ = ((lean_object*)(lp_aesop_Aesop_instReprStrategy_repr___closed__1));
lean_inc(v___y_93_);
v___x_95_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_95_, 0, v___y_93_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = 0;
v___x_97_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_97_, 0, v___x_95_);
lean_ctor_set_uint8(v___x_97_, sizeof(void*)*1, v___x_96_);
v___x_98_ = l_Repr_addAppParen(v___x_97_, v_prec_91_);
return v___x_98_;
}
v___jp_99_:
{
lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_101_ = ((lean_object*)(lp_aesop_Aesop_instReprStrategy_repr___closed__3));
lean_inc(v___y_100_);
v___x_102_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_102_, 0, v___y_100_);
lean_ctor_set(v___x_102_, 1, v___x_101_);
v___x_103_ = 0;
v___x_104_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_104_, 0, v___x_102_);
lean_ctor_set_uint8(v___x_104_, sizeof(void*)*1, v___x_103_);
v___x_105_ = l_Repr_addAppParen(v___x_104_, v_prec_91_);
return v___x_105_;
}
v___jp_106_:
{
lean_object* v___x_108_; lean_object* v___x_109_; uint8_t v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_108_ = ((lean_object*)(lp_aesop_Aesop_instReprStrategy_repr___closed__5));
lean_inc(v___y_107_);
v___x_109_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_109_, 0, v___y_107_);
lean_ctor_set(v___x_109_, 1, v___x_108_);
v___x_110_ = 0;
v___x_111_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_111_, 0, v___x_109_);
lean_ctor_set_uint8(v___x_111_, sizeof(void*)*1, v___x_110_);
v___x_112_ = l_Repr_addAppParen(v___x_111_, v_prec_91_);
return v___x_112_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprStrategy_repr___boxed(lean_object* v_x_125_, lean_object* v_prec_126_){
_start:
{
uint8_t v_x_177__boxed_127_; lean_object* v_res_128_; 
v_x_177__boxed_127_ = lean_unbox(v_x_125_);
v_res_128_ = lp_aesop_Aesop_instReprStrategy_repr(v_x_177__boxed_127_, v_prec_126_);
lean_dec(v_prec_126_);
return v_res_128_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0(lean_object* v_x_145_, lean_object* v_x_146_){
_start:
{
if (lean_obj_tag(v_x_145_) == 0)
{
if (lean_obj_tag(v_x_146_) == 0)
{
uint8_t v___x_147_; 
v___x_147_ = 1;
return v___x_147_;
}
else
{
uint8_t v___x_148_; 
v___x_148_ = 0;
return v___x_148_;
}
}
else
{
if (lean_obj_tag(v_x_146_) == 0)
{
uint8_t v___x_149_; 
v___x_149_ = 0;
return v___x_149_;
}
else
{
lean_object* v_val_150_; lean_object* v_val_151_; uint8_t v___x_152_; uint8_t v___x_153_; uint8_t v___x_154_; 
v_val_150_ = lean_ctor_get(v_x_145_, 0);
v_val_151_ = lean_ctor_get(v_x_146_, 0);
v___x_152_ = lean_unbox(v_val_150_);
v___x_153_ = lean_unbox(v_val_151_);
v___x_154_ = l_Lean_Meta_instBEqTransparencyMode_beq(v___x_152_, v___x_153_);
return v___x_154_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0___boxed(lean_object* v_x_155_, lean_object* v_x_156_){
_start:
{
uint8_t v_res_157_; lean_object* v_r_158_; 
v_res_157_ = lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0(v_x_155_, v_x_156_);
lean_dec(v_x_156_);
lean_dec(v_x_155_);
v_r_158_ = lean_box(v_res_157_);
return v_r_158_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqOptions_beq(lean_object* v_x_159_, lean_object* v_x_160_){
_start:
{
uint8_t v_strategy_161_; lean_object* v_maxRuleApplicationDepth_162_; lean_object* v_maxRuleApplications_163_; lean_object* v_maxGoals_164_; lean_object* v_maxNormIterations_165_; lean_object* v_maxSafePrefixRuleApplications_166_; uint8_t v_applyHypsTransparency_167_; uint8_t v_assumptionTransparency_168_; uint8_t v_destructProductsTransparency_169_; lean_object* v_introsTransparency_x3f_170_; uint8_t v_terminal_171_; uint8_t v_warnOnNonterminal_172_; uint8_t v_traceScript_173_; uint8_t v_enableSimp_174_; uint8_t v_useSimpAll_175_; uint8_t v_useDefaultSimpSet_176_; uint8_t v_enableUnfold_177_; uint8_t v_strategy_178_; lean_object* v_maxRuleApplicationDepth_179_; lean_object* v_maxRuleApplications_180_; lean_object* v_maxGoals_181_; lean_object* v_maxNormIterations_182_; lean_object* v_maxSafePrefixRuleApplications_183_; uint8_t v_applyHypsTransparency_184_; uint8_t v_assumptionTransparency_185_; uint8_t v_destructProductsTransparency_186_; lean_object* v_introsTransparency_x3f_187_; uint8_t v_terminal_188_; uint8_t v_warnOnNonterminal_189_; uint8_t v_traceScript_190_; uint8_t v_enableSimp_191_; uint8_t v_useSimpAll_192_; uint8_t v_useDefaultSimpSet_193_; uint8_t v_enableUnfold_194_; uint8_t v___y_196_; uint8_t v___y_198_; uint8_t v___y_200_; uint8_t v___y_202_; uint8_t v___y_204_; uint8_t v___y_206_; uint8_t v___x_207_; 
v_strategy_161_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6);
v_maxRuleApplicationDepth_162_ = lean_ctor_get(v_x_159_, 0);
v_maxRuleApplications_163_ = lean_ctor_get(v_x_159_, 1);
v_maxGoals_164_ = lean_ctor_get(v_x_159_, 2);
v_maxNormIterations_165_ = lean_ctor_get(v_x_159_, 3);
v_maxSafePrefixRuleApplications_166_ = lean_ctor_get(v_x_159_, 4);
v_applyHypsTransparency_167_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 1);
v_assumptionTransparency_168_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 2);
v_destructProductsTransparency_169_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 3);
v_introsTransparency_x3f_170_ = lean_ctor_get(v_x_159_, 5);
v_terminal_171_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 4);
v_warnOnNonterminal_172_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 5);
v_traceScript_173_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 6);
v_enableSimp_174_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 7);
v_useSimpAll_175_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 8);
v_useDefaultSimpSet_176_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 9);
v_enableUnfold_177_ = lean_ctor_get_uint8(v_x_159_, sizeof(void*)*6 + 10);
v_strategy_178_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6);
v_maxRuleApplicationDepth_179_ = lean_ctor_get(v_x_160_, 0);
v_maxRuleApplications_180_ = lean_ctor_get(v_x_160_, 1);
v_maxGoals_181_ = lean_ctor_get(v_x_160_, 2);
v_maxNormIterations_182_ = lean_ctor_get(v_x_160_, 3);
v_maxSafePrefixRuleApplications_183_ = lean_ctor_get(v_x_160_, 4);
v_applyHypsTransparency_184_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 1);
v_assumptionTransparency_185_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 2);
v_destructProductsTransparency_186_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 3);
v_introsTransparency_x3f_187_ = lean_ctor_get(v_x_160_, 5);
v_terminal_188_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 4);
v_warnOnNonterminal_189_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 5);
v_traceScript_190_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 6);
v_enableSimp_191_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 7);
v_useSimpAll_192_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 8);
v_useDefaultSimpSet_193_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 9);
v_enableUnfold_194_ = lean_ctor_get_uint8(v_x_160_, sizeof(void*)*6 + 10);
v___x_207_ = lp_aesop_Aesop_instBEqStrategy_beq(v_strategy_161_, v_strategy_178_);
if (v___x_207_ == 0)
{
return v___x_207_;
}
else
{
uint8_t v___x_208_; 
v___x_208_ = lean_nat_dec_eq(v_maxRuleApplicationDepth_162_, v_maxRuleApplicationDepth_179_);
if (v___x_208_ == 0)
{
return v___x_208_;
}
else
{
uint8_t v___x_209_; 
v___x_209_ = lean_nat_dec_eq(v_maxRuleApplications_163_, v_maxRuleApplications_180_);
if (v___x_209_ == 0)
{
return v___x_209_;
}
else
{
uint8_t v___x_210_; 
v___x_210_ = lean_nat_dec_eq(v_maxGoals_164_, v_maxGoals_181_);
if (v___x_210_ == 0)
{
return v___x_210_;
}
else
{
uint8_t v___x_211_; 
v___x_211_ = lean_nat_dec_eq(v_maxNormIterations_165_, v_maxNormIterations_182_);
if (v___x_211_ == 0)
{
return v___x_211_;
}
else
{
uint8_t v___x_212_; 
v___x_212_ = lean_nat_dec_eq(v_maxSafePrefixRuleApplications_166_, v_maxSafePrefixRuleApplications_183_);
if (v___x_212_ == 0)
{
return v___x_212_;
}
else
{
uint8_t v___x_213_; 
v___x_213_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_applyHypsTransparency_167_, v_applyHypsTransparency_184_);
if (v___x_213_ == 0)
{
return v___x_213_;
}
else
{
uint8_t v___x_214_; 
v___x_214_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_assumptionTransparency_168_, v_assumptionTransparency_185_);
if (v___x_214_ == 0)
{
return v___x_214_;
}
else
{
uint8_t v___x_215_; 
v___x_215_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_destructProductsTransparency_169_, v_destructProductsTransparency_186_);
if (v___x_215_ == 0)
{
return v___x_215_;
}
else
{
uint8_t v___x_216_; 
v___x_216_ = lp_aesop_Option_instBEq_beq___at___00Aesop_instBEqOptions_beq_spec__0(v_introsTransparency_x3f_170_, v_introsTransparency_x3f_187_);
if (v___x_216_ == 0)
{
return v___x_216_;
}
else
{
if (v_terminal_171_ == 0)
{
if (v_terminal_188_ == 0)
{
v___y_206_ = v___x_216_;
goto v___jp_205_;
}
else
{
return v_terminal_171_;
}
}
else
{
v___y_206_ = v_terminal_188_;
goto v___jp_205_;
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
v___jp_195_:
{
if (v_enableUnfold_177_ == 0)
{
if (v_enableUnfold_194_ == 0)
{
return v___y_196_;
}
else
{
return v_enableUnfold_177_;
}
}
else
{
return v_enableUnfold_194_;
}
}
v___jp_197_:
{
if (v_useDefaultSimpSet_176_ == 0)
{
if (v_useDefaultSimpSet_193_ == 0)
{
v___y_196_ = v___y_198_;
goto v___jp_195_;
}
else
{
return v_useDefaultSimpSet_176_;
}
}
else
{
if (v_useDefaultSimpSet_193_ == 0)
{
return v_useDefaultSimpSet_193_;
}
else
{
v___y_196_ = v_useDefaultSimpSet_193_;
goto v___jp_195_;
}
}
}
v___jp_199_:
{
if (v_useSimpAll_175_ == 0)
{
if (v_useSimpAll_192_ == 0)
{
v___y_198_ = v___y_200_;
goto v___jp_197_;
}
else
{
return v_useSimpAll_175_;
}
}
else
{
if (v_useSimpAll_192_ == 0)
{
return v_useSimpAll_192_;
}
else
{
v___y_198_ = v_useSimpAll_192_;
goto v___jp_197_;
}
}
}
v___jp_201_:
{
if (v_enableSimp_174_ == 0)
{
if (v_enableSimp_191_ == 0)
{
v___y_200_ = v___y_202_;
goto v___jp_199_;
}
else
{
return v_enableSimp_174_;
}
}
else
{
if (v_enableSimp_191_ == 0)
{
return v_enableSimp_191_;
}
else
{
v___y_200_ = v_enableSimp_191_;
goto v___jp_199_;
}
}
}
v___jp_203_:
{
if (v_traceScript_173_ == 0)
{
if (v_traceScript_190_ == 0)
{
v___y_202_ = v___y_204_;
goto v___jp_201_;
}
else
{
return v_traceScript_173_;
}
}
else
{
if (v_traceScript_190_ == 0)
{
return v_traceScript_190_;
}
else
{
v___y_202_ = v_traceScript_190_;
goto v___jp_201_;
}
}
}
v___jp_205_:
{
if (v___y_206_ == 0)
{
return v___y_206_;
}
else
{
if (v_warnOnNonterminal_172_ == 0)
{
if (v_warnOnNonterminal_189_ == 0)
{
v___y_204_ = v___y_206_;
goto v___jp_203_;
}
else
{
return v_warnOnNonterminal_172_;
}
}
else
{
if (v_warnOnNonterminal_189_ == 0)
{
return v_warnOnNonterminal_189_;
}
else
{
v___y_204_ = v_warnOnNonterminal_189_;
goto v___jp_203_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqOptions_beq___boxed(lean_object* v_x_217_, lean_object* v_x_218_){
_start:
{
uint8_t v_res_219_; lean_object* v_r_220_; 
v_res_219_ = lp_aesop_Aesop_instBEqOptions_beq(v_x_217_, v_x_218_);
lean_dec_ref(v_x_218_);
lean_dec_ref(v_x_217_);
v_r_220_ = lean_box(v_res_219_);
return v_r_220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0(lean_object* v_x_229_, lean_object* v_x_230_){
_start:
{
if (lean_obj_tag(v_x_229_) == 0)
{
lean_object* v___x_231_; 
v___x_231_ = ((lean_object*)(lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__1));
return v___x_231_;
}
else
{
lean_object* v_val_232_; lean_object* v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v_val_232_ = lean_ctor_get(v_x_229_, 0);
v___x_233_ = ((lean_object*)(lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___closed__3));
v___x_234_ = lean_unsigned_to_nat(1024u);
v___x_235_ = lean_unbox(v_val_232_);
v___x_236_ = l_Lean_Meta_instReprTransparencyMode_repr(v___x_235_, v___x_234_);
v___x_237_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_233_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = l_Repr_addAppParen(v___x_237_, v_x_230_);
return v___x_238_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0___boxed(lean_object* v_x_239_, lean_object* v_x_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0(v_x_239_, v_x_240_);
lean_dec(v_x_240_);
lean_dec(v_x_239_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Nat_cast___at___00Aesop_instReprOptions_repr_spec__1(lean_object* v_a_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lean_nat_to_int(v_a_242_);
return v___x_243_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_257_ = lean_unsigned_to_nat(12u);
v___x_258_ = lean_nat_to_int(v___x_257_);
return v___x_258_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = lean_unsigned_to_nat(27u);
v___x_266_ = lean_nat_to_int(v___x_265_);
return v___x_266_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = lean_unsigned_to_nat(23u);
v___x_271_ = lean_nat_to_int(v___x_270_);
return v___x_271_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = lean_unsigned_to_nat(21u);
v___x_279_ = lean_nat_to_int(v___x_278_);
return v___x_279_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23(void){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_283_ = lean_unsigned_to_nat(33u);
v___x_284_ = lean_nat_to_int(v___x_283_);
return v___x_284_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_288_ = lean_unsigned_to_nat(25u);
v___x_289_ = lean_nat_to_int(v___x_288_);
return v___x_289_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_293_ = lean_unsigned_to_nat(26u);
v___x_294_ = lean_nat_to_int(v___x_293_);
return v___x_294_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = lean_unsigned_to_nat(32u);
v___x_299_ = lean_nat_to_int(v___x_298_);
return v___x_299_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41(void){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_312_ = lean_unsigned_to_nat(15u);
v___x_313_ = lean_nat_to_int(v___x_312_);
return v___x_313_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = lean_unsigned_to_nat(14u);
v___x_318_ = lean_nat_to_int(v___x_317_);
return v___x_318_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = lean_unsigned_to_nat(16u);
v___x_329_ = lean_nat_to_int(v___x_328_);
return v___x_329_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_331_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__0));
v___x_332_ = lean_string_length(v___x_331_);
return v___x_332_;
}
}
static lean_object* _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54(void){
_start:
{
lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_333_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__53);
v___x_334_ = lean_nat_to_int(v___x_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr___redArg(lean_object* v_x_339_){
_start:
{
uint8_t v_strategy_340_; lean_object* v_maxRuleApplicationDepth_341_; lean_object* v_maxRuleApplications_342_; lean_object* v_maxGoals_343_; lean_object* v_maxNormIterations_344_; lean_object* v_maxSafePrefixRuleApplications_345_; uint8_t v_applyHypsTransparency_346_; uint8_t v_assumptionTransparency_347_; uint8_t v_destructProductsTransparency_348_; lean_object* v_introsTransparency_x3f_349_; uint8_t v_terminal_350_; uint8_t v_warnOnNonterminal_351_; uint8_t v_traceScript_352_; uint8_t v_enableSimp_353_; uint8_t v_useSimpAll_354_; uint8_t v_useDefaultSimpSet_355_; uint8_t v_enableUnfold_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; uint8_t v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; 
v_strategy_340_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6);
v_maxRuleApplicationDepth_341_ = lean_ctor_get(v_x_339_, 0);
lean_inc(v_maxRuleApplicationDepth_341_);
v_maxRuleApplications_342_ = lean_ctor_get(v_x_339_, 1);
lean_inc(v_maxRuleApplications_342_);
v_maxGoals_343_ = lean_ctor_get(v_x_339_, 2);
lean_inc(v_maxGoals_343_);
v_maxNormIterations_344_ = lean_ctor_get(v_x_339_, 3);
lean_inc(v_maxNormIterations_344_);
v_maxSafePrefixRuleApplications_345_ = lean_ctor_get(v_x_339_, 4);
lean_inc(v_maxSafePrefixRuleApplications_345_);
v_applyHypsTransparency_346_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 1);
v_assumptionTransparency_347_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 2);
v_destructProductsTransparency_348_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 3);
v_introsTransparency_x3f_349_ = lean_ctor_get(v_x_339_, 5);
lean_inc(v_introsTransparency_x3f_349_);
v_terminal_350_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 4);
v_warnOnNonterminal_351_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 5);
v_traceScript_352_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 6);
v_enableSimp_353_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 7);
v_useSimpAll_354_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 8);
v_useDefaultSimpSet_355_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 9);
v_enableUnfold_356_ = lean_ctor_get_uint8(v_x_339_, sizeof(void*)*6 + 10);
lean_dec_ref(v_x_339_);
v___x_357_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__5));
v___x_358_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__6));
v___x_359_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__7);
v___x_360_ = lean_unsigned_to_nat(0u);
v___x_361_ = lp_aesop_Aesop_instReprStrategy_repr(v_strategy_340_, v___x_360_);
v___x_362_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_359_);
lean_ctor_set(v___x_362_, 1, v___x_361_);
v___x_363_ = 0;
v___x_364_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_364_, 0, v___x_362_);
lean_ctor_set_uint8(v___x_364_, sizeof(void*)*1, v___x_363_);
v___x_365_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_358_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__9));
v___x_367_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_365_);
lean_ctor_set(v___x_367_, 1, v___x_366_);
v___x_368_ = lean_box(1);
v___x_369_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_367_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__11));
v___x_371_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_369_);
lean_ctor_set(v___x_371_, 1, v___x_370_);
v___x_372_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_357_);
v___x_373_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__12);
v___x_374_ = l_Nat_reprFast(v_maxRuleApplicationDepth_341_);
v___x_375_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
v___x_376_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_376_, 0, v___x_373_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set_uint8(v___x_377_, sizeof(void*)*1, v___x_363_);
v___x_378_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_372_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
v___x_379_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
lean_ctor_set(v___x_379_, 1, v___x_366_);
v___x_380_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_380_, 0, v___x_379_);
lean_ctor_set(v___x_380_, 1, v___x_368_);
v___x_381_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__14));
v___x_382_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_382_, 0, v___x_380_);
lean_ctor_set(v___x_382_, 1, v___x_381_);
v___x_383_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_382_);
lean_ctor_set(v___x_383_, 1, v___x_357_);
v___x_384_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__15);
v___x_385_ = l_Nat_reprFast(v_maxRuleApplications_342_);
v___x_386_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
v___x_387_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_387_, 0, v___x_384_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
v___x_388_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_388_, 0, v___x_387_);
lean_ctor_set_uint8(v___x_388_, sizeof(void*)*1, v___x_363_);
v___x_389_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_383_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
v___x_390_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
lean_ctor_set(v___x_390_, 1, v___x_366_);
v___x_391_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
lean_ctor_set(v___x_391_, 1, v___x_368_);
v___x_392_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__17));
v___x_393_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_391_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
v___x_394_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v___x_357_);
v___x_395_ = l_Nat_reprFast(v_maxGoals_343_);
v___x_396_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
v___x_397_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_397_, 0, v___x_359_);
lean_ctor_set(v___x_397_, 1, v___x_396_);
v___x_398_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set_uint8(v___x_398_, sizeof(void*)*1, v___x_363_);
v___x_399_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_394_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
v___x_400_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v___x_366_);
v___x_401_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_400_);
lean_ctor_set(v___x_401_, 1, v___x_368_);
v___x_402_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__19));
v___x_403_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_401_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
v___x_404_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_404_, 0, v___x_403_);
lean_ctor_set(v___x_404_, 1, v___x_357_);
v___x_405_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__20);
v___x_406_ = l_Nat_reprFast(v_maxNormIterations_344_);
v___x_407_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_407_, 0, v___x_406_);
v___x_408_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_405_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_409_, 0, v___x_408_);
lean_ctor_set_uint8(v___x_409_, sizeof(void*)*1, v___x_363_);
v___x_410_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_404_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_410_);
lean_ctor_set(v___x_411_, 1, v___x_366_);
v___x_412_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_412_, 0, v___x_411_);
lean_ctor_set(v___x_412_, 1, v___x_368_);
v___x_413_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__22));
v___x_414_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_412_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
v___x_415_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
lean_ctor_set(v___x_415_, 1, v___x_357_);
v___x_416_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__23);
v___x_417_ = l_Nat_reprFast(v_maxSafePrefixRuleApplications_345_);
v___x_418_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
v___x_419_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_416_);
lean_ctor_set(v___x_419_, 1, v___x_418_);
v___x_420_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_420_, 0, v___x_419_);
lean_ctor_set_uint8(v___x_420_, sizeof(void*)*1, v___x_363_);
v___x_421_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_421_, 0, v___x_415_);
lean_ctor_set(v___x_421_, 1, v___x_420_);
v___x_422_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v___x_366_);
v___x_423_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
lean_ctor_set(v___x_423_, 1, v___x_368_);
v___x_424_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__25));
v___x_425_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_425_, 0, v___x_423_);
lean_ctor_set(v___x_425_, 1, v___x_424_);
v___x_426_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_426_, 0, v___x_425_);
lean_ctor_set(v___x_426_, 1, v___x_357_);
v___x_427_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__26);
v___x_428_ = l_Lean_Meta_instReprTransparencyMode_repr(v_applyHypsTransparency_346_, v___x_360_);
v___x_429_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_427_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
v___x_430_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_430_, 0, v___x_429_);
lean_ctor_set_uint8(v___x_430_, sizeof(void*)*1, v___x_363_);
v___x_431_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_431_, 0, v___x_426_);
lean_ctor_set(v___x_431_, 1, v___x_430_);
v___x_432_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
lean_ctor_set(v___x_432_, 1, v___x_366_);
v___x_433_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
lean_ctor_set(v___x_433_, 1, v___x_368_);
v___x_434_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__28));
v___x_435_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_435_, 0, v___x_433_);
lean_ctor_set(v___x_435_, 1, v___x_434_);
v___x_436_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_435_);
lean_ctor_set(v___x_436_, 1, v___x_357_);
v___x_437_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__29);
v___x_438_ = l_Lean_Meta_instReprTransparencyMode_repr(v_assumptionTransparency_347_, v___x_360_);
v___x_439_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_439_, 0, v___x_437_);
lean_ctor_set(v___x_439_, 1, v___x_438_);
v___x_440_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_440_, 0, v___x_439_);
lean_ctor_set_uint8(v___x_440_, sizeof(void*)*1, v___x_363_);
v___x_441_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_436_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
v___x_442_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
lean_ctor_set(v___x_442_, 1, v___x_366_);
v___x_443_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
lean_ctor_set(v___x_443_, 1, v___x_368_);
v___x_444_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__31));
v___x_445_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_445_, 0, v___x_443_);
lean_ctor_set(v___x_445_, 1, v___x_444_);
v___x_446_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
lean_ctor_set(v___x_446_, 1, v___x_357_);
v___x_447_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__32);
v___x_448_ = l_Lean_Meta_instReprTransparencyMode_repr(v_destructProductsTransparency_348_, v___x_360_);
v___x_449_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_449_, 0, v___x_447_);
lean_ctor_set(v___x_449_, 1, v___x_448_);
v___x_450_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_450_, 0, v___x_449_);
lean_ctor_set_uint8(v___x_450_, sizeof(void*)*1, v___x_363_);
v___x_451_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_451_, 0, v___x_446_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
v___x_452_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_451_);
lean_ctor_set(v___x_452_, 1, v___x_366_);
v___x_453_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_452_);
lean_ctor_set(v___x_453_, 1, v___x_368_);
v___x_454_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__34));
v___x_455_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_455_, 0, v___x_453_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
v___x_456_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_455_);
lean_ctor_set(v___x_456_, 1, v___x_357_);
v___x_457_ = lp_aesop_Option_repr___at___00Aesop_instReprOptions_repr_spec__0(v_introsTransparency_x3f_349_, v___x_360_);
lean_dec(v_introsTransparency_x3f_349_);
v___x_458_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_384_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
v___x_459_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_ctor_set_uint8(v___x_459_, sizeof(void*)*1, v___x_363_);
v___x_460_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_456_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_461_, 0, v___x_460_);
lean_ctor_set(v___x_461_, 1, v___x_366_);
v___x_462_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_461_);
lean_ctor_set(v___x_462_, 1, v___x_368_);
v___x_463_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__36));
v___x_464_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_464_, 0, v___x_462_);
lean_ctor_set(v___x_464_, 1, v___x_463_);
v___x_465_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v___x_357_);
v___x_466_ = l_Bool_repr___redArg(v_terminal_350_);
v___x_467_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_359_);
lean_ctor_set(v___x_467_, 1, v___x_466_);
v___x_468_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_468_, 0, v___x_467_);
lean_ctor_set_uint8(v___x_468_, sizeof(void*)*1, v___x_363_);
v___x_469_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_469_, 0, v___x_465_);
lean_ctor_set(v___x_469_, 1, v___x_468_);
v___x_470_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_469_);
lean_ctor_set(v___x_470_, 1, v___x_366_);
v___x_471_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
lean_ctor_set(v___x_471_, 1, v___x_368_);
v___x_472_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__38));
v___x_473_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_473_, 0, v___x_471_);
lean_ctor_set(v___x_473_, 1, v___x_472_);
v___x_474_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_474_, 0, v___x_473_);
lean_ctor_set(v___x_474_, 1, v___x_357_);
v___x_475_ = l_Bool_repr___redArg(v_warnOnNonterminal_351_);
v___x_476_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_476_, 0, v___x_405_);
lean_ctor_set(v___x_476_, 1, v___x_475_);
v___x_477_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_477_, 0, v___x_476_);
lean_ctor_set_uint8(v___x_477_, sizeof(void*)*1, v___x_363_);
v___x_478_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_474_);
lean_ctor_set(v___x_478_, 1, v___x_477_);
v___x_479_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_478_);
lean_ctor_set(v___x_479_, 1, v___x_366_);
v___x_480_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
lean_ctor_set(v___x_480_, 1, v___x_368_);
v___x_481_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__40));
v___x_482_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_482_, 0, v___x_480_);
lean_ctor_set(v___x_482_, 1, v___x_481_);
v___x_483_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_483_, 0, v___x_482_);
lean_ctor_set(v___x_483_, 1, v___x_357_);
v___x_484_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__41);
v___x_485_ = l_Bool_repr___redArg(v_traceScript_352_);
v___x_486_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_484_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set_uint8(v___x_487_, sizeof(void*)*1, v___x_363_);
v___x_488_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_483_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_488_);
lean_ctor_set(v___x_489_, 1, v___x_366_);
v___x_490_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
lean_ctor_set(v___x_490_, 1, v___x_368_);
v___x_491_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__43));
v___x_492_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_490_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
lean_ctor_set(v___x_493_, 1, v___x_357_);
v___x_494_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__44);
v___x_495_ = l_Bool_repr___redArg(v_enableSimp_353_);
v___x_496_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_496_, 0, v___x_494_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_497_, 0, v___x_496_);
lean_ctor_set_uint8(v___x_497_, sizeof(void*)*1, v___x_363_);
v___x_498_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_498_, 0, v___x_493_);
lean_ctor_set(v___x_498_, 1, v___x_497_);
v___x_499_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_498_);
lean_ctor_set(v___x_499_, 1, v___x_366_);
v___x_500_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
lean_ctor_set(v___x_500_, 1, v___x_368_);
v___x_501_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__46));
v___x_502_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
v___x_503_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
lean_ctor_set(v___x_503_, 1, v___x_357_);
v___x_504_ = l_Bool_repr___redArg(v_useSimpAll_354_);
v___x_505_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_494_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_506_, 0, v___x_505_);
lean_ctor_set_uint8(v___x_506_, sizeof(void*)*1, v___x_363_);
v___x_507_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_507_, 0, v___x_503_);
lean_ctor_set(v___x_507_, 1, v___x_506_);
v___x_508_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
lean_ctor_set(v___x_508_, 1, v___x_366_);
v___x_509_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
lean_ctor_set(v___x_509_, 1, v___x_368_);
v___x_510_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__48));
v___x_511_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_511_, 0, v___x_509_);
lean_ctor_set(v___x_511_, 1, v___x_510_);
v___x_512_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_512_, 0, v___x_511_);
lean_ctor_set(v___x_512_, 1, v___x_357_);
v___x_513_ = l_Bool_repr___redArg(v_useDefaultSimpSet_355_);
v___x_514_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_405_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
v___x_515_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_515_, 0, v___x_514_);
lean_ctor_set_uint8(v___x_515_, sizeof(void*)*1, v___x_363_);
v___x_516_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_512_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
v___x_517_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_517_, 0, v___x_516_);
lean_ctor_set(v___x_517_, 1, v___x_366_);
v___x_518_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_518_, 0, v___x_517_);
lean_ctor_set(v___x_518_, 1, v___x_368_);
v___x_519_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__50));
v___x_520_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_518_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
v___x_521_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v___x_357_);
v___x_522_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__51);
v___x_523_ = l_Bool_repr___redArg(v_enableUnfold_356_);
v___x_524_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_524_, 0, v___x_522_);
lean_ctor_set(v___x_524_, 1, v___x_523_);
v___x_525_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_525_, 0, v___x_524_);
lean_ctor_set_uint8(v___x_525_, sizeof(void*)*1, v___x_363_);
v___x_526_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_526_, 0, v___x_521_);
lean_ctor_set(v___x_526_, 1, v___x_525_);
v___x_527_ = lean_obj_once(&lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54, &lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54_once, _init_lp_aesop_Aesop_instReprOptions_repr___redArg___closed__54);
v___x_528_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__55));
v___x_529_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_529_, 0, v___x_528_);
lean_ctor_set(v___x_529_, 1, v___x_526_);
v___x_530_ = ((lean_object*)(lp_aesop_Aesop_instReprOptions_repr___redArg___closed__56));
v___x_531_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_529_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_527_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
v___x_533_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_533_, 0, v___x_532_);
lean_ctor_set_uint8(v___x_533_, sizeof(void*)*1, v___x_363_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr(lean_object* v_x_534_, lean_object* v_prec_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lp_aesop_Aesop_instReprOptions_repr___redArg(v_x_534_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instReprOptions_repr___boxed(lean_object* v_x_537_, lean_object* v_prec_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_aesop_Aesop_instReprOptions_repr(v_x_537_, v_prec_538_);
lean_dec(v_prec_538_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(lean_object* v_name_542_, lean_object* v_decl_543_, lean_object* v_ref_544_){
_start:
{
lean_object* v_defValue_546_; lean_object* v_descr_547_; lean_object* v_deprecation_x3f_548_; lean_object* v___x_549_; uint8_t v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v_defValue_546_ = lean_ctor_get(v_decl_543_, 0);
v_descr_547_ = lean_ctor_get(v_decl_543_, 1);
v_deprecation_x3f_548_ = lean_ctor_get(v_decl_543_, 2);
v___x_549_ = lean_alloc_ctor(1, 0, 1);
v___x_550_ = lean_unbox(v_defValue_546_);
lean_ctor_set_uint8(v___x_549_, 0, v___x_550_);
lean_inc(v_deprecation_x3f_548_);
lean_inc_ref(v_descr_547_);
lean_inc_n(v_name_542_, 2);
v___x_551_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_551_, 0, v_name_542_);
lean_ctor_set(v___x_551_, 1, v_ref_544_);
lean_ctor_set(v___x_551_, 2, v___x_549_);
lean_ctor_set(v___x_551_, 3, v_descr_547_);
lean_ctor_set(v___x_551_, 4, v_deprecation_x3f_548_);
v___x_552_ = lean_register_option(v_name_542_, v___x_551_);
if (lean_obj_tag(v___x_552_) == 0)
{
lean_object* v___x_554_; uint8_t v_isShared_555_; uint8_t v_isSharedCheck_560_; 
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_560_ == 0)
{
lean_object* v_unused_561_; 
v_unused_561_ = lean_ctor_get(v___x_552_, 0);
lean_dec(v_unused_561_);
v___x_554_ = v___x_552_;
v_isShared_555_ = v_isSharedCheck_560_;
goto v_resetjp_553_;
}
else
{
lean_dec(v___x_552_);
v___x_554_ = lean_box(0);
v_isShared_555_ = v_isSharedCheck_560_;
goto v_resetjp_553_;
}
v_resetjp_553_:
{
lean_object* v___x_556_; lean_object* v___x_558_; 
lean_inc(v_defValue_546_);
v___x_556_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_556_, 0, v_name_542_);
lean_ctor_set(v___x_556_, 1, v_defValue_546_);
if (v_isShared_555_ == 0)
{
lean_ctor_set(v___x_554_, 0, v___x_556_);
v___x_558_ = v___x_554_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v___x_556_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
else
{
lean_object* v_a_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_569_; 
lean_dec(v_name_542_);
v_a_562_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_569_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_569_ == 0)
{
v___x_564_ = v___x_552_;
v_isShared_565_ = v_isSharedCheck_569_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_a_562_);
lean_dec(v___x_552_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_569_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_567_; 
if (v_isShared_565_ == 0)
{
v___x_567_ = v___x_564_;
goto v_reusejp_566_;
}
else
{
lean_object* v_reuseFailAlloc_568_; 
v_reuseFailAlloc_568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_568_, 0, v_a_562_);
v___x_567_ = v_reuseFailAlloc_568_;
goto v_reusejp_566_;
}
v_reusejp_566_:
{
return v___x_567_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_570_, lean_object* v_decl_571_, lean_object* v_ref_572_, lean_object* v_a_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v_name_570_, v_decl_571_, v_ref_572_);
lean_dec_ref(v_decl_571_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_595_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_));
v___x_596_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_));
v___x_597_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__7_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_));
v___x_598_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_595_, v___x_596_, v___x_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4____boxed(lean_object* v_a_599_){
_start:
{
lean_object* v_res_600_; 
v_res_600_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_();
return v_res_600_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_618_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_));
v___x_619_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_));
v___x_620_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_));
v___x_621_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_618_, v___x_619_, v___x_620_);
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4____boxed(lean_object* v_a_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_();
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_641_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_));
v___x_642_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_));
v___x_643_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_));
v___x_644_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_641_, v___x_642_, v___x_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4____boxed(lean_object* v_a_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_();
return v_res_646_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_664_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_));
v___x_665_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_));
v___x_666_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_));
v___x_667_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_664_, v___x_665_, v___x_666_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4____boxed(lean_object* v_a_668_){
_start:
{
lean_object* v_res_669_; 
v_res_669_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_();
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; 
v___x_688_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_));
v___x_689_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_));
v___x_690_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_));
v___x_691_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_688_, v___x_689_, v___x_690_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4____boxed(lean_object* v_a_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_();
return v_res_693_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; 
v___x_711_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_));
v___x_712_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_));
v___x_713_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_));
v___x_714_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_711_, v___x_712_, v___x_713_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4____boxed(lean_object* v_a_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_();
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_732_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__1_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_));
v___x_733_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__3_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_));
v___x_734_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__4_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_));
v___x_735_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4__spec__0(v___x_732_, v___x_733_, v___x_734_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4____boxed(lean_object* v_a_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_();
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0(lean_object* v_name_738_, lean_object* v_decl_739_, lean_object* v_ref_740_){
_start:
{
lean_object* v_defValue_742_; lean_object* v_descr_743_; lean_object* v_deprecation_x3f_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; 
v_defValue_742_ = lean_ctor_get(v_decl_739_, 0);
v_descr_743_ = lean_ctor_get(v_decl_739_, 1);
v_deprecation_x3f_744_ = lean_ctor_get(v_decl_739_, 2);
lean_inc(v_defValue_742_);
v___x_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_745_, 0, v_defValue_742_);
lean_inc(v_deprecation_x3f_744_);
lean_inc_ref(v_descr_743_);
lean_inc_n(v_name_738_, 2);
v___x_746_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_746_, 0, v_name_738_);
lean_ctor_set(v___x_746_, 1, v_ref_740_);
lean_ctor_set(v___x_746_, 2, v___x_745_);
lean_ctor_set(v___x_746_, 3, v_descr_743_);
lean_ctor_set(v___x_746_, 4, v_deprecation_x3f_744_);
v___x_747_ = lean_register_option(v_name_738_, v___x_746_);
if (lean_obj_tag(v___x_747_) == 0)
{
lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_755_; 
v_isSharedCheck_755_ = !lean_is_exclusive(v___x_747_);
if (v_isSharedCheck_755_ == 0)
{
lean_object* v_unused_756_; 
v_unused_756_ = lean_ctor_get(v___x_747_, 0);
lean_dec(v_unused_756_);
v___x_749_ = v___x_747_;
v_isShared_750_ = v_isSharedCheck_755_;
goto v_resetjp_748_;
}
else
{
lean_dec(v___x_747_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_755_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_751_; lean_object* v___x_753_; 
lean_inc(v_defValue_742_);
v___x_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_751_, 0, v_name_738_);
lean_ctor_set(v___x_751_, 1, v_defValue_742_);
if (v_isShared_750_ == 0)
{
lean_ctor_set(v___x_749_, 0, v___x_751_);
v___x_753_ = v___x_749_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v___x_751_);
v___x_753_ = v_reuseFailAlloc_754_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
return v___x_753_;
}
}
}
else
{
lean_object* v_a_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_764_; 
lean_dec(v_name_738_);
v_a_757_ = lean_ctor_get(v___x_747_, 0);
v_isSharedCheck_764_ = !lean_is_exclusive(v___x_747_);
if (v_isSharedCheck_764_ == 0)
{
v___x_759_ = v___x_747_;
v_isShared_760_ = v_isSharedCheck_764_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_a_757_);
lean_dec(v___x_747_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_764_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
lean_object* v___x_762_; 
if (v_isShared_760_ == 0)
{
v___x_762_ = v___x_759_;
goto v_reusejp_761_;
}
else
{
lean_object* v_reuseFailAlloc_763_; 
v_reuseFailAlloc_763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_763_, 0, v_a_757_);
v___x_762_ = v_reuseFailAlloc_763_;
goto v_reusejp_761_;
}
v_reusejp_761_:
{
return v___x_762_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_765_, lean_object* v_decl_766_, lean_object* v_ref_767_, lean_object* v_a_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0(v_name_765_, v_decl_766_, v_ref_767_);
lean_dec_ref(v_decl_766_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; 
v___x_788_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__2_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_));
v___x_789_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__5_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_));
v___x_790_ = ((lean_object*)(lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn___closed__6_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_));
v___x_791_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4__spec__0(v___x_788_, v___x_789_, v___x_790_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4____boxed(lean_object* v_a_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_();
return v_res_793_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_Options(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Options_Public(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedStrategy_default = _init_lp_aesop_Aesop_instInhabitedStrategy_default();
lp_aesop_Aesop_instInhabitedStrategy = _init_lp_aesop_Aesop_instInhabitedStrategy();
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1840844879____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_dev_dynamicStructuring = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_dev_dynamicStructuring);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1603784145____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_dev_optimizedDynamicStructuring = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_dev_optimizedDynamicStructuring);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_4206781498____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_dev_generateScript = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_dev_generateScript);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_617117714____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_dev_statefulForward = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_dev_statefulForward);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3282936036____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_warn_applyIff = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_warn_applyIff);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1260513527____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_warn_nonterminal = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_warn_nonterminal);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_1566740328____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_collectStats = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_collectStats);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Options_Public_0__Aesop_initFn_00___x40_Aesop_Options_Public_3910975230____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_aesop_stats_file = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_aesop_stats_file);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Options_Public(uint8_t builtin) {
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
lean_object* initialize_Lean_Data_Options(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Options_Public(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Options_Public(builtin);
}
#ifdef __cplusplus
}
#endif
