// Lean compiler output
// Module: Mathlib.Util.AtomM.Recurse
// Imports: public import Init public meta import Init public import Mathlib.Util.AtomM
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
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_Simp_postDefault___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_instBEqTransparencyMode_beq(uint8_t, uint8_t);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_Meta_instReprTransparencyMode_repr(uint8_t, lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
lean_object* lean_string_length(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(2, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "red"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "zetaDelta"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "contextual"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_Simp_postDefault___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__3_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_self"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(224, 148, 98, 216, 254, 239, 13, 169)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "iff_self"};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(79, 255, 41, 65, 134, 196, 244, 123)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq(lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_red_8_; uint8_t v_zetaDelta_9_; uint8_t v_contextual_10_; uint8_t v_red_11_; uint8_t v_zetaDelta_12_; uint8_t v_contextual_13_; uint8_t v___y_15_; uint8_t v___x_16_; 
v_red_8_ = lean_ctor_get_uint8(v_x_6_, 0);
v_zetaDelta_9_ = lean_ctor_get_uint8(v_x_6_, 1);
v_contextual_10_ = lean_ctor_get_uint8(v_x_6_, 2);
v_red_11_ = lean_ctor_get_uint8(v_x_7_, 0);
v_zetaDelta_12_ = lean_ctor_get_uint8(v_x_7_, 1);
v_contextual_13_ = lean_ctor_get_uint8(v_x_7_, 2);
v___x_16_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_red_8_, v_red_11_);
if (v___x_16_ == 0)
{
return v___x_16_;
}
else
{
if (v_zetaDelta_9_ == 0)
{
if (v_zetaDelta_12_ == 0)
{
v___y_15_ = v___x_16_;
goto v___jp_14_;
}
else
{
return v_zetaDelta_9_;
}
}
else
{
v___y_15_ = v_zetaDelta_12_;
goto v___jp_14_;
}
}
v___jp_14_:
{
if (v___y_15_ == 0)
{
return v___y_15_;
}
else
{
if (v_contextual_10_ == 0)
{
if (v_contextual_13_ == 0)
{
return v___y_15_;
}
else
{
return v_contextual_10_;
}
}
else
{
return v_contextual_13_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq___boxed(lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instBEqConfig_beq(v_x_17_, v_x_18_);
lean_dec_ref(v_x_18_);
lean_dec_ref(v_x_17_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr_spec__0(lean_object* v_a_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_nat_to_int(v_a_23_);
return v___x_24_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lean_unsigned_to_nat(7u);
v___x_39_ = lean_nat_to_int(v___x_38_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_unsigned_to_nat(13u);
v___x_47_ = lean_nat_to_int(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lean_unsigned_to_nat(14u);
v___x_52_ = lean_nat_to_int(v___x_51_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__0));
v___x_55_ = lean_string_length(v___x_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__17);
v___x_57_ = lean_nat_to_int(v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(lean_object* v_x_62_){
_start:
{
uint8_t v_red_63_; uint8_t v_zetaDelta_64_; uint8_t v_contextual_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; uint8_t v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_red_63_ = lean_ctor_get_uint8(v_x_62_, 0);
v_zetaDelta_64_ = lean_ctor_get_uint8(v_x_62_, 1);
v_contextual_65_ = lean_ctor_get_uint8(v_x_62_, 2);
v___x_66_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__5));
v___x_67_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__6));
v___x_68_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__7);
v___x_69_ = lean_unsigned_to_nat(0u);
v___x_70_ = l_Lean_Meta_instReprTransparencyMode_repr(v_red_63_, v___x_69_);
v___x_71_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_68_);
lean_ctor_set(v___x_71_, 1, v___x_70_);
v___x_72_ = 0;
v___x_73_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_73_, 0, v___x_71_);
lean_ctor_set_uint8(v___x_73_, sizeof(void*)*1, v___x_72_);
v___x_74_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_67_);
lean_ctor_set(v___x_74_, 1, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__9));
v___x_76_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_74_);
lean_ctor_set(v___x_76_, 1, v___x_75_);
v___x_77_ = lean_box(1);
v___x_78_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v___x_79_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__11));
v___x_80_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_78_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
v___x_81_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_66_);
v___x_82_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__12);
v___x_83_ = l_Bool_repr___redArg(v_zetaDelta_64_);
v___x_84_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_82_);
lean_ctor_set(v___x_84_, 1, v___x_83_);
v___x_85_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set_uint8(v___x_85_, sizeof(void*)*1, v___x_72_);
v___x_86_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_81_);
lean_ctor_set(v___x_86_, 1, v___x_85_);
v___x_87_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_75_);
v___x_88_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v___x_77_);
v___x_89_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__14));
v___x_90_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_88_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_66_);
v___x_92_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15, &lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__15);
v___x_93_ = l_Bool_repr___redArg(v_contextual_65_);
v___x_94_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set_uint8(v___x_95_, sizeof(void*)*1, v___x_72_);
v___x_96_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_91_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18, &lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__18);
v___x_98_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__19));
v___x_99_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v___x_96_);
v___x_100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___closed__20));
v___x_101_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_99_);
lean_ctor_set(v___x_101_, 1, v___x_100_);
v___x_102_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_97_);
lean_ctor_set(v___x_102_, 1, v___x_101_);
v___x_103_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set_uint8(v___x_103_, sizeof(void*)*1, v___x_72_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg___boxed(lean_object* v_x_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(v_x_104_);
lean_dec_ref(v_x_104_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr(lean_object* v_x_106_, lean_object* v_prec_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___redArg(v_x_106_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr___boxed(lean_object* v_x_109_, lean_object* v_prec_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Mathlib_Tactic_AtomM_Recurse_instReprConfig_repr(v_x_109_, v_prec_110_);
lean_dec(v_prec_110_);
lean_dec_ref(v_x_109_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0(lean_object* v_msgData_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v___x_120_; lean_object* v_env_121_; lean_object* v___x_122_; lean_object* v_mctx_123_; lean_object* v_lctx_124_; lean_object* v_options_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_120_ = lean_st_ref_get(v___y_118_);
v_env_121_ = lean_ctor_get(v___x_120_, 0);
lean_inc_ref(v_env_121_);
lean_dec(v___x_120_);
v___x_122_ = lean_st_ref_get(v___y_116_);
v_mctx_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc_ref(v_mctx_123_);
lean_dec(v___x_122_);
v_lctx_124_ = lean_ctor_get(v___y_115_, 2);
v_options_125_ = lean_ctor_get(v___y_117_, 2);
lean_inc_ref(v_options_125_);
lean_inc_ref(v_lctx_124_);
v___x_126_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_126_, 0, v_env_121_);
lean_ctor_set(v___x_126_, 1, v_mctx_123_);
lean_ctor_set(v___x_126_, 2, v_lctx_124_);
lean_ctor_set(v___x_126_, 3, v_options_125_);
v___x_127_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v_msgData_114_);
v___x_128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0___boxed(lean_object* v_msgData_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0(v_msgData_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg(lean_object* v_msg_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_ref_142_; lean_object* v___x_143_; lean_object* v_a_144_; lean_object* v___x_146_; uint8_t v_isShared_147_; uint8_t v_isSharedCheck_152_; 
v_ref_142_ = lean_ctor_get(v___y_139_, 5);
v___x_143_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0_spec__0(v_msg_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_);
v_a_144_ = lean_ctor_get(v___x_143_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_143_);
if (v_isSharedCheck_152_ == 0)
{
v___x_146_ = v___x_143_;
v_isShared_147_ = v_isSharedCheck_152_;
goto v_resetjp_145_;
}
else
{
lean_inc(v_a_144_);
lean_dec(v___x_143_);
v___x_146_ = lean_box(0);
v_isShared_147_ = v_isSharedCheck_152_;
goto v_resetjp_145_;
}
v_resetjp_145_:
{
lean_object* v___x_148_; lean_object* v___x_150_; 
lean_inc(v_ref_142_);
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, v_ref_142_);
lean_ctor_set(v___x_148_, 1, v_a_144_);
if (v_isShared_147_ == 0)
{
lean_ctor_set_tag(v___x_146_, 1);
lean_ctor_set(v___x_146_, 0, v___x_148_);
v___x_150_ = v___x_146_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_148_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg___boxed(lean_object* v_msg_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg(v_msg_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
return v_res_159_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__1));
v___x_164_ = l_Lean_stringToMessageData(v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0(lean_object* v_eval_165_, lean_object* v_rctx_166_, lean_object* v_s_167_, lean_object* v_simp_168_, uint8_t v_root_169_, lean_object* v_parent_170_, lean_object* v_e_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_){
_start:
{
lean_object* v___y_181_; uint8_t v___y_182_; lean_object* v_a_187_; lean_object* v___y_191_; lean_object* v___y_192_; uint8_t v___y_193_; uint8_t v_a_194_; uint8_t v___y_202_; 
if (v_root_169_ == 0)
{
uint8_t v___x_232_; 
v___x_232_ = lean_expr_eqv(v_parent_170_, v_e_171_);
if (v___x_232_ == 0)
{
goto v___jp_230_;
}
else
{
lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v_a_235_; 
lean_dec_ref(v_e_171_);
lean_dec_ref(v_simp_168_);
lean_dec_ref(v_eval_165_);
v___x_233_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__2);
v___x_234_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg(v___x_233_, v___y_175_, v___y_176_, v___y_177_, v___y_178_);
v_a_235_ = lean_ctor_get(v___x_234_, 0);
lean_inc(v_a_235_);
lean_dec_ref(v___x_234_);
v_a_187_ = v_a_235_;
goto v___jp_186_;
}
}
else
{
goto v___jp_230_;
}
v___jp_180_:
{
if (v___y_182_ == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; 
lean_dec_ref(v___y_181_);
v___x_183_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___closed__0));
v___x_184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
return v___x_184_;
}
else
{
lean_object* v___x_185_; 
v___x_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_185_, 0, v___y_181_);
return v___x_185_;
}
}
v___jp_186_:
{
uint8_t v___x_188_; 
v___x_188_ = l_Lean_Exception_isInterrupt(v_a_187_);
if (v___x_188_ == 0)
{
uint8_t v___x_189_; 
lean_inc_ref(v_a_187_);
v___x_189_ = l_Lean_Exception_isRuntime(v_a_187_);
v___y_181_ = v_a_187_;
v___y_182_ = v___x_189_;
goto v___jp_180_;
}
else
{
v___y_181_ = v_a_187_;
v___y_182_ = v___x_188_;
goto v___jp_180_;
}
}
v___jp_190_:
{
if (v_a_194_ == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec_ref(v___y_192_);
v___x_195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_195_, 0, v___y_191_);
v___x_196_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
return v___x_196_;
}
else
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
lean_dec_ref(v___y_191_);
v___x_197_ = lean_box(0);
v___x_198_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_198_, 0, v___y_192_);
lean_ctor_set(v___x_198_, 1, v___x_197_);
lean_ctor_set_uint8(v___x_198_, sizeof(void*)*2, v___y_193_);
v___x_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
return v___x_200_;
}
}
v___jp_201_:
{
lean_object* v___x_203_; 
lean_inc(v___y_178_);
lean_inc_ref(v___y_177_);
lean_inc(v___y_176_);
lean_inc_ref(v___y_175_);
lean_inc(v_s_167_);
lean_inc_ref(v_rctx_166_);
lean_inc_ref(v_e_171_);
v___x_203_ = lean_apply_8(v_eval_165_, v_e_171_, v_rctx_166_, v_s_167_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, lean_box(0));
if (lean_obj_tag(v___x_203_) == 0)
{
lean_object* v_a_204_; lean_object* v___x_205_; 
v_a_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_a_204_);
lean_dec_ref_known(v___x_203_, 1);
lean_inc(v___y_178_);
lean_inc_ref(v___y_177_);
lean_inc(v___y_176_);
lean_inc_ref(v___y_175_);
v___x_205_ = lean_apply_6(v_simp_168_, v_a_204_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, lean_box(0));
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v_a_206_; lean_object* v_expr_207_; lean_object* v_keyedConfig_208_; uint8_t v_trackZetaDelta_209_; lean_object* v_zetaDeltaSet_210_; lean_object* v_lctx_211_; lean_object* v_localInstances_212_; lean_object* v_defEqCtx_x3f_213_; lean_object* v_synthPendingDepth_214_; lean_object* v_customCanUnfoldPredicate_x3f_215_; uint8_t v_univApprox_216_; uint8_t v_inTypeClassResolution_217_; uint8_t v_cacheInferType_218_; uint8_t v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v_a_206_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_206_);
lean_dec_ref_known(v___x_205_, 1);
v_expr_207_ = lean_ctor_get(v_a_206_, 0);
lean_inc_ref_n(v_expr_207_, 2);
v_keyedConfig_208_ = lean_ctor_get(v___y_175_, 0);
v_trackZetaDelta_209_ = lean_ctor_get_uint8(v___y_175_, sizeof(void*)*7);
v_zetaDeltaSet_210_ = lean_ctor_get(v___y_175_, 1);
v_lctx_211_ = lean_ctor_get(v___y_175_, 2);
v_localInstances_212_ = lean_ctor_get(v___y_175_, 3);
v_defEqCtx_x3f_213_ = lean_ctor_get(v___y_175_, 4);
v_synthPendingDepth_214_ = lean_ctor_get(v___y_175_, 5);
v_customCanUnfoldPredicate_x3f_215_ = lean_ctor_get(v___y_175_, 6);
v_univApprox_216_ = lean_ctor_get_uint8(v___y_175_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_217_ = lean_ctor_get_uint8(v___y_175_, sizeof(void*)*7 + 2);
v_cacheInferType_218_ = lean_ctor_get_uint8(v___y_175_, sizeof(void*)*7 + 3);
v___x_219_ = 2;
lean_inc_ref(v_keyedConfig_208_);
v___x_220_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_219_, v_keyedConfig_208_);
lean_inc(v_customCanUnfoldPredicate_x3f_215_);
lean_inc(v_synthPendingDepth_214_);
lean_inc(v_defEqCtx_x3f_213_);
lean_inc_ref(v_localInstances_212_);
lean_inc_ref(v_lctx_211_);
lean_inc(v_zetaDeltaSet_210_);
v___x_221_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v_zetaDeltaSet_210_);
lean_ctor_set(v___x_221_, 2, v_lctx_211_);
lean_ctor_set(v___x_221_, 3, v_localInstances_212_);
lean_ctor_set(v___x_221_, 4, v_defEqCtx_x3f_213_);
lean_ctor_set(v___x_221_, 5, v_synthPendingDepth_214_);
lean_ctor_set(v___x_221_, 6, v_customCanUnfoldPredicate_x3f_215_);
lean_ctor_set_uint8(v___x_221_, sizeof(void*)*7, v_trackZetaDelta_209_);
lean_ctor_set_uint8(v___x_221_, sizeof(void*)*7 + 1, v_univApprox_216_);
lean_ctor_set_uint8(v___x_221_, sizeof(void*)*7 + 2, v_inTypeClassResolution_217_);
lean_ctor_set_uint8(v___x_221_, sizeof(void*)*7 + 3, v_cacheInferType_218_);
v___x_222_ = l_Lean_Meta_isExprDefEq(v_expr_207_, v_e_171_, v___x_221_, v___y_176_, v___y_177_, v___y_178_);
lean_dec_ref_known(v___x_221_, 7);
if (lean_obj_tag(v___x_222_) == 0)
{
lean_object* v_a_223_; uint8_t v___x_224_; 
v_a_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_a_223_);
lean_dec_ref_known(v___x_222_, 1);
v___x_224_ = lean_unbox(v_a_223_);
lean_dec(v_a_223_);
v___y_191_ = v_a_206_;
v___y_192_ = v_expr_207_;
v___y_193_ = v___y_202_;
v_a_194_ = v___x_224_;
goto v___jp_190_;
}
else
{
if (lean_obj_tag(v___x_222_) == 0)
{
lean_object* v_a_225_; uint8_t v___x_226_; 
v_a_225_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_a_225_);
lean_dec_ref_known(v___x_222_, 1);
v___x_226_ = lean_unbox(v_a_225_);
lean_dec(v_a_225_);
v___y_191_ = v_a_206_;
v___y_192_ = v_expr_207_;
v___y_193_ = v___y_202_;
v_a_194_ = v___x_226_;
goto v___jp_190_;
}
else
{
lean_object* v_a_227_; 
lean_dec_ref(v_expr_207_);
lean_dec(v_a_206_);
v_a_227_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_a_227_);
lean_dec_ref_known(v___x_222_, 1);
v_a_187_ = v_a_227_;
goto v___jp_186_;
}
}
}
else
{
lean_object* v_a_228_; 
lean_dec_ref(v_e_171_);
v_a_228_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_228_);
lean_dec_ref_known(v___x_205_, 1);
v_a_187_ = v_a_228_;
goto v___jp_186_;
}
}
else
{
lean_object* v_a_229_; 
lean_dec_ref(v_e_171_);
lean_dec_ref(v_simp_168_);
v_a_229_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_a_229_);
lean_dec_ref_known(v___x_203_, 1);
v_a_187_ = v_a_229_;
goto v___jp_186_;
}
}
v___jp_230_:
{
uint8_t v___x_231_; 
v___x_231_ = 1;
v___y_202_ = v___x_231_;
goto v___jp_201_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___boxed(lean_object* v_eval_236_, lean_object* v_rctx_237_, lean_object* v_s_238_, lean_object* v_simp_239_, lean_object* v_root_240_, lean_object* v_parent_241_, lean_object* v_e_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_){
_start:
{
uint8_t v_root_boxed_251_; lean_object* v_res_252_; 
v_root_boxed_251_ = lean_unbox(v_root_240_);
v_res_252_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0(v_eval_236_, v_rctx_237_, v_s_238_, v_simp_239_, v_root_boxed_251_, v_parent_241_, v_e_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v_parent_241_);
lean_dec(v_s_238_);
lean_dec_ref(v_rctx_237_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1(lean_object* v_x_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___closed__0));
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1___boxed(lean_object* v_x_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__1(v_x_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_);
lean_dec(v___y_273_);
lean_dec_ref(v___y_272_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v_x_266_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2(lean_object* v_e_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_285_, 0, v_e_276_);
v___x_286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2___boxed(lean_object* v_e_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__2(v_e_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_);
lean_dec(v___y_294_);
lean_dec_ref(v___y_293_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3(lean_object* v_x_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_306_ = lean_box(0);
v___x_307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_307_, 0, v___x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3___boxed(lean_object* v_x_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__3(v_x_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_);
lean_dec(v___y_315_);
lean_dec_ref(v___y_314_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v___y_309_);
lean_dec_ref(v_x_308_);
return v_res_317_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5(void){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_325_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6(void){
_start:
{
lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_326_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__5);
v___x_327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_unsigned_to_nat(0u);
v___x_329_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6);
v___x_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
lean_ctor_set(v___x_330_, 1, v___x_328_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_331_ = lean_unsigned_to_nat(32u);
v___x_332_ = lean_mk_empty_array_with_capacity(v___x_331_);
v___x_333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9(void){
_start:
{
size_t v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_334_ = ((size_t)5ULL);
v___x_335_ = lean_unsigned_to_nat(0u);
v___x_336_ = lean_unsigned_to_nat(32u);
v___x_337_ = lean_mk_empty_array_with_capacity(v___x_336_);
v___x_338_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__8);
v___x_339_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v___x_337_);
lean_ctor_set(v___x_339_, 2, v___x_335_);
lean_ctor_set(v___x_339_, 3, v___x_335_);
lean_ctor_set_usize(v___x_339_, 4, v___x_334_);
return v___x_339_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_340_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__9);
v___x_341_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__6);
v___x_342_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
lean_ctor_set(v___x_342_, 2, v___x_341_);
lean_ctor_set(v___x_342_, 3, v___x_340_);
return v___x_342_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11(void){
_start:
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_343_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__10);
v___x_344_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__7);
v___x_345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_344_);
lean_ctor_set(v___x_345_, 1, v___x_343_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions(lean_object* v_eval_346_, lean_object* v_parent_347_, uint8_t v_wellBehavedDischarge_348_, uint8_t v_root_349_, lean_object* v_nctx_350_, lean_object* v_rctx_351_, lean_object* v_s_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_){
_start:
{
lean_object* v_ctx_358_; lean_object* v_simp_359_; lean_object* v___x_360_; lean_object* v_pre_361_; lean_object* v___f_362_; lean_object* v___f_363_; lean_object* v___f_364_; lean_object* v_post_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v_ctx_358_ = lean_ctor_get(v_nctx_350_, 0);
v_simp_359_ = lean_ctor_get(v_nctx_350_, 1);
v___x_360_ = lean_box(v_root_349_);
lean_inc_ref(v_parent_347_);
lean_inc_ref(v_simp_359_);
lean_inc(v_s_352_);
lean_inc_ref(v_rctx_351_);
v_pre_361_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___lam__0___boxed), 15, 6);
lean_closure_set(v_pre_361_, 0, v_eval_346_);
lean_closure_set(v_pre_361_, 1, v_rctx_351_);
lean_closure_set(v_pre_361_, 2, v_s_352_);
lean_closure_set(v_pre_361_, 3, v_simp_359_);
lean_closure_set(v_pre_361_, 4, v___x_360_);
lean_closure_set(v_pre_361_, 5, v_parent_347_);
v___f_362_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__0));
v___f_363_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__1));
v___f_364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__2));
v_post_365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__4));
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11, &lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___closed__11);
v___x_367_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_367_, 0, v_pre_361_);
lean_ctor_set(v___x_367_, 1, v_post_365_);
lean_ctor_set(v___x_367_, 2, v___f_362_);
lean_ctor_set(v___x_367_, 3, v___f_363_);
lean_ctor_set(v___x_367_, 4, v___f_364_);
lean_ctor_set_uint8(v___x_367_, sizeof(void*)*5, v_wellBehavedDischarge_348_);
lean_inc_ref(v_ctx_358_);
v___x_368_ = l_Lean_Meta_Simp_main(v_parent_347_, v_ctx_358_, v___x_366_, v___x_367_, v_a_353_, v_a_354_, v_a_355_, v_a_356_);
if (lean_obj_tag(v___x_368_) == 0)
{
lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_377_; 
v_a_369_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_377_ == 0)
{
v___x_371_ = v___x_368_;
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_368_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v_fst_373_; lean_object* v___x_375_; 
v_fst_373_ = lean_ctor_get(v_a_369_, 0);
lean_inc(v_fst_373_);
lean_dec(v_a_369_);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 0, v_fst_373_);
v___x_375_ = v___x_371_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_fst_373_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
else
{
lean_object* v_a_378_; lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_385_; 
v_a_378_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_385_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_385_ == 0)
{
v___x_380_ = v___x_368_;
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
else
{
lean_inc(v_a_378_);
lean_dec(v___x_368_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v___x_383_; 
if (v_isShared_381_ == 0)
{
v___x_383_ = v___x_380_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v_a_378_);
v___x_383_ = v_reuseFailAlloc_384_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
return v___x_383_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___boxed(lean_object* v_eval_386_, lean_object* v_parent_387_, lean_object* v_wellBehavedDischarge_388_, lean_object* v_root_389_, lean_object* v_nctx_390_, lean_object* v_rctx_391_, lean_object* v_s_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_398_; uint8_t v_root_boxed_399_; lean_object* v_res_400_; 
v_wellBehavedDischarge_boxed_398_ = lean_unbox(v_wellBehavedDischarge_388_);
v_root_boxed_399_ = lean_unbox(v_root_389_);
v_res_400_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions(v_eval_386_, v_parent_387_, v_wellBehavedDischarge_boxed_398_, v_root_boxed_399_, v_nctx_390_, v_rctx_391_, v_s_392_, v_a_393_, v_a_394_, v_a_395_, v_a_396_);
lean_dec(v_a_396_);
lean_dec_ref(v_a_395_);
lean_dec(v_a_394_);
lean_dec_ref(v_a_393_);
lean_dec(v_s_392_);
lean_dec_ref(v_rctx_391_);
lean_dec_ref(v_nctx_390_);
return v_res_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0(lean_object* v_00_u03b1_401_, lean_object* v_msg_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___redArg(v_msg_402_, v___y_403_, v___y_404_, v___y_405_, v___y_406_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0___boxed(lean_object* v_00_u03b1_409_, lean_object* v_msg_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_AtomM_onSubexpressions_spec__0(v_00_u03b1_409_, v_msg_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom___boxed(lean_object* v_s_417_, lean_object* v_cfg_418_, lean_object* v_wellBehavedDischarge_419_, lean_object* v_eval_420_, lean_object* v_nctx_421_, lean_object* v_e_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_, lean_object* v_a_426_, lean_object* v_a_427_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_428_; lean_object* v_res_429_; 
v_wellBehavedDischarge_boxed_428_ = lean_unbox(v_wellBehavedDischarge_419_);
v_res_429_ = lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom(v_s_417_, v_cfg_418_, v_wellBehavedDischarge_boxed_428_, v_eval_420_, v_nctx_421_, v_e_422_, v_a_423_, v_a_424_, v_a_425_, v_a_426_);
lean_dec(v_a_426_);
lean_dec_ref(v_a_425_);
lean_dec(v_a_424_);
lean_dec_ref(v_a_423_);
lean_dec_ref(v_nctx_421_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx(lean_object* v_s_430_, lean_object* v_cfg_431_, uint8_t v_wellBehavedDischarge_432_, lean_object* v_eval_433_, lean_object* v_nctx_434_){
_start:
{
uint8_t v_red_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v_red_435_ = lean_ctor_get_uint8(v_cfg_431_, 0);
v___x_436_ = lean_box(v_wellBehavedDischarge_432_);
lean_inc_ref(v_nctx_434_);
v___x_437_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom___boxed), 11, 5);
lean_closure_set(v___x_437_, 0, v_s_430_);
lean_closure_set(v___x_437_, 1, v_cfg_431_);
lean_closure_set(v___x_437_, 2, v___x_436_);
lean_closure_set(v___x_437_, 3, v_eval_433_);
lean_closure_set(v___x_437_, 4, v_nctx_434_);
v___x_438_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_438_, 0, v___x_437_);
lean_ctor_set_uint8(v___x_438_, sizeof(void*)*1, v_red_435_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_evalAtom(lean_object* v_s_439_, lean_object* v_cfg_440_, uint8_t v_wellBehavedDischarge_441_, lean_object* v_eval_442_, lean_object* v_nctx_443_, lean_object* v_e_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_){
_start:
{
uint8_t v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_450_ = 0;
lean_inc_ref(v_eval_442_);
lean_inc(v_s_439_);
v___x_451_ = lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx(v_s_439_, v_cfg_440_, v_wellBehavedDischarge_441_, v_eval_442_, v_nctx_443_);
v___x_452_ = lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions(v_eval_442_, v_e_444_, v_wellBehavedDischarge_441_, v___x_450_, v_nctx_443_, v___x_451_, v_s_439_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
lean_dec(v_s_439_);
lean_dec_ref(v___x_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx___boxed(lean_object* v_s_453_, lean_object* v_cfg_454_, lean_object* v_wellBehavedDischarge_455_, lean_object* v_eval_456_, lean_object* v_nctx_457_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_458_; lean_object* v_res_459_; 
v_wellBehavedDischarge_boxed_458_ = lean_unbox(v_wellBehavedDischarge_455_);
v_res_459_ = lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx(v_s_453_, v_cfg_454_, v_wellBehavedDischarge_boxed_458_, v_eval_456_, v_nctx_457_);
lean_dec_ref(v_nctx_457_);
return v_res_459_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0(void){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_460_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1(void){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; 
v___x_461_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__0);
v___x_462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_462_, 0, v___x_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0(lean_object* v_00_u03b2_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0___closed__1);
return v___x_464_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0(void){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_465_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_466_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__0);
v___x_467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1(lean_object* v_00_u03b2_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1___closed__1);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2(lean_object* v_x_470_, lean_object* v_x_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
if (lean_obj_tag(v_x_471_) == 0)
{
lean_object* v___x_477_; 
v___x_477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_477_, 0, v_x_470_);
return v___x_477_;
}
else
{
lean_object* v_head_478_; lean_object* v_tail_479_; uint8_t v___x_480_; uint8_t v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; 
v_head_478_ = lean_ctor_get(v_x_471_, 0);
lean_inc(v_head_478_);
v_tail_479_ = lean_ctor_get(v_x_471_, 1);
lean_inc(v_tail_479_);
lean_dec_ref_known(v_x_471_, 2);
v___x_480_ = 1;
v___x_481_ = 0;
v___x_482_ = lean_unsigned_to_nat(1000u);
v___x_483_ = l_Lean_Meta_SimpTheorems_addConst(v_x_470_, v_head_478_, v___x_480_, v___x_481_, v___x_482_, v___y_472_, v___y_473_, v___y_474_, v___y_475_);
if (lean_obj_tag(v___x_483_) == 0)
{
lean_object* v_a_484_; 
v_a_484_ = lean_ctor_get(v___x_483_, 0);
lean_inc(v_a_484_);
lean_dec_ref_known(v___x_483_, 1);
v_x_470_ = v_a_484_;
v_x_471_ = v_tail_479_;
goto _start;
}
else
{
lean_dec(v_tail_479_);
return v___x_483_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2___boxed(lean_object* v_x_486_, lean_object* v_x_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2(v_x_486_, v_x_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
lean_dec(v___y_489_);
lean_dec_ref(v___y_488_);
return v_res_493_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0(void){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_494_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1(void){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__0(lean_box(0));
return v___x_495_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2(void){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__1(lean_box(0));
return v___x_496_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3(void){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__3);
v___x_499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_499_, 0, v___x_498_);
return v___x_499_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5(void){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_500_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__4);
v___x_501_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__2);
v___x_502_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__1);
v___x_503_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__0);
v___x_504_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_504_, 0, v___x_503_);
lean_ctor_set(v___x_504_, 1, v___x_503_);
lean_ctor_set(v___x_504_, 2, v___x_502_);
lean_ctor_set(v___x_504_, 3, v___x_501_);
lean_ctor_set(v___x_504_, 4, v___x_502_);
lean_ctor_set(v___x_504_, 5, v___x_500_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(lean_object* v_s_517_, lean_object* v_cfg_518_, uint8_t v_wellBehavedDischarge_519_, lean_object* v_eval_520_, lean_object* v_simp_521_, lean_object* v_x_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_, lean_object* v_a_526_){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_528_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__5);
v___x_529_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___closed__11));
v___x_530_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_AtomM_RecurseM_run_spec__2(v___x_528_, v___x_529_, v_a_523_, v_a_524_, v_a_525_, v_a_526_);
if (lean_obj_tag(v___x_530_) == 0)
{
lean_object* v_a_531_; lean_object* v___x_532_; 
v_a_531_ = lean_ctor_get(v___x_530_, 0);
lean_inc(v_a_531_);
lean_dec_ref_known(v___x_530_, 1);
v___x_532_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_526_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; uint8_t v_zetaDelta_534_; uint8_t v_contextual_535_; lean_object* v___x_536_; lean_object* v___x_537_; uint8_t v___x_538_; uint8_t v___x_539_; uint8_t v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
lean_inc(v_a_533_);
lean_dec_ref_known(v___x_532_, 1);
v_zetaDelta_534_ = lean_ctor_get_uint8(v_cfg_518_, 1);
v_contextual_535_ = lean_ctor_get_uint8(v_cfg_518_, 2);
v___x_536_ = lean_unsigned_to_nat(100000u);
v___x_537_ = lean_unsigned_to_nat(2u);
v___x_538_ = 1;
v___x_539_ = 0;
v___x_540_ = 0;
v___x_541_ = lean_box(0);
v___x_542_ = lean_alloc_ctor(0, 3, 29);
lean_ctor_set(v___x_542_, 0, v___x_536_);
lean_ctor_set(v___x_542_, 1, v___x_537_);
lean_ctor_set(v___x_542_, 2, v___x_541_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3, v_contextual_535_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 1, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 2, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 3, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 4, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 5, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 6, v___x_539_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 7, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 8, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 9, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 10, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 11, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 12, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 13, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 14, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 15, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 16, v_zetaDelta_534_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 17, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 18, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 19, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 20, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 21, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 22, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 23, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 24, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 25, v___x_538_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 26, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 27, v___x_540_);
lean_ctor_set_uint8(v___x_542_, sizeof(void*)*3 + 28, v___x_540_);
v___x_543_ = lean_unsigned_to_nat(1u);
v___x_544_ = lean_mk_empty_array_with_capacity(v___x_543_);
v___x_545_ = lean_array_push(v___x_544_, v_a_531_);
v___x_546_ = l_Lean_Options_empty;
v___x_547_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_542_, v___x_545_, v_a_533_, v___x_546_, v_a_523_, v_a_525_, v_a_526_);
if (lean_obj_tag(v___x_547_) == 0)
{
lean_object* v_a_548_; lean_object* v___x_549_; uint8_t v_foApprox_550_; uint8_t v_ctxApprox_551_; uint8_t v_quasiPatternApprox_552_; uint8_t v_constApprox_553_; uint8_t v_isDefEqStuckEx_554_; uint8_t v_unificationHints_555_; uint8_t v_proofIrrelevance_556_; uint8_t v_assignSyntheticOpaque_557_; uint8_t v_offsetCnstrs_558_; uint8_t v_transparency_559_; uint8_t v_etaStruct_560_; uint8_t v_univApprox_561_; uint8_t v_iota_562_; uint8_t v_beta_563_; uint8_t v_proj_564_; uint8_t v_zeta_565_; uint8_t v_zetaUnused_566_; uint8_t v_zetaHave_567_; uint8_t v_canUnfoldPredicateConfig_568_; lean_object* v___x_570_; uint8_t v_isShared_571_; uint8_t v_isSharedCheck_591_; 
v_a_548_ = lean_ctor_get(v___x_547_, 0);
lean_inc(v_a_548_);
lean_dec_ref_known(v___x_547_, 1);
v___x_549_ = l_Lean_Meta_Context_config(v_a_523_);
v_foApprox_550_ = lean_ctor_get_uint8(v___x_549_, 0);
v_ctxApprox_551_ = lean_ctor_get_uint8(v___x_549_, 1);
v_quasiPatternApprox_552_ = lean_ctor_get_uint8(v___x_549_, 2);
v_constApprox_553_ = lean_ctor_get_uint8(v___x_549_, 3);
v_isDefEqStuckEx_554_ = lean_ctor_get_uint8(v___x_549_, 4);
v_unificationHints_555_ = lean_ctor_get_uint8(v___x_549_, 5);
v_proofIrrelevance_556_ = lean_ctor_get_uint8(v___x_549_, 6);
v_assignSyntheticOpaque_557_ = lean_ctor_get_uint8(v___x_549_, 7);
v_offsetCnstrs_558_ = lean_ctor_get_uint8(v___x_549_, 8);
v_transparency_559_ = lean_ctor_get_uint8(v___x_549_, 9);
v_etaStruct_560_ = lean_ctor_get_uint8(v___x_549_, 10);
v_univApprox_561_ = lean_ctor_get_uint8(v___x_549_, 11);
v_iota_562_ = lean_ctor_get_uint8(v___x_549_, 12);
v_beta_563_ = lean_ctor_get_uint8(v___x_549_, 13);
v_proj_564_ = lean_ctor_get_uint8(v___x_549_, 14);
v_zeta_565_ = lean_ctor_get_uint8(v___x_549_, 15);
v_zetaUnused_566_ = lean_ctor_get_uint8(v___x_549_, 17);
v_zetaHave_567_ = lean_ctor_get_uint8(v___x_549_, 18);
v_canUnfoldPredicateConfig_568_ = lean_ctor_get_uint8(v___x_549_, 19);
v_isSharedCheck_591_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_591_ == 0)
{
v___x_570_ = v___x_549_;
v_isShared_571_ = v_isSharedCheck_591_;
goto v_resetjp_569_;
}
else
{
lean_dec(v___x_549_);
v___x_570_ = lean_box(0);
v_isShared_571_ = v_isSharedCheck_591_;
goto v_resetjp_569_;
}
v_resetjp_569_:
{
uint8_t v_trackZetaDelta_572_; lean_object* v_zetaDeltaSet_573_; lean_object* v_lctx_574_; lean_object* v_localInstances_575_; lean_object* v_defEqCtx_x3f_576_; lean_object* v_synthPendingDepth_577_; lean_object* v_customCanUnfoldPredicate_x3f_578_; uint8_t v_univApprox_579_; uint8_t v_inTypeClassResolution_580_; uint8_t v_cacheInferType_581_; lean_object* v___x_583_; 
v_trackZetaDelta_572_ = lean_ctor_get_uint8(v_a_523_, sizeof(void*)*7);
v_zetaDeltaSet_573_ = lean_ctor_get(v_a_523_, 1);
v_lctx_574_ = lean_ctor_get(v_a_523_, 2);
v_localInstances_575_ = lean_ctor_get(v_a_523_, 3);
v_defEqCtx_x3f_576_ = lean_ctor_get(v_a_523_, 4);
v_synthPendingDepth_577_ = lean_ctor_get(v_a_523_, 5);
v_customCanUnfoldPredicate_x3f_578_ = lean_ctor_get(v_a_523_, 6);
v_univApprox_579_ = lean_ctor_get_uint8(v_a_523_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_580_ = lean_ctor_get_uint8(v_a_523_, sizeof(void*)*7 + 2);
v_cacheInferType_581_ = lean_ctor_get_uint8(v_a_523_, sizeof(void*)*7 + 3);
if (v_isShared_571_ == 0)
{
v___x_583_ = v___x_570_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 0, v_foApprox_550_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 1, v_ctxApprox_551_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 2, v_quasiPatternApprox_552_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 3, v_constApprox_553_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 4, v_isDefEqStuckEx_554_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 5, v_unificationHints_555_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 6, v_proofIrrelevance_556_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 7, v_assignSyntheticOpaque_557_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 8, v_offsetCnstrs_558_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 9, v_transparency_559_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 10, v_etaStruct_560_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 11, v_univApprox_561_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 12, v_iota_562_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 13, v_beta_563_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 14, v_proj_564_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 15, v_zeta_565_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 17, v_zetaUnused_566_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 18, v_zetaHave_567_);
lean_ctor_set_uint8(v_reuseFailAlloc_590_, 19, v_canUnfoldPredicateConfig_568_);
v___x_583_ = v_reuseFailAlloc_590_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
uint64_t v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
lean_ctor_set_uint8(v___x_583_, 16, v_zetaDelta_534_);
v___x_584_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_583_);
v___x_585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_585_, 0, v_a_548_);
lean_ctor_set(v___x_585_, 1, v_simp_521_);
lean_inc(v_s_517_);
v___x_586_ = lp_mathlib___private_Mathlib_Util_AtomM_Recurse_0__Mathlib_Tactic_AtomM_RecurseM_run_rctx(v_s_517_, v_cfg_518_, v_wellBehavedDischarge_519_, v_eval_520_, v___x_585_);
v___x_587_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_587_, 0, v___x_583_);
lean_ctor_set_uint64(v___x_587_, sizeof(void*)*1, v___x_584_);
lean_inc(v_customCanUnfoldPredicate_x3f_578_);
lean_inc(v_synthPendingDepth_577_);
lean_inc(v_defEqCtx_x3f_576_);
lean_inc_ref(v_localInstances_575_);
lean_inc_ref(v_lctx_574_);
lean_inc(v_zetaDeltaSet_573_);
v___x_588_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_588_, 0, v___x_587_);
lean_ctor_set(v___x_588_, 1, v_zetaDeltaSet_573_);
lean_ctor_set(v___x_588_, 2, v_lctx_574_);
lean_ctor_set(v___x_588_, 3, v_localInstances_575_);
lean_ctor_set(v___x_588_, 4, v_defEqCtx_x3f_576_);
lean_ctor_set(v___x_588_, 5, v_synthPendingDepth_577_);
lean_ctor_set(v___x_588_, 6, v_customCanUnfoldPredicate_x3f_578_);
lean_ctor_set_uint8(v___x_588_, sizeof(void*)*7, v_trackZetaDelta_572_);
lean_ctor_set_uint8(v___x_588_, sizeof(void*)*7 + 1, v_univApprox_579_);
lean_ctor_set_uint8(v___x_588_, sizeof(void*)*7 + 2, v_inTypeClassResolution_580_);
lean_ctor_set_uint8(v___x_588_, sizeof(void*)*7 + 3, v_cacheInferType_581_);
lean_inc(v_a_526_);
lean_inc_ref(v_a_525_);
lean_inc(v_a_524_);
v___x_589_ = lean_apply_8(v_x_522_, v___x_585_, v___x_586_, v_s_517_, v___x_588_, v_a_524_, v_a_525_, v_a_526_, lean_box(0));
return v___x_589_;
}
}
}
else
{
lean_object* v_a_592_; lean_object* v___x_594_; uint8_t v_isShared_595_; uint8_t v_isSharedCheck_599_; 
lean_dec_ref(v_x_522_);
lean_dec_ref(v_simp_521_);
lean_dec_ref(v_eval_520_);
lean_dec_ref(v_cfg_518_);
lean_dec(v_s_517_);
v_a_592_ = lean_ctor_get(v___x_547_, 0);
v_isSharedCheck_599_ = !lean_is_exclusive(v___x_547_);
if (v_isSharedCheck_599_ == 0)
{
v___x_594_ = v___x_547_;
v_isShared_595_ = v_isSharedCheck_599_;
goto v_resetjp_593_;
}
else
{
lean_inc(v_a_592_);
lean_dec(v___x_547_);
v___x_594_ = lean_box(0);
v_isShared_595_ = v_isSharedCheck_599_;
goto v_resetjp_593_;
}
v_resetjp_593_:
{
lean_object* v___x_597_; 
if (v_isShared_595_ == 0)
{
v___x_597_ = v___x_594_;
goto v_reusejp_596_;
}
else
{
lean_object* v_reuseFailAlloc_598_; 
v_reuseFailAlloc_598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_598_, 0, v_a_592_);
v___x_597_ = v_reuseFailAlloc_598_;
goto v_reusejp_596_;
}
v_reusejp_596_:
{
return v___x_597_;
}
}
}
}
else
{
lean_object* v_a_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_607_; 
lean_dec(v_a_531_);
lean_dec_ref(v_x_522_);
lean_dec_ref(v_simp_521_);
lean_dec_ref(v_eval_520_);
lean_dec_ref(v_cfg_518_);
lean_dec(v_s_517_);
v_a_600_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_607_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_607_ == 0)
{
v___x_602_ = v___x_532_;
v_isShared_603_ = v_isSharedCheck_607_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_a_600_);
lean_dec(v___x_532_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_607_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v___x_605_; 
if (v_isShared_603_ == 0)
{
v___x_605_ = v___x_602_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v_a_600_);
v___x_605_ = v_reuseFailAlloc_606_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
return v___x_605_;
}
}
}
}
else
{
lean_object* v_a_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_615_; 
lean_dec_ref(v_x_522_);
lean_dec_ref(v_simp_521_);
lean_dec_ref(v_eval_520_);
lean_dec_ref(v_cfg_518_);
lean_dec(v_s_517_);
v_a_608_ = lean_ctor_get(v___x_530_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v___x_530_);
if (v_isSharedCheck_615_ == 0)
{
v___x_610_ = v___x_530_;
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_a_608_);
lean_dec(v___x_530_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_613_; 
if (v_isShared_611_ == 0)
{
v___x_613_ = v___x_610_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v_a_608_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg___boxed(lean_object* v_s_616_, lean_object* v_cfg_617_, lean_object* v_wellBehavedDischarge_618_, lean_object* v_eval_619_, lean_object* v_simp_620_, lean_object* v_x_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_627_; lean_object* v_res_628_; 
v_wellBehavedDischarge_boxed_627_ = lean_unbox(v_wellBehavedDischarge_618_);
v_res_628_ = lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(v_s_616_, v_cfg_617_, v_wellBehavedDischarge_boxed_627_, v_eval_619_, v_simp_620_, v_x_621_, v_a_622_, v_a_623_, v_a_624_, v_a_625_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
lean_dec(v_a_623_);
lean_dec_ref(v_a_622_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run(lean_object* v_00_u03b1_629_, lean_object* v_s_630_, lean_object* v_cfg_631_, uint8_t v_wellBehavedDischarge_632_, lean_object* v_eval_633_, lean_object* v_simp_634_, lean_object* v_x_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(v_s_630_, v_cfg_631_, v_wellBehavedDischarge_632_, v_eval_633_, v_simp_634_, v_x_635_, v_a_636_, v_a_637_, v_a_638_, v_a_639_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___boxed(lean_object* v_00_u03b1_642_, lean_object* v_s_643_, lean_object* v_cfg_644_, lean_object* v_wellBehavedDischarge_645_, lean_object* v_eval_646_, lean_object* v_simp_647_, lean_object* v_x_648_, lean_object* v_a_649_, lean_object* v_a_650_, lean_object* v_a_651_, lean_object* v_a_652_, lean_object* v_a_653_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_654_; lean_object* v_res_655_; 
v_wellBehavedDischarge_boxed_654_ = lean_unbox(v_wellBehavedDischarge_645_);
v_res_655_ = lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run(v_00_u03b1_642_, v_s_643_, v_cfg_644_, v_wellBehavedDischarge_boxed_654_, v_eval_646_, v_simp_647_, v_x_648_, v_a_649_, v_a_650_, v_a_651_, v_a_652_);
lean_dec(v_a_652_);
lean_dec_ref(v_a_651_);
lean_dec(v_a_650_);
lean_dec_ref(v_a_649_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse(lean_object* v_s_656_, lean_object* v_cfg_657_, uint8_t v_wellBehavedDischarge_658_, lean_object* v_eval_659_, lean_object* v_simp_660_, lean_object* v_tgt_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_){
_start:
{
uint8_t v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_667_ = 1;
v___x_668_ = lean_box(v_wellBehavedDischarge_658_);
v___x_669_ = lean_box(v___x_667_);
lean_inc_ref(v_eval_659_);
v___x_670_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_AtomM_onSubexpressions___boxed), 12, 4);
lean_closure_set(v___x_670_, 0, v_eval_659_);
lean_closure_set(v___x_670_, 1, v_tgt_661_);
lean_closure_set(v___x_670_, 2, v___x_668_);
lean_closure_set(v___x_670_, 3, v___x_669_);
v___x_671_ = lp_mathlib_Mathlib_Tactic_AtomM_RecurseM_run___redArg(v_s_656_, v_cfg_657_, v_wellBehavedDischarge_658_, v_eval_659_, v_simp_660_, v___x_670_, v_a_662_, v_a_663_, v_a_664_, v_a_665_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse___boxed(lean_object* v_s_672_, lean_object* v_cfg_673_, lean_object* v_wellBehavedDischarge_674_, lean_object* v_eval_675_, lean_object* v_simp_676_, lean_object* v_tgt_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_, lean_object* v_a_681_, lean_object* v_a_682_){
_start:
{
uint8_t v_wellBehavedDischarge_boxed_683_; lean_object* v_res_684_; 
v_wellBehavedDischarge_boxed_683_ = lean_unbox(v_wellBehavedDischarge_674_);
v_res_684_ = lp_mathlib_Mathlib_Tactic_AtomM_recurse(v_s_672_, v_cfg_673_, v_wellBehavedDischarge_boxed_683_, v_eval_675_, v_simp_676_, v_tgt_677_, v_a_678_, v_a_679_, v_a_680_, v_a_681_);
lean_dec(v_a_681_);
lean_dec_ref(v_a_680_);
lean_dec(v_a_679_);
lean_dec_ref(v_a_678_);
return v_res_684_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_AtomM_Recurse(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_AtomM_Recurse(builtin);
}
#ifdef __cplusplus
}
#endif
