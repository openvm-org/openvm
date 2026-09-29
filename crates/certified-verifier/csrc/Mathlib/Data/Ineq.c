// Lean compiler output
// Module: Mathlib.Data.Ineq
// Imports: public import Init public meta import Init public import Mathlib.Lean.Expr.Basic
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_instDecidableEqIneq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instDecidableEqIneq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_instInhabitedIneq_default;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_instInhabitedIneq;
static const lean_string_object lp_mathlib_Mathlib_instReprIneq_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Mathlib.Ineq.eq"};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instReprIneq_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_instReprIneq_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Mathlib.Ineq.le"};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_instReprIneq_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_instReprIneq_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Mathlib.Ineq.lt"};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_instReprIneq_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq_repr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_instReprIneq_repr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_instReprIneq_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instReprIneq_repr___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instReprIneq_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instReprIneq_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instReprIneq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instReprIneq_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instReprIneq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_instReprIneq = (const lean_object*)&lp_mathlib_Mathlib_instReprIneq___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_max(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_max___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_cmp(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_cmp___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Ineq_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_mathlib_Mathlib_Ineq_toString___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toString___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≤"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toString___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toString___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toString___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toString___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toString(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toString___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Ineq_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Ineq_toString___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Ineq_instToString___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Ineq_instToString = (const lean_object*)&lp_mathlib_Mathlib_Ineq_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_instToFormat___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_instToFormat___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Ineq_instToFormat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Ineq_instToFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Ineq_instToFormat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_instToFormat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Ineq_instToFormat = (const lean_object*)&lp_mathlib_Mathlib_Ineq_instToFormat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ineq_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ineq_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_ineq_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ineq_x3f___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__5_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_ineq_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__6_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__7 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Expr_ineq_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Not a comparison: "};
static const lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__8 = (const lean_object*)&lp_mathlib_Lean_Expr_ineq_x3f___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_ineq_x3f___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_ineq_x3f___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineq_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorIdx(uint8_t v_x_1_){
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
uint8_t v_x_boxed_6_; lean_object* v_res_7_; 
v_x_boxed_6_ = lean_unbox(v_x_5_);
v_res_7_ = lp_mathlib_Mathlib_Ineq_ctorIdx(v_x_boxed_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___redArg(lean_object* v_k_8_){
_start:
{
lean_inc(v_k_8_);
return v_k_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___redArg___boxed(lean_object* v_k_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Mathlib_Ineq_ctorElim___redArg(v_k_9_);
lean_dec(v_k_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, uint8_t v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_inc(v_k_15_);
return v_k_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
uint8_t v_t_boxed_21_; lean_object* v_res_22_; 
v_t_boxed_21_ = lean_unbox(v_t_18_);
v_res_22_ = lp_mathlib_Mathlib_Ineq_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_boxed_21_, v_h_19_, v_k_20_);
lean_dec(v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___redArg(lean_object* v_eq_23_){
_start:
{
lean_inc(v_eq_23_);
return v_eq_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___redArg___boxed(lean_object* v_eq_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Mathlib_Ineq_eq_elim___redArg(v_eq_24_);
lean_dec(v_eq_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim(lean_object* v_motive_26_, uint8_t v_t_27_, lean_object* v_h_28_, lean_object* v_eq_29_){
_start:
{
lean_inc(v_eq_29_);
return v_eq_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_eq_elim___boxed(lean_object* v_motive_30_, lean_object* v_t_31_, lean_object* v_h_32_, lean_object* v_eq_33_){
_start:
{
uint8_t v_t_boxed_34_; lean_object* v_res_35_; 
v_t_boxed_34_ = lean_unbox(v_t_31_);
v_res_35_ = lp_mathlib_Mathlib_Ineq_eq_elim(v_motive_30_, v_t_boxed_34_, v_h_32_, v_eq_33_);
lean_dec(v_eq_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___redArg(lean_object* v_le_36_){
_start:
{
lean_inc(v_le_36_);
return v_le_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___redArg___boxed(lean_object* v_le_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Mathlib_Ineq_le_elim___redArg(v_le_37_);
lean_dec(v_le_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim(lean_object* v_motive_39_, uint8_t v_t_40_, lean_object* v_h_41_, lean_object* v_le_42_){
_start:
{
lean_inc(v_le_42_);
return v_le_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_le_elim___boxed(lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_le_46_){
_start:
{
uint8_t v_t_boxed_47_; lean_object* v_res_48_; 
v_t_boxed_47_ = lean_unbox(v_t_44_);
v_res_48_ = lp_mathlib_Mathlib_Ineq_le_elim(v_motive_43_, v_t_boxed_47_, v_h_45_, v_le_46_);
lean_dec(v_le_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___redArg(lean_object* v_lt_49_){
_start:
{
lean_inc(v_lt_49_);
return v_lt_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___redArg___boxed(lean_object* v_lt_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Mathlib_Ineq_lt_elim___redArg(v_lt_50_);
lean_dec(v_lt_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim(lean_object* v_motive_52_, uint8_t v_t_53_, lean_object* v_h_54_, lean_object* v_lt_55_){
_start:
{
lean_inc(v_lt_55_);
return v_lt_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_lt_elim___boxed(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_lt_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_mathlib_Mathlib_Ineq_lt_elim(v_motive_56_, v_t_boxed_60_, v_h_58_, v_lt_59_);
lean_dec(v_lt_59_);
return v_res_61_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_ofNat(lean_object* v_n_62_){
_start:
{
lean_object* v___x_63_; uint8_t v___x_64_; 
v___x_63_ = lean_unsigned_to_nat(0u);
v___x_64_ = lean_nat_dec_le(v_n_62_, v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_unsigned_to_nat(1u);
v___x_66_ = lean_nat_dec_le(v_n_62_, v___x_65_);
if (v___x_66_ == 0)
{
uint8_t v___x_67_; 
v___x_67_ = 2;
return v___x_67_;
}
else
{
uint8_t v___x_68_; 
v___x_68_ = 1;
return v___x_68_;
}
}
else
{
uint8_t v___x_69_; 
v___x_69_ = 0;
return v___x_69_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_ofNat___boxed(lean_object* v_n_70_){
_start:
{
uint8_t v_res_71_; lean_object* v_r_72_; 
v_res_71_ = lp_mathlib_Mathlib_Ineq_ofNat(v_n_70_);
lean_dec(v_n_70_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_instDecidableEqIneq(uint8_t v_x_73_, uint8_t v_y_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_75_ = lp_mathlib_Mathlib_Ineq_ctorIdx(v_x_73_);
v___x_76_ = lp_mathlib_Mathlib_Ineq_ctorIdx(v_y_74_);
v___x_77_ = lean_nat_dec_eq(v___x_75_, v___x_76_);
lean_dec(v___x_76_);
lean_dec(v___x_75_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instDecidableEqIneq___boxed(lean_object* v_x_78_, lean_object* v_y_79_){
_start:
{
uint8_t v_x_13__boxed_80_; uint8_t v_y_14__boxed_81_; uint8_t v_res_82_; lean_object* v_r_83_; 
v_x_13__boxed_80_ = lean_unbox(v_x_78_);
v_y_14__boxed_81_ = lean_unbox(v_y_79_);
v_res_82_ = lp_mathlib_Mathlib_instDecidableEqIneq(v_x_13__boxed_80_, v_y_14__boxed_81_);
v_r_83_ = lean_box(v_res_82_);
return v_r_83_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_instInhabitedIneq_default(void){
_start:
{
uint8_t v___x_84_; 
v___x_84_ = 0;
return v___x_84_;
}
}
static uint8_t _init_lp_mathlib_Mathlib_instInhabitedIneq(void){
_start:
{
uint8_t v___x_85_; 
v___x_85_ = 0;
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__6(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = lean_unsigned_to_nat(2u);
v___x_96_ = lean_nat_to_int(v___x_95_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__7(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = lean_unsigned_to_nat(1u);
v___x_98_ = lean_nat_to_int(v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instReprIneq_repr(uint8_t v_x_99_, lean_object* v_prec_100_){
_start:
{
lean_object* v___y_102_; lean_object* v___y_109_; lean_object* v___y_116_; 
switch(v_x_99_)
{
case 0:
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = lean_unsigned_to_nat(1024u);
v___x_123_ = lean_nat_dec_le(v___x_122_, v_prec_100_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__6, &lp_mathlib_Mathlib_instReprIneq_repr___closed__6_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__6);
v___y_102_ = v___x_124_;
goto v___jp_101_;
}
else
{
lean_object* v___x_125_; 
v___x_125_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__7, &lp_mathlib_Mathlib_instReprIneq_repr___closed__7_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__7);
v___y_102_ = v___x_125_;
goto v___jp_101_;
}
}
case 1:
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = lean_unsigned_to_nat(1024u);
v___x_127_ = lean_nat_dec_le(v___x_126_, v_prec_100_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__6, &lp_mathlib_Mathlib_instReprIneq_repr___closed__6_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__6);
v___y_109_ = v___x_128_;
goto v___jp_108_;
}
else
{
lean_object* v___x_129_; 
v___x_129_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__7, &lp_mathlib_Mathlib_instReprIneq_repr___closed__7_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__7);
v___y_109_ = v___x_129_;
goto v___jp_108_;
}
}
default: 
{
lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_130_ = lean_unsigned_to_nat(1024u);
v___x_131_ = lean_nat_dec_le(v___x_130_, v_prec_100_);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__6, &lp_mathlib_Mathlib_instReprIneq_repr___closed__6_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__6);
v___y_116_ = v___x_132_;
goto v___jp_115_;
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_obj_once(&lp_mathlib_Mathlib_instReprIneq_repr___closed__7, &lp_mathlib_Mathlib_instReprIneq_repr___closed__7_once, _init_lp_mathlib_Mathlib_instReprIneq_repr___closed__7);
v___y_116_ = v___x_133_;
goto v___jp_115_;
}
}
}
v___jp_101_:
{
lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_103_ = ((lean_object*)(lp_mathlib_Mathlib_instReprIneq_repr___closed__1));
lean_inc(v___y_102_);
v___x_104_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_104_, 0, v___y_102_);
lean_ctor_set(v___x_104_, 1, v___x_103_);
v___x_105_ = 0;
v___x_106_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_106_, 0, v___x_104_);
lean_ctor_set_uint8(v___x_106_, sizeof(void*)*1, v___x_105_);
v___x_107_ = l_Repr_addAppParen(v___x_106_, v_prec_100_);
return v___x_107_;
}
v___jp_108_:
{
lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_110_ = ((lean_object*)(lp_mathlib_Mathlib_instReprIneq_repr___closed__3));
lean_inc(v___y_109_);
v___x_111_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_111_, 0, v___y_109_);
lean_ctor_set(v___x_111_, 1, v___x_110_);
v___x_112_ = 0;
v___x_113_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_113_, 0, v___x_111_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*1, v___x_112_);
v___x_114_ = l_Repr_addAppParen(v___x_113_, v_prec_100_);
return v___x_114_;
}
v___jp_115_:
{
lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Mathlib_instReprIneq_repr___closed__5));
lean_inc(v___y_116_);
v___x_118_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_118_, 0, v___y_116_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = 0;
v___x_120_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_120_, 0, v___x_118_);
lean_ctor_set_uint8(v___x_120_, sizeof(void*)*1, v___x_119_);
v___x_121_ = l_Repr_addAppParen(v___x_120_, v_prec_100_);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instReprIneq_repr___boxed(lean_object* v_x_134_, lean_object* v_prec_135_){
_start:
{
uint8_t v_x_177__boxed_136_; lean_object* v_res_137_; 
v_x_177__boxed_136_ = lean_unbox(v_x_134_);
v_res_137_ = lp_mathlib_Mathlib_instReprIneq_repr(v_x_177__boxed_136_, v_prec_135_);
lean_dec(v_prec_135_);
return v_res_137_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_max(uint8_t v_x_140_, uint8_t v_x_141_){
_start:
{
switch(v_x_140_)
{
case 0:
{
if (v_x_141_ == 0)
{
return v_x_141_;
}
else
{
return v_x_141_;
}
}
case 1:
{
switch(v_x_141_)
{
case 2:
{
return v_x_141_;
}
case 1:
{
return v_x_141_;
}
default: 
{
return v_x_140_;
}
}
}
default: 
{
return v_x_140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_max___boxed(lean_object* v_x_142_, lean_object* v_x_143_){
_start:
{
uint8_t v_x_44__boxed_144_; uint8_t v_x_45__boxed_145_; uint8_t v_res_146_; lean_object* v_r_147_; 
v_x_44__boxed_144_ = lean_unbox(v_x_142_);
v_x_45__boxed_145_ = lean_unbox(v_x_143_);
v_res_146_ = lp_mathlib_Mathlib_Ineq_max(v_x_44__boxed_144_, v_x_45__boxed_145_);
v_r_147_ = lean_box(v_res_146_);
return v_r_147_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Ineq_cmp(uint8_t v_x_148_, uint8_t v_x_149_){
_start:
{
switch(v_x_148_)
{
case 0:
{
if (v_x_149_ == 0)
{
uint8_t v___x_150_; 
v___x_150_ = 1;
return v___x_150_;
}
else
{
uint8_t v___x_151_; 
v___x_151_ = 0;
return v___x_151_;
}
}
case 1:
{
switch(v_x_149_)
{
case 1:
{
uint8_t v___x_152_; 
v___x_152_ = 1;
return v___x_152_;
}
case 2:
{
uint8_t v___x_153_; 
v___x_153_ = 0;
return v___x_153_;
}
default: 
{
uint8_t v___x_154_; 
v___x_154_ = 2;
return v___x_154_;
}
}
}
default: 
{
if (v_x_149_ == 2)
{
uint8_t v___x_155_; 
v___x_155_ = 1;
return v___x_155_;
}
else
{
uint8_t v___x_156_; 
v___x_156_ = 2;
return v___x_156_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_cmp___boxed(lean_object* v_x_157_, lean_object* v_x_158_){
_start:
{
uint8_t v_x_63__boxed_159_; uint8_t v_x_64__boxed_160_; uint8_t v_res_161_; lean_object* v_r_162_; 
v_x_63__boxed_159_ = lean_unbox(v_x_157_);
v_x_64__boxed_160_ = lean_unbox(v_x_158_);
v_res_161_ = lp_mathlib_Mathlib_Ineq_cmp(v_x_63__boxed_159_, v_x_64__boxed_160_);
v_r_162_ = lean_box(v_res_161_);
return v_r_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toString(uint8_t v_x_166_){
_start:
{
switch(v_x_166_)
{
case 0:
{
lean_object* v___x_167_; 
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toString___closed__0));
return v___x_167_;
}
case 1:
{
lean_object* v___x_168_; 
v___x_168_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toString___closed__1));
return v___x_168_;
}
default: 
{
lean_object* v___x_169_; 
v___x_169_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toString___closed__2));
return v___x_169_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toString___boxed(lean_object* v_x_170_){
_start:
{
uint8_t v_x_31__boxed_171_; lean_object* v_res_172_; 
v_x_31__boxed_171_ = lean_unbox(v_x_170_);
v_res_172_ = lp_mathlib_Mathlib_Ineq_toString(v_x_31__boxed_171_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_instToFormat___lam__0(uint8_t v_i_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lp_mathlib_Mathlib_Ineq_toString(v_i_175_);
v___x_177_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_instToFormat___lam__0___boxed(lean_object* v_i_178_){
_start:
{
uint8_t v_i_boxed_179_; lean_object* v_res_180_; 
v_i_boxed_179_ = lean_unbox(v_i_178_);
v_res_180_ = lp_mathlib_Mathlib_Ineq_instToFormat___lam__0(v_i_boxed_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg(lean_object* v_e_183_, lean_object* v___y_184_){
_start:
{
uint8_t v___x_186_; 
v___x_186_ = l_Lean_Expr_hasMVar(v_e_183_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; 
v___x_187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_187_, 0, v_e_183_);
return v___x_187_;
}
else
{
lean_object* v___x_188_; lean_object* v_mctx_189_; lean_object* v___x_190_; lean_object* v_fst_191_; lean_object* v_snd_192_; lean_object* v___x_193_; lean_object* v_cache_194_; lean_object* v_zetaDeltaFVarIds_195_; lean_object* v_postponed_196_; lean_object* v_diag_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_206_; 
v___x_188_ = lean_st_ref_get(v___y_184_);
v_mctx_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc_ref(v_mctx_189_);
lean_dec(v___x_188_);
v___x_190_ = l_Lean_instantiateMVarsCore(v_mctx_189_, v_e_183_);
v_fst_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_fst_191_);
v_snd_192_ = lean_ctor_get(v___x_190_, 1);
lean_inc(v_snd_192_);
lean_dec_ref(v___x_190_);
v___x_193_ = lean_st_ref_take(v___y_184_);
v_cache_194_ = lean_ctor_get(v___x_193_, 1);
v_zetaDeltaFVarIds_195_ = lean_ctor_get(v___x_193_, 2);
v_postponed_196_ = lean_ctor_get(v___x_193_, 3);
v_diag_197_ = lean_ctor_get(v___x_193_, 4);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_206_ == 0)
{
lean_object* v_unused_207_; 
v_unused_207_ = lean_ctor_get(v___x_193_, 0);
lean_dec(v_unused_207_);
v___x_199_ = v___x_193_;
v_isShared_200_ = v_isSharedCheck_206_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_diag_197_);
lean_inc(v_postponed_196_);
lean_inc(v_zetaDeltaFVarIds_195_);
lean_inc(v_cache_194_);
lean_dec(v___x_193_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_206_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_202_; 
if (v_isShared_200_ == 0)
{
lean_ctor_set(v___x_199_, 0, v_snd_192_);
v___x_202_ = v___x_199_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_snd_192_);
lean_ctor_set(v_reuseFailAlloc_205_, 1, v_cache_194_);
lean_ctor_set(v_reuseFailAlloc_205_, 2, v_zetaDeltaFVarIds_195_);
lean_ctor_set(v_reuseFailAlloc_205_, 3, v_postponed_196_);
lean_ctor_set(v_reuseFailAlloc_205_, 4, v_diag_197_);
v___x_202_ = v_reuseFailAlloc_205_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = lean_st_ref_set(v___y_184_, v___x_202_);
v___x_204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_204_, 0, v_fst_191_);
return v___x_204_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg___boxed(lean_object* v_e_208_, lean_object* v___y_209_, lean_object* v___y_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg(v_e_208_, v___y_209_);
lean_dec(v___y_209_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0(lean_object* v_e_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg(v_e_212_, v___y_214_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___boxed(lean_object* v_e_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0(v_e_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1(lean_object* v_msgData_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; lean_object* v_env_233_; lean_object* v___x_234_; lean_object* v_mctx_235_; lean_object* v_lctx_236_; lean_object* v_options_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_232_ = lean_st_ref_get(v___y_230_);
v_env_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc_ref(v_env_233_);
lean_dec(v___x_232_);
v___x_234_ = lean_st_ref_get(v___y_228_);
v_mctx_235_ = lean_ctor_get(v___x_234_, 0);
lean_inc_ref(v_mctx_235_);
lean_dec(v___x_234_);
v_lctx_236_ = lean_ctor_get(v___y_227_, 2);
v_options_237_ = lean_ctor_get(v___y_229_, 2);
lean_inc_ref(v_options_237_);
lean_inc_ref(v_lctx_236_);
v___x_238_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_238_, 0, v_env_233_);
lean_ctor_set(v___x_238_, 1, v_mctx_235_);
lean_ctor_set(v___x_238_, 2, v_lctx_236_);
lean_ctor_set(v___x_238_, 3, v_options_237_);
v___x_239_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
lean_ctor_set(v___x_239_, 1, v_msgData_226_);
v___x_240_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1___boxed(lean_object* v_msgData_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1(v_msgData_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(lean_object* v_msg_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
lean_object* v_ref_254_; lean_object* v___x_255_; lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_264_; 
v_ref_254_ = lean_ctor_get(v___y_251_, 5);
v___x_255_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1_spec__1(v_msg_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
v_a_256_ = lean_ctor_get(v___x_255_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_264_ == 0)
{
v___x_258_ = v___x_255_;
v_isShared_259_ = v_isSharedCheck_264_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_255_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_264_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_260_; lean_object* v___x_262_; 
lean_inc(v_ref_254_);
v___x_260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_260_, 0, v_ref_254_);
lean_ctor_set(v___x_260_, 1, v_a_256_);
if (v_isShared_259_ == 0)
{
lean_ctor_set_tag(v___x_258_, 1);
lean_ctor_set(v___x_258_, 0, v___x_260_);
v___x_262_ = v___x_258_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v___x_260_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg___boxed(lean_object* v_msg_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(v_msg_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
return v_res_271_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_ineq_x3f___closed__9(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_286_ = ((lean_object*)(lp_mathlib_Lean_Expr_ineq_x3f___closed__8));
v___x_287_ = l_Lean_stringToMessageData(v___x_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineq_x3f(lean_object* v_e_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_){
_start:
{
lean_object* v___x_294_; lean_object* v_a_295_; lean_object* v___x_296_; 
v___x_294_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Expr_ineq_x3f_spec__0___redArg(v_e_288_, v_a_290_);
v_a_295_ = lean_ctor_get(v___x_294_, 0);
lean_inc(v_a_295_);
lean_dec_ref(v___x_294_);
v___x_296_ = l_Lean_Meta_whnfR(v_a_295_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_354_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_354_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_354_ == 0)
{
v___x_299_ = v___x_296_;
v_isShared_300_ = v_isSharedCheck_354_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_a_297_);
lean_dec(v___x_296_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_354_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
lean_object* v___x_301_; lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_301_ = ((lean_object*)(lp_mathlib_Lean_Expr_ineq_x3f___closed__1));
v___x_302_ = lean_unsigned_to_nat(3u);
v___x_303_ = l_Lean_Expr_isAppOfArity(v_a_297_, v___x_301_, v___x_302_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; lean_object* v___x_305_; uint8_t v___x_306_; 
v___x_304_ = ((lean_object*)(lp_mathlib_Lean_Expr_ineq_x3f___closed__4));
v___x_305_ = lean_unsigned_to_nat(4u);
v___x_306_ = l_Lean_Expr_isAppOfArity(v_a_297_, v___x_304_, v___x_305_);
if (v___x_306_ == 0)
{
lean_object* v___x_307_; uint8_t v___x_308_; 
v___x_307_ = ((lean_object*)(lp_mathlib_Lean_Expr_ineq_x3f___closed__7));
v___x_308_ = l_Lean_Expr_isAppOfArity(v_a_297_, v___x_307_, v___x_305_);
if (v___x_308_ == 0)
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
lean_del_object(v___x_299_);
v___x_309_ = lean_obj_once(&lp_mathlib_Lean_Expr_ineq_x3f___closed__9, &lp_mathlib_Lean_Expr_ineq_x3f___closed__9_once, _init_lp_mathlib_Lean_Expr_ineq_x3f___closed__9);
v___x_310_ = l_Lean_MessageData_ofExpr(v_a_297_);
v___x_311_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_309_);
lean_ctor_set(v___x_311_, 1, v___x_310_);
v___x_312_ = lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(v___x_311_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
return v___x_312_;
}
else
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; uint8_t v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_325_; 
v___x_313_ = l_Lean_Expr_appFn_x21(v_a_297_);
v___x_314_ = l_Lean_Expr_appFn_x21(v___x_313_);
v___x_315_ = l_Lean_Expr_appFn_x21(v___x_314_);
lean_dec_ref(v___x_314_);
v___x_316_ = l_Lean_Expr_appArg_x21(v___x_315_);
lean_dec_ref(v___x_315_);
v___x_317_ = l_Lean_Expr_appArg_x21(v___x_313_);
lean_dec_ref(v___x_313_);
v___x_318_ = l_Lean_Expr_appArg_x21(v_a_297_);
lean_dec(v_a_297_);
v___x_319_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_319_, 0, v___x_317_);
lean_ctor_set(v___x_319_, 1, v___x_318_);
v___x_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_316_);
lean_ctor_set(v___x_320_, 1, v___x_319_);
v___x_321_ = 2;
v___x_322_ = lean_box(v___x_321_);
v___x_323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
lean_ctor_set(v___x_323_, 1, v___x_320_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v___x_323_);
v___x_325_ = v___x_299_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v___x_323_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
else
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; uint8_t v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_339_; 
v___x_327_ = l_Lean_Expr_appFn_x21(v_a_297_);
v___x_328_ = l_Lean_Expr_appFn_x21(v___x_327_);
v___x_329_ = l_Lean_Expr_appFn_x21(v___x_328_);
lean_dec_ref(v___x_328_);
v___x_330_ = l_Lean_Expr_appArg_x21(v___x_329_);
lean_dec_ref(v___x_329_);
v___x_331_ = l_Lean_Expr_appArg_x21(v___x_327_);
lean_dec_ref(v___x_327_);
v___x_332_ = l_Lean_Expr_appArg_x21(v_a_297_);
lean_dec(v_a_297_);
v___x_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_331_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_330_);
lean_ctor_set(v___x_334_, 1, v___x_333_);
v___x_335_ = 1;
v___x_336_ = lean_box(v___x_335_);
v___x_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
lean_ctor_set(v___x_337_, 1, v___x_334_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v___x_337_);
v___x_339_ = v___x_299_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_337_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
else
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; uint8_t v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_352_; 
v___x_341_ = l_Lean_Expr_appFn_x21(v_a_297_);
v___x_342_ = l_Lean_Expr_appFn_x21(v___x_341_);
v___x_343_ = l_Lean_Expr_appArg_x21(v___x_342_);
lean_dec_ref(v___x_342_);
v___x_344_ = l_Lean_Expr_appArg_x21(v___x_341_);
lean_dec_ref(v___x_341_);
v___x_345_ = l_Lean_Expr_appArg_x21(v_a_297_);
lean_dec(v_a_297_);
v___x_346_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_344_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_343_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = 0;
v___x_349_ = lean_box(v___x_348_);
v___x_350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
lean_ctor_set(v___x_350_, 1, v___x_347_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v___x_350_);
v___x_352_ = v___x_299_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v___x_350_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
else
{
lean_object* v_a_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_362_; 
v_a_355_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_362_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_362_ == 0)
{
v___x_357_ = v___x_296_;
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_a_355_);
lean_dec(v___x_296_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_360_; 
if (v_isShared_358_ == 0)
{
v___x_360_ = v___x_357_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_a_355_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineq_x3f___boxed(lean_object* v_e_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_, lean_object* v_a_367_, lean_object* v_a_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Lean_Expr_ineq_x3f(v_e_363_, v_a_364_, v_a_365_, v_a_366_, v_a_367_);
lean_dec(v_a_367_);
lean_dec_ref(v_a_366_);
lean_dec(v_a_365_);
lean_dec_ref(v_a_364_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1(lean_object* v_00_u03b1_370_, lean_object* v_msg_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(v_msg_371_, v___y_372_, v___y_373_, v___y_374_, v___y_375_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___boxed(lean_object* v_00_u03b1_378_, lean_object* v_msg_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1(v_00_u03b1_378_, v_msg_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(lean_object* v_e_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_){
_start:
{
lean_object* v___x_395_; 
lean_inc_ref(v_e_389_);
v___x_395_ = lp_mathlib_Lean_Expr_ineq_x3f(v_e_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_);
if (lean_obj_tag(v___x_395_) == 0)
{
lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_406_; 
lean_dec_ref(v_e_389_);
v_a_396_ = lean_ctor_get(v___x_395_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_395_);
if (v_isSharedCheck_406_ == 0)
{
v___x_398_ = v___x_395_;
v_isShared_399_ = v_isSharedCheck_406_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_395_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_406_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
uint8_t v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_404_; 
v___x_400_ = 1;
v___x_401_ = lean_box(v___x_400_);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v_a_396_);
if (v_isShared_399_ == 0)
{
lean_ctor_set(v___x_398_, 0, v___x_402_);
v___x_404_ = v___x_398_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v___x_402_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
else
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_445_; 
v_a_407_ = lean_ctor_get(v___x_395_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_395_);
if (v_isSharedCheck_445_ == 0)
{
v___x_409_ = v___x_395_;
v_isShared_410_ = v_isSharedCheck_445_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_395_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_445_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
uint8_t v___y_412_; uint8_t v___x_443_; 
v___x_443_ = l_Lean_Exception_isInterrupt(v_a_407_);
if (v___x_443_ == 0)
{
uint8_t v___x_444_; 
lean_inc(v_a_407_);
v___x_444_ = l_Lean_Exception_isRuntime(v_a_407_);
v___y_412_ = v___x_444_;
goto v___jp_411_;
}
else
{
v___y_412_ = v___x_443_;
goto v___jp_411_;
}
v___jp_411_:
{
if (v___y_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___x_414_; uint8_t v___x_415_; 
lean_del_object(v___x_409_);
lean_dec(v_a_407_);
v___x_413_ = ((lean_object*)(lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___closed__1));
v___x_414_ = lean_unsigned_to_nat(1u);
v___x_415_ = l_Lean_Expr_isAppOfArity(v_e_389_, v___x_413_, v___x_414_);
if (v___x_415_ == 0)
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_416_ = lean_obj_once(&lp_mathlib_Lean_Expr_ineq_x3f___closed__9, &lp_mathlib_Lean_Expr_ineq_x3f___closed__9_once, _init_lp_mathlib_Lean_Expr_ineq_x3f___closed__9);
v___x_417_ = l_Lean_MessageData_ofExpr(v_e_389_);
v___x_418_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_418_, 0, v___x_416_);
lean_ctor_set(v___x_418_, 1, v___x_417_);
v___x_419_ = lp_mathlib_Lean_throwError___at___00Lean_Expr_ineq_x3f_spec__1___redArg(v___x_418_, v_a_390_, v_a_391_, v_a_392_, v_a_393_);
return v___x_419_;
}
else
{
lean_object* v___x_420_; lean_object* v___x_421_; 
v___x_420_ = l_Lean_Expr_appArg_x21(v_e_389_);
lean_dec_ref(v_e_389_);
v___x_421_ = lp_mathlib_Lean_Expr_ineq_x3f(v___x_420_, v_a_390_, v_a_391_, v_a_392_, v_a_393_);
if (lean_obj_tag(v___x_421_) == 0)
{
lean_object* v_a_422_; lean_object* v___x_424_; uint8_t v_isShared_425_; uint8_t v_isSharedCheck_431_; 
v_a_422_ = lean_ctor_get(v___x_421_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_421_);
if (v_isSharedCheck_431_ == 0)
{
v___x_424_ = v___x_421_;
v_isShared_425_ = v_isSharedCheck_431_;
goto v_resetjp_423_;
}
else
{
lean_inc(v_a_422_);
lean_dec(v___x_421_);
v___x_424_ = lean_box(0);
v_isShared_425_ = v_isSharedCheck_431_;
goto v_resetjp_423_;
}
v_resetjp_423_:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_429_; 
v___x_426_ = lean_box(v___y_412_);
v___x_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_427_, 0, v___x_426_);
lean_ctor_set(v___x_427_, 1, v_a_422_);
if (v_isShared_425_ == 0)
{
lean_ctor_set(v___x_424_, 0, v___x_427_);
v___x_429_ = v___x_424_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_427_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
v_a_432_ = lean_ctor_get(v___x_421_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_421_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_421_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_421_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
}
else
{
lean_object* v___x_441_; 
lean_dec_ref(v_e_389_);
if (v_isShared_410_ == 0)
{
v___x_441_ = v___x_409_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_442_; 
v_reuseFailAlloc_442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_442_, 0, v_a_407_);
v___x_441_ = v_reuseFailAlloc_442_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
return v___x_441_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f___boxed(lean_object* v_e_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_Lean_Expr_ineqOrNotIneq_x3f(v_e_446_, v_a_447_, v_a_448_, v_a_449_, v_a_450_);
lean_dec(v_a_450_);
lean_dec_ref(v_a_449_);
lean_dec(v_a_448_);
lean_dec_ref(v_a_447_);
return v_res_452_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_instInhabitedIneq_default = _init_lp_mathlib_Mathlib_instInhabitedIneq_default();
lp_mathlib_Mathlib_instInhabitedIneq = _init_lp_mathlib_Mathlib_instInhabitedIneq();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Ineq(builtin);
}
#ifdef __cplusplus
}
#endif
