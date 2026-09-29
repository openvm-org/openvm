// Lean compiler output
// Module: Qq.ForLean.ReduceEval
// Imports: public import Init public meta import Init public import Lean.Meta.ReduceEval
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_instReduceEvalNat___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getRevArg_x21(lean_object*, lean_object*);
lean_object* l_Lean_Meta_reduceEval___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instReduceEvalName___private__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instReduceEvalString___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
uint64_t lean_uint64_of_nat_mk(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "reduceEval: failed to evaluate argument"};
static const lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__0_value;
static lean_once_cell_t lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Meta_evalList___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_Qq_Lean_Meta_evalList___redArg___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_evalList___redArg___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_evalList___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_Qq_Lean_Meta_evalList___redArg___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_evalList___redArg___closed__1_value;
static const lean_string_object lp_Qq_Lean_Meta_evalList___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_Qq_Lean_Meta_evalList___redArg___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_evalList___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalList__qq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalList__qq(lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(30, 240, 210, 97, 67, 170, 216, 80)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2_value;
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instReduceEvalNat___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "BitVec"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofFin"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 178, 58, 132, 143, 189, 222, 74)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 167, 55, 152, 45, 146, 42, 51)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq(lean_object*);
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "UInt64"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ofBitVec"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(58, 113, 45, 150, 103, 228, 0, 41)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 53, 147, 133, 131, 240, 238, 68)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2_value;
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___boxed, .m_arity = 7, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(64) << 1) | 1))} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "USize"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 217, 26, 131, 232, 198, 207, 245)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 209, 178, 30, 88, 155, 129, 160)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalUSize__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalUSize__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__3_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalBool__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBool__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "BinderInfo"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "implicit"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__3_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "strictImplicit"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__4 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__4_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instImplicit"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__5 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Literal"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "natVal"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value_aux_1),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(64, 199, 201, 37, 137, 51, 1, 129)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strVal"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__3_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value_aux_1),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(68, 214, 249, 146, 84, 160, 212, 27)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4_value;
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instReduceEvalString___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__5 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MVarId"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 186, 234, 138, 172, 166, 87, 74)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value_aux_1),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(93, 44, 60, 136, 72, 250, 230, 141)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1_value;
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instReduceEvalName___private__1___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LevelMVarId"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 60, 85, 89, 175, 240, 129, 147)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value_aux_1),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(213, 157, 226, 48, 182, 72, 20, 234)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___closed__0_value;
static const lean_string_object lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "FVarId"};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(134, 80, 170, 214, 218, 146, 55, 86)}};
static const lean_ctor_object lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value_aux_1),((lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(246, 212, 153, 136, 172, 214, 179, 96)}};
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___closed__0 = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq = (const lean_object*)&lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = ((lean_object*)(lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__0));
v___x_49_ = l_Lean_stringToMessageData(v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg(lean_object* v_e_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = lean_obj_once(&lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1, &lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1_once, _init_lp_Qq_Lean_Meta_throwFailedToEval___redArg___closed__1);
v___x_57_ = l_Lean_indentExpr(v_e_50_);
v___x_58_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_56_);
lean_ctor_set(v___x_58_, 1, v___x_57_);
v___x_59_ = lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg(v___x_58_, v_a_51_, v_a_52_, v_a_53_, v_a_54_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___redArg___boxed(lean_object* v_e_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_60_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
lean_dec(v_a_64_);
lean_dec_ref(v_a_63_);
lean_dec(v_a_62_);
lean_dec_ref(v_a_61_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval(lean_object* v_00_u03b1_67_, lean_object* v_e_68_, lean_object* v_a_69_, lean_object* v_a_70_, lean_object* v_a_71_, lean_object* v_a_72_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_68_, v_a_69_, v_a_70_, v_a_71_, v_a_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_throwFailedToEval___boxed(lean_object* v_00_u03b1_75_, lean_object* v_e_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_Qq_Lean_Meta_throwFailedToEval(v_00_u03b1_75_, v_e_76_, v_a_77_, v_a_78_, v_a_79_, v_a_80_);
lean_dec(v_a_80_);
lean_dec_ref(v_a_79_);
lean_dec(v_a_78_);
lean_dec_ref(v_a_77_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0(lean_object* v_00_u03b1_83_, lean_object* v_msg_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___redArg(v_msg_84_, v___y_85_, v___y_86_, v___y_87_, v___y_88_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0___boxed(lean_object* v_00_u03b1_91_, lean_object* v_msg_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_Qq_Lean_throwError___at___00Lean_Meta_throwFailedToEval_spec__0(v_00_u03b1_91_, v_msg_92_, v___y_93_, v___y_94_, v___y_95_, v___y_96_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___redArg(lean_object* v_inst_102_, lean_object* v_e_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___x_109_; 
lean_inc(v_a_107_);
lean_inc_ref(v_a_106_);
lean_inc(v_a_105_);
lean_inc_ref(v_a_104_);
v___x_109_ = lean_whnf(v_e_103_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
if (lean_obj_tag(v___x_109_) == 0)
{
lean_object* v_a_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_171_; 
v_a_110_ = lean_ctor_get(v___x_109_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_171_ == 0)
{
v___x_112_ = v___x_109_;
v_isShared_113_ = v_isSharedCheck_171_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_a_110_);
lean_dec(v___x_109_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_171_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_114_; 
v___x_114_ = l_Lean_Expr_getAppFn(v_a_110_);
if (lean_obj_tag(v___x_114_) == 4)
{
lean_object* v_declName_115_; 
v_declName_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_declName_115_);
lean_dec_ref_known(v___x_114_, 2);
if (lean_obj_tag(v_declName_115_) == 1)
{
lean_object* v_pre_116_; 
v_pre_116_ = lean_ctor_get(v_declName_115_, 0);
lean_inc(v_pre_116_);
if (lean_obj_tag(v_pre_116_) == 1)
{
lean_object* v_pre_117_; 
v_pre_117_ = lean_ctor_get(v_pre_116_, 0);
if (lean_obj_tag(v_pre_117_) == 0)
{
lean_object* v_str_118_; lean_object* v_str_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v_str_118_ = lean_ctor_get(v_declName_115_, 1);
lean_inc_ref(v_str_118_);
lean_dec_ref_known(v_declName_115_, 2);
v_str_119_ = lean_ctor_get(v_pre_116_, 1);
lean_inc_ref(v_str_119_);
lean_dec_ref_known(v_pre_116_, 2);
v___x_120_ = ((lean_object*)(lp_Qq_Lean_Meta_evalList___redArg___closed__0));
v___x_121_ = lean_string_dec_eq(v_str_119_, v___x_120_);
lean_dec_ref(v_str_119_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
lean_dec_ref(v_str_118_);
lean_del_object(v___x_112_);
lean_dec_ref(v_inst_102_);
v___x_122_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_123_ = l_Lean_Expr_getAppNumArgs(v_a_110_);
v___x_124_ = ((lean_object*)(lp_Qq_Lean_Meta_evalList___redArg___closed__1));
v___x_125_ = lean_string_dec_eq(v_str_118_, v___x_124_);
if (v___x_125_ == 0)
{
lean_object* v___x_126_; uint8_t v___x_127_; 
lean_del_object(v___x_112_);
v___x_126_ = ((lean_object*)(lp_Qq_Lean_Meta_evalList___redArg___closed__2));
v___x_127_ = lean_string_dec_eq(v_str_118_, v___x_126_);
lean_dec_ref(v_str_118_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; 
lean_dec(v___x_123_);
lean_dec_ref(v_inst_102_);
v___x_128_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; uint8_t v___x_130_; 
v___x_129_ = lean_unsigned_to_nat(3u);
v___x_130_ = lean_nat_dec_eq(v___x_123_, v___x_129_);
if (v___x_130_ == 0)
{
lean_object* v___x_131_; 
lean_dec(v___x_123_);
lean_dec_ref(v_inst_102_);
v___x_131_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_131_;
}
else
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_132_ = lean_unsigned_to_nat(1u);
v___x_133_ = lean_nat_sub(v___x_123_, v___x_132_);
v___x_134_ = lean_nat_sub(v___x_133_, v___x_132_);
lean_dec(v___x_133_);
v___x_135_ = l_Lean_Expr_getRevArg_x21(v_a_110_, v___x_134_);
lean_inc_ref(v_inst_102_);
v___x_136_ = l_Lean_Meta_reduceEval___redArg(v_inst_102_, v___x_135_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
if (lean_obj_tag(v___x_136_) == 0)
{
lean_object* v_a_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_a_137_ = lean_ctor_get(v___x_136_, 0);
lean_inc(v_a_137_);
lean_dec_ref_known(v___x_136_, 1);
v___x_138_ = lean_unsigned_to_nat(2u);
v___x_139_ = lean_nat_sub(v___x_123_, v___x_138_);
lean_dec(v___x_123_);
v___x_140_ = lean_nat_sub(v___x_139_, v___x_132_);
lean_dec(v___x_139_);
v___x_141_ = l_Lean_Expr_getRevArg_x21(v_a_110_, v___x_140_);
lean_dec(v_a_110_);
v___x_142_ = lp_Qq_Lean_Meta_evalList___redArg(v_inst_102_, v___x_141_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v_a_143_; lean_object* v___x_145_; uint8_t v_isShared_146_; uint8_t v_isSharedCheck_151_; 
v_a_143_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_151_ == 0)
{
v___x_145_ = v___x_142_;
v_isShared_146_ = v_isSharedCheck_151_;
goto v_resetjp_144_;
}
else
{
lean_inc(v_a_143_);
lean_dec(v___x_142_);
v___x_145_ = lean_box(0);
v_isShared_146_ = v_isSharedCheck_151_;
goto v_resetjp_144_;
}
v_resetjp_144_:
{
lean_object* v___x_147_; lean_object* v___x_149_; 
v___x_147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_147_, 0, v_a_137_);
lean_ctor_set(v___x_147_, 1, v_a_143_);
if (v_isShared_146_ == 0)
{
lean_ctor_set(v___x_145_, 0, v___x_147_);
v___x_149_ = v___x_145_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v___x_147_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
else
{
lean_dec(v_a_137_);
return v___x_142_;
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
lean_dec(v___x_123_);
lean_dec(v_a_110_);
lean_dec_ref(v_inst_102_);
v_a_152_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_136_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_136_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
}
}
else
{
lean_object* v___x_160_; uint8_t v___x_161_; 
lean_dec_ref(v_str_118_);
lean_dec_ref(v_inst_102_);
v___x_160_ = lean_unsigned_to_nat(1u);
v___x_161_ = lean_nat_dec_eq(v___x_123_, v___x_160_);
lean_dec(v___x_123_);
if (v___x_161_ == 0)
{
lean_object* v___x_162_; 
lean_del_object(v___x_112_);
v___x_162_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_162_;
}
else
{
lean_object* v___x_163_; lean_object* v___x_165_; 
lean_dec(v_a_110_);
v___x_163_ = lean_box(0);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 0, v___x_163_);
v___x_165_ = v___x_112_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
}
else
{
lean_object* v___x_167_; 
lean_dec_ref_known(v_pre_116_, 2);
lean_dec_ref_known(v_declName_115_, 2);
lean_del_object(v___x_112_);
lean_dec_ref(v_inst_102_);
v___x_167_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_167_;
}
}
else
{
lean_object* v___x_168_; 
lean_dec_ref_known(v_declName_115_, 2);
lean_dec(v_pre_116_);
lean_del_object(v___x_112_);
lean_dec_ref(v_inst_102_);
v___x_168_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_168_;
}
}
else
{
lean_object* v___x_169_; 
lean_dec(v_declName_115_);
lean_del_object(v___x_112_);
lean_dec_ref(v_inst_102_);
v___x_169_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_169_;
}
}
else
{
lean_object* v___x_170_; 
lean_dec_ref(v___x_114_);
lean_del_object(v___x_112_);
lean_dec_ref(v_inst_102_);
v___x_170_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_110_, v_a_104_, v_a_105_, v_a_106_, v_a_107_);
return v___x_170_;
}
}
}
else
{
lean_object* v_a_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
lean_dec_ref(v_inst_102_);
v_a_172_ = lean_ctor_get(v___x_109_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_109_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_a_172_);
lean_dec(v___x_109_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_a_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___redArg___boxed(lean_object* v_inst_180_, lean_object* v_e_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_Qq_Lean_Meta_evalList___redArg(v_inst_180_, v_e_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_);
lean_dec(v_a_185_);
lean_dec_ref(v_a_184_);
lean_dec(v_a_183_);
lean_dec_ref(v_a_182_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList(lean_object* v_00_u03b1_188_, lean_object* v_inst_189_, lean_object* v_e_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_Qq_Lean_Meta_evalList___redArg(v_inst_189_, v_e_190_, v_a_191_, v_a_192_, v_a_193_, v_a_194_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_evalList___boxed(lean_object* v_00_u03b1_197_, lean_object* v_inst_198_, lean_object* v_e_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_Qq_Lean_Meta_evalList(v_00_u03b1_197_, v_inst_198_, v_e_199_, v_a_200_, v_a_201_, v_a_202_, v_a_203_);
lean_dec(v_a_203_);
lean_dec_ref(v_a_202_);
lean_dec(v_a_201_);
lean_dec_ref(v_a_200_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalList__qq___redArg(lean_object* v_inst_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_evalList___boxed), 8, 2);
lean_closure_set(v___x_207_, 0, lean_box(0));
lean_closure_set(v___x_207_, 1, v_inst_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalList__qq(lean_object* v_00_u03b1_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_evalList___boxed), 8, 2);
lean_closure_set(v___x_210_, 0, lean_box(0));
lean_closure_set(v___x_210_, 1, v_inst_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0(lean_object* v_n_217_, lean_object* v_e_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v___x_224_; 
lean_inc(v___y_222_);
lean_inc_ref(v___y_221_);
lean_inc(v___y_220_);
lean_inc_ref(v___y_219_);
v___x_224_ = lean_whnf(v_e_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
if (lean_obj_tag(v___x_224_) == 0)
{
lean_object* v_a_225_; lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v_a_225_ = lean_ctor_get(v___x_224_, 0);
lean_inc(v_a_225_);
lean_dec_ref_known(v___x_224_, 1);
v___x_226_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2));
v___x_227_ = lean_unsigned_to_nat(3u);
v___x_228_ = l_Lean_Expr_isAppOfArity(v_a_225_, v___x_226_, v___x_227_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; 
v___x_229_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_225_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
return v___x_229_;
}
else
{
lean_object* v___f_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___f_230_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3));
v___x_231_ = lean_unsigned_to_nat(1u);
v___x_232_ = l_Lean_Expr_getAppNumArgs(v_a_225_);
v___x_233_ = lean_nat_sub(v___x_232_, v___x_231_);
lean_dec(v___x_232_);
v___x_234_ = lean_nat_sub(v___x_233_, v___x_231_);
lean_dec(v___x_233_);
v___x_235_ = l_Lean_Expr_getRevArg_x21(v_a_225_, v___x_234_);
lean_dec(v_a_225_);
v___x_236_ = l_Lean_Meta_reduceEval___redArg(v___f_230_, v___x_235_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v_a_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_245_; 
v_a_237_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_245_ == 0)
{
v___x_239_ = v___x_236_;
v_isShared_240_ = v_isSharedCheck_245_;
goto v_resetjp_238_;
}
else
{
lean_inc(v_a_237_);
lean_dec(v___x_236_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_245_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_241_; lean_object* v___x_243_; 
v___x_241_ = lean_nat_mod(v_a_237_, v_n_217_);
lean_dec(v_a_237_);
if (v_isShared_240_ == 0)
{
lean_ctor_set(v___x_239_, 0, v___x_241_);
v___x_243_ = v___x_239_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_241_);
v___x_243_ = v_reuseFailAlloc_244_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
return v___x_243_;
}
}
}
else
{
lean_object* v_a_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_253_; 
v_a_246_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_253_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_253_ == 0)
{
v___x_248_ = v___x_236_;
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_a_246_);
lean_dec(v___x_236_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
lean_object* v___x_251_; 
if (v_isShared_249_ == 0)
{
v___x_251_ = v___x_248_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v_a_246_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
return v___x_251_;
}
}
}
}
}
else
{
lean_object* v_a_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_261_; 
v_a_254_ = lean_ctor_get(v___x_224_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_261_ == 0)
{
v___x_256_ = v___x_224_;
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_a_254_);
lean_dec(v___x_224_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_259_; 
if (v_isShared_257_ == 0)
{
v___x_259_ = v___x_256_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v_a_254_);
v___x_259_ = v_reuseFailAlloc_260_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
return v___x_259_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___boxed(lean_object* v_n_262_, lean_object* v_e_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0(v_n_262_, v_e_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v_n_262_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg(lean_object* v_n_270_){
_start:
{
lean_object* v___f_271_; 
v___f_271_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_271_, 0, v_n_270_);
return v___f_271_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq(lean_object* v_n_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v___f_274_; 
v___f_274_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_274_, 0, v_n_272_);
return v___f_274_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0(lean_object* v___x_275_, lean_object* v_n_276_, lean_object* v_e_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v___x_283_; 
lean_inc(v___y_281_);
lean_inc_ref(v___y_280_);
lean_inc(v___y_279_);
lean_inc_ref(v___y_278_);
v___x_283_ = lean_whnf(v_e_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v_a_284_; lean_object* v___x_285_; lean_object* v___x_286_; uint8_t v___x_287_; 
v_a_284_ = lean_ctor_get(v___x_283_, 0);
lean_inc(v_a_284_);
lean_dec_ref_known(v___x_283_, 1);
v___x_285_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2));
v___x_286_ = lean_unsigned_to_nat(3u);
v___x_287_ = l_Lean_Expr_isAppOfArity(v_a_284_, v___x_285_, v___x_286_);
if (v___x_287_ == 0)
{
lean_object* v___x_288_; 
v___x_288_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_284_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
return v___x_288_;
}
else
{
lean_object* v___f_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___f_289_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3));
v___x_290_ = lean_unsigned_to_nat(1u);
v___x_291_ = l_Lean_Expr_getAppNumArgs(v_a_284_);
v___x_292_ = lean_nat_sub(v___x_291_, v___x_290_);
lean_dec(v___x_291_);
v___x_293_ = lean_nat_sub(v___x_292_, v___x_290_);
lean_dec(v___x_292_);
v___x_294_ = l_Lean_Expr_getRevArg_x21(v_a_284_, v___x_293_);
lean_dec(v_a_284_);
v___x_295_ = l_Lean_Meta_reduceEval___redArg(v___f_289_, v___x_294_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
if (lean_obj_tag(v___x_295_) == 0)
{
lean_object* v_a_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_307_; 
v_a_296_ = lean_ctor_get(v___x_295_, 0);
v_isSharedCheck_307_ = !lean_is_exclusive(v___x_295_);
if (v_isSharedCheck_307_ == 0)
{
v___x_298_ = v___x_295_;
v_isShared_299_ = v_isSharedCheck_307_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_a_296_);
lean_dec(v___x_295_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_307_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_305_; 
v___x_300_ = lean_nat_pow(v___x_275_, v_n_276_);
v___x_301_ = lean_nat_sub(v___x_300_, v___x_290_);
lean_dec(v___x_300_);
v___x_302_ = lean_nat_add(v___x_301_, v___x_290_);
lean_dec(v___x_301_);
v___x_303_ = lean_nat_mod(v_a_296_, v___x_302_);
lean_dec(v___x_302_);
lean_dec(v_a_296_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_303_);
v___x_305_ = v___x_298_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v___x_303_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
else
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_315_; 
v_a_308_ = lean_ctor_get(v___x_295_, 0);
v_isSharedCheck_315_ = !lean_is_exclusive(v___x_295_);
if (v_isSharedCheck_315_ == 0)
{
v___x_310_ = v___x_295_;
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_295_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_313_; 
if (v_isShared_311_ == 0)
{
v___x_313_ = v___x_310_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v_a_308_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
}
else
{
lean_object* v_a_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_323_; 
v_a_316_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_323_ == 0)
{
v___x_318_ = v___x_283_;
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_a_316_);
lean_dec(v___x_283_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_321_; 
if (v_isShared_319_ == 0)
{
v___x_321_ = v___x_318_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_a_316_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0___boxed(lean_object* v___x_324_, lean_object* v_n_325_, lean_object* v_e_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0(v___x_324_, v_n_325_, v_e_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_);
lean_dec(v___y_330_);
lean_dec_ref(v___y_329_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v_n_325_);
lean_dec(v___x_324_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1(lean_object* v_n_338_, lean_object* v_e_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
lean_object* v___x_345_; 
lean_inc(v___y_343_);
lean_inc_ref(v___y_342_);
lean_inc(v___y_341_);
lean_inc_ref(v___y_340_);
v___x_345_ = lean_whnf(v_e_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
if (lean_obj_tag(v___x_345_) == 0)
{
lean_object* v_a_346_; lean_object* v___x_347_; lean_object* v___x_348_; uint8_t v___x_349_; 
v_a_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc(v_a_346_);
lean_dec_ref_known(v___x_345_, 1);
v___x_347_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___closed__2));
v___x_348_ = lean_unsigned_to_nat(2u);
v___x_349_ = l_Lean_Expr_isAppOfArity(v_a_346_, v___x_347_, v___x_348_);
if (v___x_349_ == 0)
{
lean_object* v___x_350_; 
lean_dec(v_n_338_);
v___x_350_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_346_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
return v___x_350_;
}
else
{
lean_object* v___f_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___f_351_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__0___boxed), 8, 2);
lean_closure_set(v___f_351_, 0, v___x_348_);
lean_closure_set(v___f_351_, 1, v_n_338_);
v___x_352_ = lean_unsigned_to_nat(1u);
v___x_353_ = l_Lean_Expr_getAppNumArgs(v_a_346_);
v___x_354_ = lean_nat_sub(v___x_353_, v___x_352_);
lean_dec(v___x_353_);
v___x_355_ = lean_nat_sub(v___x_354_, v___x_352_);
lean_dec(v___x_354_);
v___x_356_ = l_Lean_Expr_getRevArg_x21(v_a_346_, v___x_355_);
lean_dec(v_a_346_);
v___x_357_ = l_Lean_Meta_reduceEval___redArg(v___f_351_, v___x_356_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
if (lean_obj_tag(v___x_357_) == 0)
{
lean_object* v_a_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_365_; 
v_a_358_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_365_ == 0)
{
v___x_360_ = v___x_357_;
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_a_358_);
lean_dec(v___x_357_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___x_363_; 
if (v_isShared_361_ == 0)
{
v___x_363_ = v___x_360_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_a_358_);
v___x_363_ = v_reuseFailAlloc_364_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
return v___x_363_;
}
}
}
else
{
lean_object* v_a_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_373_; 
v_a_366_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_373_ == 0)
{
v___x_368_ = v___x_357_;
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_a_366_);
lean_dec(v___x_357_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_371_; 
if (v_isShared_369_ == 0)
{
v___x_371_ = v___x_368_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v_a_366_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
}
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
lean_dec(v_n_338_);
v_a_374_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_345_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_345_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___boxed(lean_object* v_n_382_, lean_object* v_e_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1(v_n_382_, v_e_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBitVec__qq(lean_object* v_n_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_instReduceEvalBitVec__qq___lam__1___boxed), 7, 1);
lean_closure_set(v___f_391_, 0, v_n_390_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0(lean_object* v_e_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_){
_start:
{
lean_object* v___x_405_; 
lean_inc(v___y_403_);
lean_inc_ref(v___y_402_);
lean_inc(v___y_401_);
lean_inc_ref(v___y_400_);
v___x_405_ = lean_whnf(v_e_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
if (lean_obj_tag(v___x_405_) == 0)
{
lean_object* v_a_406_; lean_object* v___x_407_; lean_object* v___x_408_; uint8_t v___x_409_; 
v_a_406_ = lean_ctor_get(v___x_405_, 0);
lean_inc(v_a_406_);
lean_dec_ref_known(v___x_405_, 1);
v___x_407_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__2));
v___x_408_ = lean_unsigned_to_nat(1u);
v___x_409_ = l_Lean_Expr_isAppOfArity(v_a_406_, v___x_407_, v___x_408_);
if (v___x_409_ == 0)
{
lean_object* v___x_410_; 
v___x_410_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_406_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
return v___x_410_;
}
else
{
lean_object* v___f_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___f_411_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___closed__3));
v___x_412_ = l_Lean_Expr_getAppNumArgs(v_a_406_);
v___x_413_ = lean_nat_sub(v___x_412_, v___x_408_);
lean_dec(v___x_412_);
v___x_414_ = l_Lean_Expr_getRevArg_x21(v_a_406_, v___x_413_);
lean_dec(v_a_406_);
v___x_415_ = l_Lean_Meta_reduceEval___redArg(v___f_411_, v___x_414_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
if (lean_obj_tag(v___x_415_) == 0)
{
lean_object* v_a_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_425_; 
v_a_416_ = lean_ctor_get(v___x_415_, 0);
v_isSharedCheck_425_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_425_ == 0)
{
v___x_418_ = v___x_415_;
v_isShared_419_ = v_isSharedCheck_425_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_a_416_);
lean_dec(v___x_415_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_425_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
uint64_t v___x_420_; lean_object* v___x_421_; lean_object* v___x_423_; 
v___x_420_ = lean_uint64_of_nat_mk(v_a_416_);
v___x_421_ = lean_box_uint64(v___x_420_);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 0, v___x_421_);
v___x_423_ = v___x_418_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_424_; 
v_reuseFailAlloc_424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_424_, 0, v___x_421_);
v___x_423_ = v_reuseFailAlloc_424_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
return v___x_423_;
}
}
}
else
{
lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_433_; 
v_a_426_ = lean_ctor_get(v___x_415_, 0);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_415_);
if (v_isSharedCheck_433_ == 0)
{
v___x_428_ = v___x_415_;
v_isShared_429_ = v_isSharedCheck_433_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_415_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_433_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v___x_431_; 
if (v_isShared_429_ == 0)
{
v___x_431_ = v___x_428_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_432_; 
v_reuseFailAlloc_432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_432_, 0, v_a_426_);
v___x_431_ = v_reuseFailAlloc_432_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
return v___x_431_;
}
}
}
}
}
else
{
lean_object* v_a_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_441_; 
v_a_434_ = lean_ctor_get(v___x_405_, 0);
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_405_);
if (v_isSharedCheck_441_ == 0)
{
v___x_436_ = v___x_405_;
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_a_434_);
lean_dec(v___x_405_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_439_; 
if (v_isShared_437_ == 0)
{
v___x_439_ = v___x_436_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v_a_434_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0___boxed(lean_object* v_e_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_Qq_Lean_Meta_instReduceEvalUInt64__qq___lam__0(v_e_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_);
lean_dec(v___y_446_);
lean_dec_ref(v___y_445_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0(lean_object* v_e_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v___x_461_; 
lean_inc(v___y_459_);
lean_inc_ref(v___y_458_);
lean_inc(v___y_457_);
lean_inc_ref(v___y_456_);
v___x_461_ = lean_whnf(v_e_455_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
if (lean_obj_tag(v___x_461_) == 0)
{
lean_object* v_a_462_; lean_object* v___x_463_; lean_object* v___x_464_; uint8_t v___x_465_; 
v_a_462_ = lean_ctor_get(v___x_461_, 0);
lean_inc(v_a_462_);
lean_dec_ref_known(v___x_461_, 1);
v___x_463_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___closed__1));
v___x_464_ = lean_unsigned_to_nat(1u);
v___x_465_ = l_Lean_Expr_isAppOfArity(v_a_462_, v___x_463_, v___x_464_);
if (v___x_465_ == 0)
{
lean_object* v___x_466_; 
v___x_466_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_462_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
return v___x_466_;
}
else
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_467_ = l_Lean_Expr_getAppNumArgs(v_a_462_);
v___x_468_ = lean_nat_sub(v___x_467_, v___x_464_);
lean_dec(v___x_467_);
v___x_469_ = l_Lean_Expr_getRevArg_x21(v_a_462_, v___x_468_);
lean_inc(v___y_459_);
lean_inc_ref(v___y_458_);
lean_inc(v___y_457_);
lean_inc_ref(v___y_456_);
v___x_470_ = lean_whnf(v___x_469_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
if (lean_obj_tag(v___x_470_) == 0)
{
lean_object* v_a_471_; lean_object* v___x_472_; lean_object* v___x_473_; uint8_t v___x_474_; 
v_a_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_a_471_);
lean_dec_ref_known(v___x_470_, 1);
v___x_472_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__2));
v___x_473_ = lean_unsigned_to_nat(3u);
v___x_474_ = l_Lean_Expr_isAppOfArity(v_a_471_, v___x_472_, v___x_473_);
if (v___x_474_ == 0)
{
lean_object* v___x_475_; 
lean_dec(v_a_471_);
v___x_475_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_462_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
return v___x_475_;
}
else
{
lean_object* v___f_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
lean_dec(v_a_462_);
v___f_476_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3));
v___x_477_ = l_Lean_Expr_getAppNumArgs(v_a_471_);
v___x_478_ = lean_nat_sub(v___x_477_, v___x_464_);
lean_dec(v___x_477_);
v___x_479_ = lean_nat_sub(v___x_478_, v___x_464_);
lean_dec(v___x_478_);
v___x_480_ = l_Lean_Expr_getRevArg_x21(v_a_471_, v___x_479_);
lean_dec(v_a_471_);
v___x_481_ = l_Lean_Meta_reduceEval___redArg(v___f_476_, v___x_480_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
if (lean_obj_tag(v___x_481_) == 0)
{
lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_491_; 
v_a_482_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_491_ == 0)
{
v___x_484_ = v___x_481_;
v_isShared_485_ = v_isSharedCheck_491_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_481_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_491_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
size_t v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
v___x_486_ = lean_usize_of_nat(v_a_482_);
lean_dec(v_a_482_);
v___x_487_ = lean_box_usize(v___x_486_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 0, v___x_487_);
v___x_489_ = v___x_484_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v___x_487_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
else
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
v_a_492_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_481_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_481_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_492_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
}
}
else
{
lean_object* v_a_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_507_; 
lean_dec(v_a_462_);
v_a_500_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_507_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_507_ == 0)
{
v___x_502_ = v___x_470_;
v_isShared_503_ = v_isSharedCheck_507_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_a_500_);
lean_dec(v___x_470_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_507_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
lean_object* v___x_505_; 
if (v_isShared_503_ == 0)
{
v___x_505_ = v___x_502_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v_a_500_);
v___x_505_ = v_reuseFailAlloc_506_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
return v___x_505_;
}
}
}
}
}
else
{
lean_object* v_a_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_515_; 
v_a_508_ = lean_ctor_get(v___x_461_, 0);
v_isSharedCheck_515_ = !lean_is_exclusive(v___x_461_);
if (v_isSharedCheck_515_ == 0)
{
v___x_510_ = v___x_461_;
v_isShared_511_ = v_isSharedCheck_515_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_a_508_);
lean_dec(v___x_461_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_515_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_513_; 
if (v_isShared_511_ == 0)
{
v___x_513_ = v___x_510_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v_a_508_);
v___x_513_ = v_reuseFailAlloc_514_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
return v___x_513_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0___boxed(lean_object* v_e_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_Qq_Lean_Meta_instReduceEvalUSize__qq___lam__0(v_e_516_, v___y_517_, v___y_518_, v___y_519_, v___y_520_);
lean_dec(v___y_520_);
lean_dec_ref(v___y_519_);
lean_dec(v___y_518_);
lean_dec_ref(v___y_517_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0(lean_object* v_e_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_){
_start:
{
lean_object* v___x_540_; 
lean_inc(v___y_538_);
lean_inc_ref(v___y_537_);
lean_inc(v___y_536_);
lean_inc_ref(v___y_535_);
v___x_540_ = lean_whnf(v_e_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_);
if (lean_obj_tag(v___x_540_) == 0)
{
lean_object* v_a_541_; lean_object* v___x_543_; uint8_t v_isShared_544_; uint8_t v_isSharedCheck_558_; 
v_a_541_ = lean_ctor_get(v___x_540_, 0);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_540_);
if (v_isSharedCheck_558_ == 0)
{
v___x_543_ = v___x_540_;
v_isShared_544_ = v_isSharedCheck_558_;
goto v_resetjp_542_;
}
else
{
lean_inc(v_a_541_);
lean_dec(v___x_540_);
v___x_543_ = lean_box(0);
v_isShared_544_ = v_isSharedCheck_558_;
goto v_resetjp_542_;
}
v_resetjp_542_:
{
lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_545_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__2));
v___x_546_ = l_Lean_Expr_isAppOf(v_a_541_, v___x_545_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; uint8_t v___x_548_; 
v___x_547_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___closed__4));
v___x_548_ = l_Lean_Expr_isAppOf(v_a_541_, v___x_547_);
if (v___x_548_ == 0)
{
lean_object* v___x_549_; 
lean_del_object(v___x_543_);
v___x_549_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_541_, v___y_535_, v___y_536_, v___y_537_, v___y_538_);
return v___x_549_;
}
else
{
lean_object* v___x_550_; lean_object* v___x_552_; 
lean_dec(v_a_541_);
v___x_550_ = lean_box(v___x_546_);
if (v_isShared_544_ == 0)
{
lean_ctor_set(v___x_543_, 0, v___x_550_);
v___x_552_ = v___x_543_;
goto v_reusejp_551_;
}
else
{
lean_object* v_reuseFailAlloc_553_; 
v_reuseFailAlloc_553_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_553_, 0, v___x_550_);
v___x_552_ = v_reuseFailAlloc_553_;
goto v_reusejp_551_;
}
v_reusejp_551_:
{
return v___x_552_;
}
}
}
else
{
lean_object* v___x_554_; lean_object* v___x_556_; 
lean_dec(v_a_541_);
v___x_554_ = lean_box(v___x_546_);
if (v_isShared_544_ == 0)
{
lean_ctor_set(v___x_543_, 0, v___x_554_);
v___x_556_ = v___x_543_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v___x_554_);
v___x_556_ = v_reuseFailAlloc_557_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
return v___x_556_;
}
}
}
}
else
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
v_a_559_ = lean_ctor_get(v___x_540_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_540_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v___x_540_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_540_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_562_ == 0)
{
v___x_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0___boxed(lean_object* v_e_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_){
_start:
{
lean_object* v_res_573_; 
v_res_573_ = lp_Qq_Lean_Meta_instReduceEvalBool__qq___lam__0(v_e_567_, v___y_568_, v___y_569_, v___y_570_, v___y_571_);
lean_dec(v___y_571_);
lean_dec_ref(v___y_570_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
return v_res_573_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0(lean_object* v_e_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_){
_start:
{
lean_object* v___x_588_; 
lean_inc(v___y_586_);
lean_inc_ref(v___y_585_);
lean_inc(v___y_584_);
lean_inc_ref(v___y_583_);
lean_inc_ref(v_e_582_);
v___x_588_ = lean_whnf(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
if (lean_obj_tag(v___x_588_) == 0)
{
lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_641_; 
v_a_589_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_641_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_641_ == 0)
{
v___x_591_ = v___x_588_;
v_isShared_592_ = v_isSharedCheck_641_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_588_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_641_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_593_; 
v___x_593_ = l_Lean_Expr_constName_x3f(v_a_589_);
lean_dec(v_a_589_);
if (lean_obj_tag(v___x_593_) == 1)
{
lean_object* v_val_594_; 
v_val_594_ = lean_ctor_get(v___x_593_, 0);
lean_inc(v_val_594_);
lean_dec_ref_known(v___x_593_, 1);
if (lean_obj_tag(v_val_594_) == 1)
{
lean_object* v_pre_595_; 
v_pre_595_ = lean_ctor_get(v_val_594_, 0);
lean_inc(v_pre_595_);
if (lean_obj_tag(v_pre_595_) == 1)
{
lean_object* v_pre_596_; 
v_pre_596_ = lean_ctor_get(v_pre_595_, 0);
lean_inc(v_pre_596_);
if (lean_obj_tag(v_pre_596_) == 1)
{
lean_object* v_pre_597_; 
v_pre_597_ = lean_ctor_get(v_pre_596_, 0);
if (lean_obj_tag(v_pre_597_) == 0)
{
lean_object* v_str_598_; lean_object* v_str_599_; lean_object* v_str_600_; lean_object* v___x_601_; uint8_t v___x_602_; 
v_str_598_ = lean_ctor_get(v_val_594_, 1);
lean_inc_ref(v_str_598_);
lean_dec_ref_known(v_val_594_, 2);
v_str_599_ = lean_ctor_get(v_pre_595_, 1);
lean_inc_ref(v_str_599_);
lean_dec_ref_known(v_pre_595_, 2);
v_str_600_ = lean_ctor_get(v_pre_596_, 1);
lean_inc_ref(v_str_600_);
lean_dec_ref_known(v_pre_596_, 2);
v___x_601_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__0));
v___x_602_ = lean_string_dec_eq(v_str_600_, v___x_601_);
lean_dec_ref(v_str_600_);
if (v___x_602_ == 0)
{
lean_object* v___x_603_; 
lean_dec_ref(v_str_599_);
lean_dec_ref(v_str_598_);
lean_del_object(v___x_591_);
v___x_603_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_603_;
}
else
{
lean_object* v___x_604_; uint8_t v___x_605_; 
v___x_604_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__1));
v___x_605_ = lean_string_dec_eq(v_str_599_, v___x_604_);
lean_dec_ref(v_str_599_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; 
lean_dec_ref(v_str_598_);
lean_del_object(v___x_591_);
v___x_606_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_606_;
}
else
{
lean_object* v___x_607_; uint8_t v___x_608_; 
v___x_607_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__2));
v___x_608_ = lean_string_dec_eq(v_str_598_, v___x_607_);
if (v___x_608_ == 0)
{
lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_609_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__3));
v___x_610_ = lean_string_dec_eq(v_str_598_, v___x_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; uint8_t v___x_612_; 
v___x_611_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__4));
v___x_612_ = lean_string_dec_eq(v_str_598_, v___x_611_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; uint8_t v___x_614_; 
v___x_613_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___closed__5));
v___x_614_ = lean_string_dec_eq(v_str_598_, v___x_613_);
lean_dec_ref(v_str_598_);
if (v___x_614_ == 0)
{
lean_object* v___x_615_; 
lean_del_object(v___x_591_);
v___x_615_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_615_;
}
else
{
uint8_t v___x_616_; lean_object* v___x_617_; lean_object* v___x_619_; 
lean_dec_ref(v_e_582_);
v___x_616_ = 3;
v___x_617_ = lean_box(v___x_616_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_617_);
v___x_619_ = v___x_591_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v___x_617_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
}
else
{
uint8_t v___x_621_; lean_object* v___x_622_; lean_object* v___x_624_; 
lean_dec_ref(v_str_598_);
lean_dec_ref(v_e_582_);
v___x_621_ = 2;
v___x_622_ = lean_box(v___x_621_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_622_);
v___x_624_ = v___x_591_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v___x_622_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
return v___x_624_;
}
}
}
else
{
uint8_t v___x_626_; lean_object* v___x_627_; lean_object* v___x_629_; 
lean_dec_ref(v_str_598_);
lean_dec_ref(v_e_582_);
v___x_626_ = 1;
v___x_627_ = lean_box(v___x_626_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_627_);
v___x_629_ = v___x_591_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v___x_627_);
v___x_629_ = v_reuseFailAlloc_630_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
return v___x_629_;
}
}
}
else
{
uint8_t v___x_631_; lean_object* v___x_632_; lean_object* v___x_634_; 
lean_dec_ref(v_str_598_);
lean_dec_ref(v_e_582_);
v___x_631_ = 0;
v___x_632_ = lean_box(v___x_631_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_632_);
v___x_634_ = v___x_591_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v___x_632_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
}
else
{
lean_object* v___x_636_; 
lean_dec_ref_known(v_pre_596_, 2);
lean_dec_ref_known(v_pre_595_, 2);
lean_dec_ref_known(v_val_594_, 2);
lean_del_object(v___x_591_);
v___x_636_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_636_;
}
}
else
{
lean_object* v___x_637_; 
lean_dec(v_pre_596_);
lean_dec_ref_known(v_pre_595_, 2);
lean_dec_ref_known(v_val_594_, 2);
lean_del_object(v___x_591_);
v___x_637_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_637_;
}
}
else
{
lean_object* v___x_638_; 
lean_dec_ref_known(v_val_594_, 2);
lean_dec(v_pre_595_);
lean_del_object(v___x_591_);
v___x_638_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_638_;
}
}
else
{
lean_object* v___x_639_; 
lean_dec(v_val_594_);
lean_del_object(v___x_591_);
v___x_639_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_639_;
}
}
else
{
lean_object* v___x_640_; 
lean_dec(v___x_593_);
lean_del_object(v___x_591_);
v___x_640_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_e_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
return v___x_640_;
}
}
}
else
{
lean_object* v_a_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_649_; 
lean_dec_ref(v_e_582_);
v_a_642_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_649_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_649_ == 0)
{
v___x_644_ = v___x_588_;
v_isShared_645_ = v_isSharedCheck_649_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_a_642_);
lean_dec(v___x_588_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_649_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v___x_647_; 
if (v_isShared_645_ == 0)
{
v___x_647_ = v___x_644_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v_a_642_);
v___x_647_ = v_reuseFailAlloc_648_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
return v___x_647_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0___boxed(lean_object* v_e_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_Qq_Lean_Meta_instReduceEvalBinderInfo__qq___lam__0(v_e_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_653_);
lean_dec(v___y_652_);
lean_dec_ref(v___y_651_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0(lean_object* v_e_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_){
_start:
{
lean_object* v___x_677_; 
lean_inc(v___y_675_);
lean_inc_ref(v___y_674_);
lean_inc(v___y_673_);
lean_inc_ref(v___y_672_);
v___x_677_ = lean_whnf(v_e_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v_a_678_; lean_object* v___x_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v_a_678_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_a_678_);
lean_dec_ref_known(v___x_677_, 1);
v___x_679_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__2));
v___x_680_ = lean_unsigned_to_nat(1u);
v___x_681_ = l_Lean_Expr_isAppOfArity(v_a_678_, v___x_679_, v___x_680_);
if (v___x_681_ == 0)
{
lean_object* v___x_682_; uint8_t v___x_683_; 
v___x_682_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__4));
v___x_683_ = l_Lean_Expr_isAppOfArity(v_a_678_, v___x_682_, v___x_680_);
if (v___x_683_ == 0)
{
lean_object* v___x_684_; 
v___x_684_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_678_, v___y_672_, v___y_673_, v___y_674_, v___y_675_);
return v___x_684_;
}
else
{
lean_object* v___f_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; 
v___f_685_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___closed__5));
v___x_686_ = l_Lean_Expr_getAppNumArgs(v_a_678_);
v___x_687_ = lean_nat_sub(v___x_686_, v___x_680_);
lean_dec(v___x_686_);
v___x_688_ = l_Lean_Expr_getRevArg_x21(v_a_678_, v___x_687_);
lean_dec(v_a_678_);
v___x_689_ = l_Lean_Meta_reduceEval___redArg(v___f_685_, v___x_688_, v___y_672_, v___y_673_, v___y_674_, v___y_675_);
if (lean_obj_tag(v___x_689_) == 0)
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_698_; 
v_a_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_698_ == 0)
{
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_698_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_689_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_698_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_694_; lean_object* v___x_696_; 
v___x_694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_694_, 0, v_a_690_);
if (v_isShared_693_ == 0)
{
lean_ctor_set(v___x_692_, 0, v___x_694_);
v___x_696_ = v___x_692_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_694_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
else
{
lean_object* v_a_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_706_; 
v_a_699_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_706_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_706_ == 0)
{
v___x_701_ = v___x_689_;
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_a_699_);
lean_dec(v___x_689_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_704_; 
if (v_isShared_702_ == 0)
{
v___x_704_ = v___x_701_;
goto v_reusejp_703_;
}
else
{
lean_object* v_reuseFailAlloc_705_; 
v_reuseFailAlloc_705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_705_, 0, v_a_699_);
v___x_704_ = v_reuseFailAlloc_705_;
goto v_reusejp_703_;
}
v_reusejp_703_:
{
return v___x_704_;
}
}
}
}
}
else
{
lean_object* v___f_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; 
v___f_707_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFinOfNeZeroNat__qq___redArg___lam__0___closed__3));
v___x_708_ = l_Lean_Expr_getAppNumArgs(v_a_678_);
v___x_709_ = lean_nat_sub(v___x_708_, v___x_680_);
lean_dec(v___x_708_);
v___x_710_ = l_Lean_Expr_getRevArg_x21(v_a_678_, v___x_709_);
lean_dec(v_a_678_);
v___x_711_ = l_Lean_Meta_reduceEval___redArg(v___f_707_, v___x_710_, v___y_672_, v___y_673_, v___y_674_, v___y_675_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_object* v_a_712_; lean_object* v___x_714_; uint8_t v_isShared_715_; uint8_t v_isSharedCheck_720_; 
v_a_712_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_720_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_720_ == 0)
{
v___x_714_ = v___x_711_;
v_isShared_715_ = v_isSharedCheck_720_;
goto v_resetjp_713_;
}
else
{
lean_inc(v_a_712_);
lean_dec(v___x_711_);
v___x_714_ = lean_box(0);
v_isShared_715_ = v_isSharedCheck_720_;
goto v_resetjp_713_;
}
v_resetjp_713_:
{
lean_object* v___x_716_; lean_object* v___x_718_; 
v___x_716_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_716_, 0, v_a_712_);
if (v_isShared_715_ == 0)
{
lean_ctor_set(v___x_714_, 0, v___x_716_);
v___x_718_ = v___x_714_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v___x_716_);
v___x_718_ = v_reuseFailAlloc_719_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
return v___x_718_;
}
}
}
else
{
lean_object* v_a_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_728_; 
v_a_721_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_728_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_728_ == 0)
{
v___x_723_ = v___x_711_;
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_a_721_);
lean_dec(v___x_711_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_726_; 
if (v_isShared_724_ == 0)
{
v___x_726_ = v___x_723_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_727_; 
v_reuseFailAlloc_727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_727_, 0, v_a_721_);
v___x_726_ = v_reuseFailAlloc_727_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
return v___x_726_;
}
}
}
}
}
else
{
lean_object* v_a_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_736_; 
v_a_729_ = lean_ctor_get(v___x_677_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_677_);
if (v_isSharedCheck_736_ == 0)
{
v___x_731_ = v___x_677_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_a_729_);
lean_dec(v___x_677_);
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
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0___boxed(lean_object* v_e_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_Qq_Lean_Meta_instReduceEvalLiteral__qq___lam__0(v_e_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0(lean_object* v_e_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
lean_object* v___x_758_; 
lean_inc(v___y_756_);
lean_inc_ref(v___y_755_);
lean_inc(v___y_754_);
lean_inc_ref(v___y_753_);
v___x_758_ = lean_whnf(v_e_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_760_; lean_object* v___x_761_; uint8_t v___x_762_; 
v_a_759_ = lean_ctor_get(v___x_758_, 0);
lean_inc(v_a_759_);
lean_dec_ref_known(v___x_758_, 1);
v___x_760_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__1));
v___x_761_ = lean_unsigned_to_nat(1u);
v___x_762_ = l_Lean_Expr_isAppOfArity(v_a_759_, v___x_760_, v___x_761_);
if (v___x_762_ == 0)
{
lean_object* v___x_763_; 
v___x_763_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_759_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
return v___x_763_;
}
else
{
lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_764_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2));
v___x_765_ = l_Lean_Expr_getAppNumArgs(v_a_759_);
v___x_766_ = lean_nat_sub(v___x_765_, v___x_761_);
lean_dec(v___x_765_);
v___x_767_ = l_Lean_Expr_getRevArg_x21(v_a_759_, v___x_766_);
lean_dec(v_a_759_);
v___x_768_ = l_Lean_Meta_reduceEval___redArg(v___x_764_, v___x_767_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
if (lean_obj_tag(v___x_768_) == 0)
{
lean_object* v_a_769_; lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_776_; 
v_a_769_ = lean_ctor_get(v___x_768_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_776_ == 0)
{
v___x_771_ = v___x_768_;
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
else
{
lean_inc(v_a_769_);
lean_dec(v___x_768_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_776_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v___x_774_; 
if (v_isShared_772_ == 0)
{
v___x_774_ = v___x_771_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_a_769_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
else
{
lean_object* v_a_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_784_; 
v_a_777_ = lean_ctor_get(v___x_768_, 0);
v_isSharedCheck_784_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_784_ == 0)
{
v___x_779_ = v___x_768_;
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_a_777_);
lean_dec(v___x_768_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_782_; 
if (v_isShared_780_ == 0)
{
v___x_782_ = v___x_779_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v_a_777_);
v___x_782_ = v_reuseFailAlloc_783_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
return v___x_782_;
}
}
}
}
}
else
{
lean_object* v_a_785_; lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_792_; 
v_a_785_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_792_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_792_ == 0)
{
v___x_787_ = v___x_758_;
v_isShared_788_ = v_isSharedCheck_792_;
goto v_resetjp_786_;
}
else
{
lean_inc(v_a_785_);
lean_dec(v___x_758_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_792_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
lean_object* v___x_790_; 
if (v_isShared_788_ == 0)
{
v___x_790_ = v___x_787_;
goto v_reusejp_789_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v_a_785_);
v___x_790_ = v_reuseFailAlloc_791_;
goto v_reusejp_789_;
}
v_reusejp_789_:
{
return v___x_790_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___boxed(lean_object* v_e_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0(v_e_793_, v___y_794_, v___y_795_, v___y_796_, v___y_797_);
lean_dec(v___y_797_);
lean_dec_ref(v___y_796_);
lean_dec(v___y_795_);
lean_dec_ref(v___y_794_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0(lean_object* v_e_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_){
_start:
{
lean_object* v___x_813_; 
lean_inc(v___y_811_);
lean_inc_ref(v___y_810_);
lean_inc(v___y_809_);
lean_inc_ref(v___y_808_);
v___x_813_ = lean_whnf(v_e_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_);
if (lean_obj_tag(v___x_813_) == 0)
{
lean_object* v_a_814_; lean_object* v___x_815_; lean_object* v___x_816_; uint8_t v___x_817_; 
v_a_814_ = lean_ctor_get(v___x_813_, 0);
lean_inc(v_a_814_);
lean_dec_ref_known(v___x_813_, 1);
v___x_815_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___closed__1));
v___x_816_ = lean_unsigned_to_nat(1u);
v___x_817_ = l_Lean_Expr_isAppOfArity(v_a_814_, v___x_815_, v___x_816_);
if (v___x_817_ == 0)
{
lean_object* v___x_818_; 
v___x_818_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_814_, v___y_808_, v___y_809_, v___y_810_, v___y_811_);
return v___x_818_;
}
else
{
lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_819_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2));
v___x_820_ = l_Lean_Expr_getAppNumArgs(v_a_814_);
v___x_821_ = lean_nat_sub(v___x_820_, v___x_816_);
lean_dec(v___x_820_);
v___x_822_ = l_Lean_Expr_getRevArg_x21(v_a_814_, v___x_821_);
lean_dec(v_a_814_);
v___x_823_ = l_Lean_Meta_reduceEval___redArg(v___x_819_, v___x_822_, v___y_808_, v___y_809_, v___y_810_, v___y_811_);
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v_a_824_; lean_object* v___x_826_; uint8_t v_isShared_827_; uint8_t v_isSharedCheck_831_; 
v_a_824_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_831_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_831_ == 0)
{
v___x_826_ = v___x_823_;
v_isShared_827_ = v_isSharedCheck_831_;
goto v_resetjp_825_;
}
else
{
lean_inc(v_a_824_);
lean_dec(v___x_823_);
v___x_826_ = lean_box(0);
v_isShared_827_ = v_isSharedCheck_831_;
goto v_resetjp_825_;
}
v_resetjp_825_:
{
lean_object* v___x_829_; 
if (v_isShared_827_ == 0)
{
v___x_829_ = v___x_826_;
goto v_reusejp_828_;
}
else
{
lean_object* v_reuseFailAlloc_830_; 
v_reuseFailAlloc_830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_830_, 0, v_a_824_);
v___x_829_ = v_reuseFailAlloc_830_;
goto v_reusejp_828_;
}
v_reusejp_828_:
{
return v___x_829_;
}
}
}
else
{
lean_object* v_a_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_839_; 
v_a_832_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_839_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_839_ == 0)
{
v___x_834_ = v___x_823_;
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_a_832_);
lean_dec(v___x_823_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v___x_837_; 
if (v_isShared_835_ == 0)
{
v___x_837_ = v___x_834_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v_a_832_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
}
}
else
{
lean_object* v_a_840_; lean_object* v___x_842_; uint8_t v_isShared_843_; uint8_t v_isSharedCheck_847_; 
v_a_840_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_847_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_847_ == 0)
{
v___x_842_ = v___x_813_;
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
else
{
lean_inc(v_a_840_);
lean_dec(v___x_813_);
v___x_842_ = lean_box(0);
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
v_resetjp_841_:
{
lean_object* v___x_845_; 
if (v_isShared_843_ == 0)
{
v___x_845_ = v___x_842_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_846_; 
v_reuseFailAlloc_846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_846_, 0, v_a_840_);
v___x_845_ = v_reuseFailAlloc_846_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
return v___x_845_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0___boxed(lean_object* v_e_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_res_854_; 
v_res_854_ = lp_Qq_Lean_Meta_instReduceEvalLevelMVarId__qq___lam__0(v_e_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
return v_res_854_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0(lean_object* v_e_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v___x_868_; 
lean_inc(v___y_866_);
lean_inc_ref(v___y_865_);
lean_inc(v___y_864_);
lean_inc_ref(v___y_863_);
v___x_868_ = lean_whnf(v_e_862_, v___y_863_, v___y_864_, v___y_865_, v___y_866_);
if (lean_obj_tag(v___x_868_) == 0)
{
lean_object* v_a_869_; lean_object* v___x_870_; lean_object* v___x_871_; uint8_t v___x_872_; 
v_a_869_ = lean_ctor_get(v___x_868_, 0);
lean_inc(v_a_869_);
lean_dec_ref_known(v___x_868_, 1);
v___x_870_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___closed__1));
v___x_871_ = lean_unsigned_to_nat(1u);
v___x_872_ = l_Lean_Expr_isAppOfArity(v_a_869_, v___x_870_, v___x_871_);
if (v___x_872_ == 0)
{
lean_object* v___x_873_; 
v___x_873_ = lp_Qq_Lean_Meta_throwFailedToEval___redArg(v_a_869_, v___y_863_, v___y_864_, v___y_865_, v___y_866_);
return v___x_873_;
}
else
{
lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; 
v___x_874_ = ((lean_object*)(lp_Qq_Lean_Meta_instReduceEvalMVarId__qq___lam__0___closed__2));
v___x_875_ = l_Lean_Expr_getAppNumArgs(v_a_869_);
v___x_876_ = lean_nat_sub(v___x_875_, v___x_871_);
lean_dec(v___x_875_);
v___x_877_ = l_Lean_Expr_getRevArg_x21(v_a_869_, v___x_876_);
lean_dec(v_a_869_);
v___x_878_ = l_Lean_Meta_reduceEval___redArg(v___x_874_, v___x_877_, v___y_863_, v___y_864_, v___y_865_, v___y_866_);
if (lean_obj_tag(v___x_878_) == 0)
{
lean_object* v_a_879_; lean_object* v___x_881_; uint8_t v_isShared_882_; uint8_t v_isSharedCheck_886_; 
v_a_879_ = lean_ctor_get(v___x_878_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_878_);
if (v_isSharedCheck_886_ == 0)
{
v___x_881_ = v___x_878_;
v_isShared_882_ = v_isSharedCheck_886_;
goto v_resetjp_880_;
}
else
{
lean_inc(v_a_879_);
lean_dec(v___x_878_);
v___x_881_ = lean_box(0);
v_isShared_882_ = v_isSharedCheck_886_;
goto v_resetjp_880_;
}
v_resetjp_880_:
{
lean_object* v___x_884_; 
if (v_isShared_882_ == 0)
{
v___x_884_ = v___x_881_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v_a_879_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
}
else
{
lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_894_; 
v_a_887_ = lean_ctor_get(v___x_878_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_878_);
if (v_isSharedCheck_894_ == 0)
{
v___x_889_ = v___x_878_;
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_878_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_a_887_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
}
else
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
v_a_895_ = lean_ctor_get(v___x_868_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_868_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_868_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_868_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0___boxed(lean_object* v_e_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_){
_start:
{
lean_object* v_res_909_; 
v_res_909_ = lp_Qq_Lean_Meta_instReduceEvalFVarId__qq___lam__0(v_e_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_);
lean_dec(v___y_907_);
lean_dec_ref(v___y_906_);
lean_dec(v___y_905_);
lean_dec_ref(v___y_904_);
return v_res_909_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_ReduceEval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_ForLean_ReduceEval(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_ReduceEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_ForLean_ReduceEval(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_ReduceEval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_ForLean_ReduceEval(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_ReduceEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_ForLean_ReduceEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_ForLean_ReduceEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_ForLean_ReduceEval(builtin);
}
#ifdef __cplusplus
}
#endif
