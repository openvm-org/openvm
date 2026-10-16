// Lean compiler output
// Module: Plausible.Gen
// Imports: public import Init public meta import Init public import Plausible.Random
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
lean_object* lean_string_append(lean_object*, lean_object*);
extern lean_object* l_IO_stdGenRef;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_IO_println___redArg(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
extern lean_object* l_stdRange;
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_stdNext(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_instMonadExceptOfExcept(lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
extern lean_object* l_instRandomGenStdGen;
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_insertIdxTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_pure(lean_object*, lean_object*, lean_object*);
lean_object* l_Except_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Except_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_instToStringString___lam__0___boxed(lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_forIn_x27_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_StateT_instMonadExceptOf___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonadExceptOf___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadExceptOfMonadExceptOf___redArg(lean_object*);
static const lean_string_object lp_plausible_Plausible_instInhabitedGenError_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_plausible_Plausible_instInhabitedGenError_default___closed__0 = (const lean_object*)&lp_plausible_Plausible_instInhabitedGenError_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instInhabitedGenError_default = (const lean_object*)&lp_plausible_Plausible_instInhabitedGenError_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instInhabitedGenError = (const lean_object*)&lp_plausible_Plausible_instInhabitedGenError_default___closed__0_value;
static const lean_string_object lp_plausible_Plausible_instReprGenError_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Plausible.GenError.genError"};
static const lean_object* lp_plausible_Plausible_instReprGenError_repr___closed__0 = (const lean_object*)&lp_plausible_Plausible_instReprGenError_repr___closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_instReprGenError_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instReprGenError_repr___closed__0_value)}};
static const lean_object* lp_plausible_Plausible_instReprGenError_repr___closed__1 = (const lean_object*)&lp_plausible_Plausible_instReprGenError_repr___closed__1_value;
static const lean_ctor_object lp_plausible_Plausible_instReprGenError_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instReprGenError_repr___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_instReprGenError_repr___closed__2 = (const lean_object*)&lp_plausible_Plausible_instReprGenError_repr___closed__2_value;
static lean_once_cell_t lp_plausible_Plausible_instReprGenError_repr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instReprGenError_repr___closed__3;
static lean_once_cell_t lp_plausible_Plausible_instReprGenError_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instReprGenError_repr___closed__4;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instReprGenError_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instReprGenError_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_instReprGenError___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instReprGenError_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instReprGenError___closed__0 = (const lean_object*)&lp_plausible_Plausible_instReprGenError___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instReprGenError = (const lean_object*)&lp_plausible_Plausible_instReprGenError___closed__0_value;
LEAN_EXPORT uint8_t lp_plausible_Plausible_instBEqGenError_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instBEqGenError_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_instBEqGenError___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instBEqGenError_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instBEqGenError___closed__0 = (const lean_object*)&lp_plausible_Plausible_instBEqGenError___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instBEqGenError = (const lean_object*)&lp_plausible_Plausible_instBEqGenError___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Gen_genericFailure___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Generation failure."};
static const lean_object* lp_plausible_Plausible_Gen_genericFailure___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_genericFailure___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Gen_genericFailure = (const lean_object*)&lp_plausible_Plausible_Gen_genericFailure___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instPartialOrderExceptGenError(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instCCPOExceptGenError(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen(lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__0 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__1 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__1_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__2 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__2_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__3 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__3_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__4 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_instMonadErrorGen___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__4_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__0_value)}};
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__5 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__5_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__6 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_instMonadErrorGen___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__5_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__6_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__1_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__2_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__3_value)}};
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__7 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__7_value;
static const lean_closure_object lp_plausible_Plausible_instMonadErrorGen___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__8 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_instMonadErrorGen___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__7_value),((lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__8_value)}};
static const lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__9 = (const lean_object*)&lp_plausible_Plausible_instMonadErrorGen___closed__9_value;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__10;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__11;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__12;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__13;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__14;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__15;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__16;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__17;
static lean_once_cell_t lp_plausible_Plausible_instMonadErrorGen___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instMonadErrorGen___closed__18;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadErrorGen;
static const lean_string_object lp_plausible_Plausible_Gen_genFailure___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "generation failure: "};
static const lean_object* lp_plausible_Plausible_Gen_genFailure___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_genFailure___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_genFailure(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_genFailure___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___closed__0 = (const lean_object*)&lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_getSize(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_getSize___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNat___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Gen_outOfFuel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "out of fuel"};
static const lean_object* lp_plausible_Plausible_Gen_outOfFuel___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_outOfFuel___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Gen_outOfFuel = (const lean_object*)&lp_plausible_Plausible_Gen_outOfFuel___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Gen_inhabitedGen___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "inhabitedWitness"};
static const lean_object* lp_plausible_Plausible_Gen_inhabitedGen___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_inhabitedGen___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_Gen_inhabitedGen___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Gen_inhabitedGen___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_inhabitedGen___closed__0_value)} };
static const lean_object* lp_plausible_Plausible_Gen_inhabitedGen___closed__1 = (const lean_object*)&lp_plausible_Plausible_Gen_inhabitedGen___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen(lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "Plausible.Chamelean.Gen.pickDrop: out of options."};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__0_value;
static const lean_closure_object lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__0_value)} };
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_outOfFuel___closed__0_value)}};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__3 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__3_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_0),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_1),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value_aux_2),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4_value;
static const lean_array_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__6 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_0),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_1),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value_aux_2),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__8 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__9 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__9_value;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__10 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__10_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_0),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_1),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value_aux_2),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(53, 158, 1, 232, 101, 200, 191, 197)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11_value;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13;
static const lean_string_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__14 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__14_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_0),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_1),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value_aux_2),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15_value;
static const lean_ctor_object lp_plausible_Plausible_Gen_oneOf___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__9_value),((lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5_value)}};
static const lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__16 = (const lean_object*)&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__16_value;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25;
static lean_once_cell_t lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___auto__1;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Plausible_Gen_permutationOf___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_Gen_permutationOf___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Gen_permutationOf___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Generation failure:"};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_instMonadLiftStateIOGen___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___closed__0 = (const lean_object*)&lp_plausible_Plausible_instMonadLiftStateIOGen___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen = (const lean_object*)&lp_plausible_Plausible_instMonadLiftStateIOGen___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Plausible_Gen_printSamples___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___closed__0;
static const lean_closure_object lp_plausible_Plausible_Gen_printSamples___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instToStringString___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_Gen_printSamples___redArg___closed__1_value;
static lean_once_cell_t lp_plausible_Plausible_Gen_printSamples___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_decr(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Gen.runUntil: Out of attempts"};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__0_value)}};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "uh oh"};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__0_value)}};
static const lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_plausible_Plausible_instReprGenError_repr___closed__3(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_unsigned_to_nat(2u);
v___x_11_ = lean_nat_to_int(v___x_10_);
return v___x_11_;
}
}
static lean_object* _init_lp_plausible_Plausible_instReprGenError_repr___closed__4(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_unsigned_to_nat(1u);
v___x_13_ = lean_nat_to_int(v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instReprGenError_repr(lean_object* v_x_14_, lean_object* v_prec_15_){
_start:
{
lean_object* v___y_17_; lean_object* v___x_26_; uint8_t v___x_27_; 
v___x_26_ = lean_unsigned_to_nat(1024u);
v___x_27_ = lean_nat_dec_le(v___x_26_, v_prec_15_);
if (v___x_27_ == 0)
{
lean_object* v___x_28_; 
v___x_28_ = lean_obj_once(&lp_plausible_Plausible_instReprGenError_repr___closed__3, &lp_plausible_Plausible_instReprGenError_repr___closed__3_once, _init_lp_plausible_Plausible_instReprGenError_repr___closed__3);
v___y_17_ = v___x_28_;
goto v___jp_16_;
}
else
{
lean_object* v___x_29_; 
v___x_29_ = lean_obj_once(&lp_plausible_Plausible_instReprGenError_repr___closed__4, &lp_plausible_Plausible_instReprGenError_repr___closed__4_once, _init_lp_plausible_Plausible_instReprGenError_repr___closed__4);
v___y_17_ = v___x_29_;
goto v___jp_16_;
}
v___jp_16_:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; uint8_t v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_18_ = ((lean_object*)(lp_plausible_Plausible_instReprGenError_repr___closed__2));
v___x_19_ = l_String_quote(v_x_14_);
v___x_20_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
v___x_21_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_21_, 0, v___x_18_);
lean_ctor_set(v___x_21_, 1, v___x_20_);
lean_inc(v___y_17_);
v___x_22_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_22_, 0, v___y_17_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
v___x_23_ = 0;
v___x_24_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_24_, 0, v___x_22_);
lean_ctor_set_uint8(v___x_24_, sizeof(void*)*1, v___x_23_);
v___x_25_ = l_Repr_addAppParen(v___x_24_, v_prec_15_);
return v___x_25_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instReprGenError_repr___boxed(lean_object* v_x_30_, lean_object* v_prec_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_plausible_Plausible_instReprGenError_repr(v_x_30_, v_prec_31_);
lean_dec(v_prec_31_);
return v_res_32_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Plausible_instBEqGenError_beq(lean_object* v_x_35_, lean_object* v_x_36_){
_start:
{
uint8_t v___x_37_; 
v___x_37_ = lean_string_dec_eq(v_x_35_, v_x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instBEqGenError_beq___boxed(lean_object* v_x_38_, lean_object* v_x_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_plausible_Plausible_instBEqGenError_beq(v_x_38_, v_x_39_);
lean_dec_ref(v_x_39_);
lean_dec_ref(v_x_38_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instPartialOrderExceptGenError(lean_object* v_00_u03b1_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_box(0);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instCCPOExceptGenError(lean_object* v_00_u03b1_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0(lean_object* v_inst_50_, lean_object* v_00_u03b1_51_, lean_object* v_m_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_apply_1(v_m_52_, v___y_53_);
lean_inc(v___y_54_);
v___x_56_ = lean_apply_3(v_inst_50_, lean_box(0), v___x_55_, v___y_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0___boxed(lean_object* v_inst_57_, lean_object* v_00_u03b1_58_, lean_object* v_m_59_, lean_object* v___y_60_, lean_object* v___y_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0(v_inst_57_, v_00_u03b1_58_, v_m_59_, v___y_60_, v___y_61_);
lean_dec(v___y_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen___redArg(lean_object* v_inst_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_64_, 0, v_inst_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftGen(lean_object* v_m_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; 
v___f_67_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instMonadLiftGen___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_67_, 0, v_inst_66_);
return v___f_67_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__10(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = ((lean_object*)(lp_plausible_Plausible_instMonadErrorGen___closed__9));
v___x_88_ = l_ReaderT_instMonad___redArg(v___x_87_);
return v___x_88_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__11(void){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = l_instMonadExceptOfExcept(lean_box(0));
return v___x_89_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__12(void){
_start:
{
lean_object* v___x_90_; lean_object* v___f_91_; 
v___x_90_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__11, &lp_plausible_Plausible_instMonadErrorGen___closed__11_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__11);
v___f_91_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_91_, 0, v___x_90_);
return v___f_91_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__13(void){
_start:
{
lean_object* v___x_92_; lean_object* v___f_93_; 
v___x_92_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__11, &lp_plausible_Plausible_instMonadErrorGen___closed__11_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__11);
v___f_93_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_93_, 0, v___x_92_);
return v___f_93_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__14(void){
_start:
{
lean_object* v___f_94_; lean_object* v___f_95_; lean_object* v___x_96_; 
v___f_94_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__13, &lp_plausible_Plausible_instMonadErrorGen___closed__13_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__13);
v___f_95_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__12, &lp_plausible_Plausible_instMonadErrorGen___closed__12_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__12);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___f_95_);
lean_ctor_set(v___x_96_, 1, v___f_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__15(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___f_99_; 
v___x_97_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__10, &lp_plausible_Plausible_instMonadErrorGen___closed__10_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__10);
v___x_98_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__14, &lp_plausible_Plausible_instMonadErrorGen___closed__14_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__14);
v___f_99_ = lean_alloc_closure((void*)(l_StateT_instMonadExceptOf___redArg___lam__1), 5, 2);
lean_closure_set(v___f_99_, 0, v___x_98_);
lean_closure_set(v___f_99_, 1, v___x_97_);
return v___f_99_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__16(void){
_start:
{
lean_object* v___x_100_; lean_object* v___f_101_; 
v___x_100_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__14, &lp_plausible_Plausible_instMonadErrorGen___closed__14_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__14);
v___f_101_ = lean_alloc_closure((void*)(l_StateT_instMonadExceptOf___redArg___lam__3), 5, 1);
lean_closure_set(v___f_101_, 0, v___x_100_);
return v___f_101_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__17(void){
_start:
{
lean_object* v___f_102_; lean_object* v___f_103_; lean_object* v___x_104_; 
v___f_102_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__16, &lp_plausible_Plausible_instMonadErrorGen___closed__16_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__16);
v___f_103_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__15, &lp_plausible_Plausible_instMonadErrorGen___closed__15_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__15);
v___x_104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_104_, 0, v___f_103_);
lean_ctor_set(v___x_104_, 1, v___f_102_);
return v___x_104_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen___closed__18(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__17, &lp_plausible_Plausible_instMonadErrorGen___closed__17_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__17);
v___x_106_ = l_instMonadExceptOfMonadExceptOf___redArg(v___x_105_);
return v___x_106_;
}
}
static lean_object* _init_lp_plausible_Plausible_instMonadErrorGen(void){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_obj_once(&lp_plausible_Plausible_instMonadErrorGen___closed__18, &lp_plausible_Plausible_instMonadErrorGen___closed__18_once, _init_lp_plausible_Plausible_instMonadErrorGen___closed__18);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_genFailure(lean_object* v_e_109_){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = ((lean_object*)(lp_plausible_Plausible_Gen_genFailure___closed__0));
v___x_111_ = lean_string_append(v___x_110_, v_e_109_);
v___x_112_ = lean_mk_io_user_error(v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_genFailure___boxed(lean_object* v_e_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_plausible_Plausible_Gen_genFailure(v_e_113_);
lean_dec_ref(v_e_113_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___redArg(lean_object* v_x_115_, lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
lean_object* v___x_118_; 
lean_inc(v_a_117_);
v___x_118_ = lean_apply_2(v_x_115_, v_a_116_, v_a_117_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_126_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_126_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_126_ == 0)
{
v___x_121_ = v___x_118_;
v_isShared_122_ = v_isSharedCheck_126_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_a_119_);
lean_dec(v___x_118_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_126_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
lean_object* v___x_124_; 
if (v_isShared_122_ == 0)
{
v___x_124_ = v___x_121_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v_a_119_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
else
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_143_; 
v_a_127_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_143_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_143_ == 0)
{
v___x_129_ = v___x_118_;
v_isShared_130_ = v_isSharedCheck_143_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_118_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_143_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v_fst_131_; lean_object* v_snd_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_142_; 
v_fst_131_ = lean_ctor_get(v_a_127_, 0);
v_snd_132_ = lean_ctor_get(v_a_127_, 1);
v_isSharedCheck_142_ = !lean_is_exclusive(v_a_127_);
if (v_isSharedCheck_142_ == 0)
{
v___x_134_ = v_a_127_;
v_isShared_135_ = v_isSharedCheck_142_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_snd_132_);
lean_inc(v_fst_131_);
lean_dec(v_a_127_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_142_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v_fst_131_);
lean_ctor_set(v_reuseFailAlloc_141_, 1, v_snd_132_);
v___x_137_ = v_reuseFailAlloc_141_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
lean_object* v___x_139_; 
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v___x_137_);
v___x_139_ = v___x_129_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v___x_137_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___redArg___boxed(lean_object* v_x_144_, lean_object* v_a_145_, lean_object* v_a_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_plausible_Plausible_Gen_up___redArg(v_x_144_, v_a_145_, v_a_146_);
lean_dec(v_a_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up(lean_object* v_00_u03b1_148_, lean_object* v_x_149_, lean_object* v_a_150_, lean_object* v_a_151_){
_start:
{
lean_object* v___x_152_; 
lean_inc(v_a_151_);
v___x_152_ = lean_apply_2(v_x_149_, v_a_150_, v_a_151_);
if (lean_obj_tag(v___x_152_) == 0)
{
lean_object* v_a_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
v_a_153_ = lean_ctor_get(v___x_152_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_152_);
if (v_isSharedCheck_160_ == 0)
{
v___x_155_ = v___x_152_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_a_153_);
lean_dec(v___x_152_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_158_; 
if (v_isShared_156_ == 0)
{
v___x_158_ = v___x_155_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v_a_153_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
else
{
lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_177_; 
v_a_161_ = lean_ctor_get(v___x_152_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_152_);
if (v_isSharedCheck_177_ == 0)
{
v___x_163_ = v___x_152_;
v_isShared_164_ = v_isSharedCheck_177_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_152_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_177_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v_fst_165_; lean_object* v_snd_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_176_; 
v_fst_165_ = lean_ctor_get(v_a_161_, 0);
v_snd_166_ = lean_ctor_get(v_a_161_, 1);
v_isSharedCheck_176_ = !lean_is_exclusive(v_a_161_);
if (v_isSharedCheck_176_ == 0)
{
v___x_168_ = v_a_161_;
v_isShared_169_ = v_isSharedCheck_176_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_snd_166_);
lean_inc(v_fst_165_);
lean_dec(v_a_161_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_176_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_171_; 
if (v_isShared_169_ == 0)
{
v___x_171_ = v___x_168_;
goto v_reusejp_170_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_fst_165_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v_snd_166_);
v___x_171_ = v_reuseFailAlloc_175_;
goto v_reusejp_170_;
}
v_reusejp_170_:
{
lean_object* v___x_173_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v___x_171_);
v___x_173_ = v___x_163_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_171_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_up___boxed(lean_object* v_00_u03b1_178_, lean_object* v_x_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_plausible_Plausible_Gen_up(v_00_u03b1_178_, v_x_179_, v_a_180_, v_a_181_);
lean_dec(v_a_181_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___redArg(lean_object* v_x_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_186_; 
lean_inc(v_a_185_);
v___x_186_ = lean_apply_2(v_x_183_, v_a_184_, v_a_185_);
if (lean_obj_tag(v___x_186_) == 0)
{
lean_object* v_a_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_194_; 
v_a_187_ = lean_ctor_get(v___x_186_, 0);
v_isSharedCheck_194_ = !lean_is_exclusive(v___x_186_);
if (v_isSharedCheck_194_ == 0)
{
v___x_189_ = v___x_186_;
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_a_187_);
lean_dec(v___x_186_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_192_; 
if (v_isShared_190_ == 0)
{
v___x_192_ = v___x_189_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_a_187_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
else
{
lean_object* v_a_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_211_; 
v_a_195_ = lean_ctor_get(v___x_186_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_186_);
if (v_isSharedCheck_211_ == 0)
{
v___x_197_ = v___x_186_;
v_isShared_198_ = v_isSharedCheck_211_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_a_195_);
lean_dec(v___x_186_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_211_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v_fst_199_; lean_object* v_snd_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_210_; 
v_fst_199_ = lean_ctor_get(v_a_195_, 0);
v_snd_200_ = lean_ctor_get(v_a_195_, 1);
v_isSharedCheck_210_ = !lean_is_exclusive(v_a_195_);
if (v_isSharedCheck_210_ == 0)
{
v___x_202_ = v_a_195_;
v_isShared_203_ = v_isSharedCheck_210_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_snd_200_);
lean_inc(v_fst_199_);
lean_dec(v_a_195_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_210_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v_fst_199_);
lean_ctor_set(v_reuseFailAlloc_209_, 1, v_snd_200_);
v___x_205_ = v_reuseFailAlloc_209_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
lean_object* v___x_207_; 
if (v_isShared_198_ == 0)
{
lean_ctor_set(v___x_197_, 0, v___x_205_);
v___x_207_ = v___x_197_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_205_);
v___x_207_ = v_reuseFailAlloc_208_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
return v___x_207_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___redArg___boxed(lean_object* v_x_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_plausible_Plausible_Gen_down___redArg(v_x_212_, v_a_213_, v_a_214_);
lean_dec(v_a_214_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down(lean_object* v_00_u03b1_216_, lean_object* v_x_217_, lean_object* v_a_218_, lean_object* v_a_219_){
_start:
{
lean_object* v___x_220_; 
lean_inc(v_a_219_);
v___x_220_ = lean_apply_2(v_x_217_, v_a_218_, v_a_219_);
if (lean_obj_tag(v___x_220_) == 0)
{
lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
v_a_221_ = lean_ctor_get(v___x_220_, 0);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_220_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_220_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
else
{
lean_object* v_a_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_245_; 
v_a_229_ = lean_ctor_get(v___x_220_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_245_ == 0)
{
v___x_231_ = v___x_220_;
v_isShared_232_ = v_isSharedCheck_245_;
goto v_resetjp_230_;
}
else
{
lean_inc(v_a_229_);
lean_dec(v___x_220_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_245_;
goto v_resetjp_230_;
}
v_resetjp_230_:
{
lean_object* v_fst_233_; lean_object* v_snd_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_244_; 
v_fst_233_ = lean_ctor_get(v_a_229_, 0);
v_snd_234_ = lean_ctor_get(v_a_229_, 1);
v_isSharedCheck_244_ = !lean_is_exclusive(v_a_229_);
if (v_isSharedCheck_244_ == 0)
{
v___x_236_ = v_a_229_;
v_isShared_237_ = v_isSharedCheck_244_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_snd_234_);
lean_inc(v_fst_233_);
lean_dec(v_a_229_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_244_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_239_; 
if (v_isShared_237_ == 0)
{
v___x_239_ = v___x_236_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_fst_233_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v_snd_234_);
v___x_239_ = v_reuseFailAlloc_243_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
lean_object* v___x_241_; 
if (v_isShared_232_ == 0)
{
lean_ctor_set(v___x_231_, 0, v___x_239_);
v___x_241_ = v___x_231_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_down___boxed(lean_object* v_00_u03b1_246_, lean_object* v_x_247_, lean_object* v_a_248_, lean_object* v_a_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_plausible_Plausible_Gen_down(v_00_u03b1_246_, v_x_247_, v_a_248_, v_a_249_);
lean_dec(v_a_249_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny___redArg(lean_object* v_inst_251_, lean_object* v_a_252_){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_253_ = l_instRandomGenStdGen;
v___x_254_ = lean_apply_3(v_inst_251_, lean_box(0), v___x_253_, v_a_252_);
v___x_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny(lean_object* v_00_u03b1_256_, lean_object* v_inst_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_plausible_Plausible_Gen_chooseAny___redArg(v_inst_257_, v_a_258_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseAny___boxed(lean_object* v_00_u03b1_261_, lean_object* v_inst_262_, lean_object* v_a_263_, lean_object* v_a_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_plausible_Plausible_Gen_chooseAny(v_00_u03b1_261_, v_inst_262_, v_a_263_, v_a_264_);
lean_dec(v_a_264_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___redArg(lean_object* v_inst_266_, lean_object* v_lo_267_, lean_object* v_hi_268_, lean_object* v_a_269_){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_270_ = l_instRandomGenStdGen;
v___x_271_ = lean_apply_6(v_inst_266_, lean_box(0), v_lo_267_, v_hi_268_, lean_box(0), v___x_270_, v_a_269_);
v___x_272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose(lean_object* v_00_u03b1_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_lo_276_, lean_object* v_hi_277_, lean_object* v_h_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_plausible_Plausible_Gen_choose___redArg(v_inst_275_, v_lo_276_, v_hi_277_, v_a_279_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___boxed(lean_object* v_00_u03b1_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_lo_285_, lean_object* v_hi_286_, lean_object* v_h_287_, lean_object* v_a_288_, lean_object* v_a_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_plausible_Plausible_Gen_choose(v_00_u03b1_282_, v_inst_283_, v_inst_284_, v_lo_285_, v_hi_286_, v_h_287_, v_a_288_, v_a_289_);
lean_dec(v_a_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1(lean_object* v___x_291_, lean_object* v___y_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lean_nat_mod(v___y_292_, v___x_291_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1___boxed(lean_object* v___x_294_, lean_object* v___y_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1(v___x_294_, v___y_295_);
lean_dec(v___y_295_);
lean_dec(v___x_294_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(lean_object* v_genLo_297_, lean_object* v_genMag_298_, lean_object* v_x_299_, lean_object* v_x_300_){
_start:
{
lean_object* v_zero_301_; uint8_t v_isZero_302_; 
v_zero_301_ = lean_unsigned_to_nat(0u);
v_isZero_302_ = lean_nat_dec_eq(v_x_299_, v_zero_301_);
if (v_isZero_302_ == 1)
{
lean_dec(v_x_299_);
return v_x_300_;
}
else
{
lean_object* v_fst_303_; lean_object* v_snd_304_; lean_object* v___x_305_; lean_object* v_fst_306_; lean_object* v_snd_307_; lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_321_; 
v_fst_303_ = lean_ctor_get(v_x_300_, 0);
lean_inc(v_fst_303_);
v_snd_304_ = lean_ctor_get(v_x_300_, 1);
lean_inc(v_snd_304_);
lean_dec_ref(v_x_300_);
v___x_305_ = l_stdNext(v_snd_304_);
v_fst_306_ = lean_ctor_get(v___x_305_, 0);
v_snd_307_ = lean_ctor_get(v___x_305_, 1);
v_isSharedCheck_321_ = !lean_is_exclusive(v___x_305_);
if (v_isSharedCheck_321_ == 0)
{
v___x_309_ = v___x_305_;
v_isShared_310_ = v_isSharedCheck_321_;
goto v_resetjp_308_;
}
else
{
lean_inc(v_snd_307_);
lean_inc(v_fst_306_);
lean_dec(v___x_305_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_321_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v_v_x27_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_318_; 
v___x_311_ = lean_nat_mul(v_fst_303_, v_genMag_298_);
lean_dec(v_fst_303_);
v___x_312_ = lean_nat_sub(v_fst_306_, v_genLo_297_);
lean_dec(v_fst_306_);
v_v_x27_313_ = lean_nat_add(v___x_311_, v___x_312_);
lean_dec(v___x_312_);
lean_dec(v___x_311_);
v___x_314_ = lean_nat_div(v_x_299_, v_genMag_298_);
lean_dec(v_x_299_);
v___x_315_ = lean_unsigned_to_nat(1u);
v___x_316_ = lean_nat_sub(v___x_314_, v___x_315_);
lean_dec(v___x_314_);
if (v_isShared_310_ == 0)
{
lean_ctor_set(v___x_309_, 0, v_v_x27_313_);
v___x_318_ = v___x_309_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v_v_x27_313_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v_snd_307_);
v___x_318_ = v_reuseFailAlloc_320_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
v_x_299_ = v___x_316_;
v_x_300_ = v___x_318_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4___boxed(lean_object* v_genLo_322_, lean_object* v_genMag_323_, lean_object* v_x_324_, lean_object* v_x_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(v_genLo_322_, v_genMag_323_, v_x_324_, v_x_325_);
lean_dec(v_genMag_323_);
lean_dec(v_genLo_322_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object* v_g_327_, lean_object* v_lo_328_, lean_object* v_hi_329_){
_start:
{
lean_object* v___y_331_; lean_object* v___y_332_; uint8_t v___x_357_; lean_object* v___y_359_; 
v___x_357_ = lean_nat_dec_lt(v_hi_329_, v_lo_328_);
if (v___x_357_ == 0)
{
v___y_359_ = v_lo_328_;
goto v___jp_358_;
}
else
{
v___y_359_ = v_hi_329_;
goto v___jp_358_;
}
v___jp_330_:
{
lean_object* v___x_333_; lean_object* v_fst_334_; lean_object* v_snd_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v_genMag_338_; lean_object* v_q_339_; lean_object* v___x_340_; lean_object* v_k_341_; lean_object* v_tgtMag_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v_fst_346_; lean_object* v_snd_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_356_; 
v___x_333_ = l_stdRange;
v_fst_334_ = lean_ctor_get(v___x_333_, 0);
v_snd_335_ = lean_ctor_get(v___x_333_, 1);
v___x_336_ = lean_nat_sub(v_snd_335_, v_fst_334_);
v___x_337_ = lean_unsigned_to_nat(1u);
v_genMag_338_ = lean_nat_add(v___x_336_, v___x_337_);
lean_dec(v___x_336_);
v_q_339_ = lean_unsigned_to_nat(1000u);
v___x_340_ = lean_nat_sub(v___y_332_, v___y_331_);
v_k_341_ = lean_nat_add(v___x_340_, v___x_337_);
lean_dec(v___x_340_);
v_tgtMag_342_ = lean_nat_mul(v_k_341_, v_q_339_);
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_343_);
lean_ctor_set(v___x_344_, 1, v_g_327_);
v___x_345_ = lp_plausible___private_Init_Data_Random_0__randNatAux___at___00randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(v_fst_334_, v_genMag_338_, v_tgtMag_342_, v___x_344_);
lean_dec(v_genMag_338_);
v_fst_346_ = lean_ctor_get(v___x_345_, 0);
v_snd_347_ = lean_ctor_get(v___x_345_, 1);
v_isSharedCheck_356_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_356_ == 0)
{
v___x_349_ = v___x_345_;
v_isShared_350_ = v_isSharedCheck_356_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_snd_347_);
lean_inc(v_fst_346_);
lean_dec(v___x_345_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_356_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_351_; lean_object* v_v_x27_352_; lean_object* v___x_354_; 
v___x_351_ = lean_nat_mod(v_fst_346_, v_k_341_);
lean_dec(v_k_341_);
lean_dec(v_fst_346_);
v_v_x27_352_ = lean_nat_add(v___y_331_, v___x_351_);
lean_dec(v___x_351_);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 0, v_v_x27_352_);
v___x_354_ = v___x_349_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v_v_x27_352_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v_snd_347_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
v___jp_358_:
{
if (v___x_357_ == 0)
{
v___y_331_ = v___y_359_;
v___y_332_ = v_hi_329_;
goto v___jp_330_;
}
else
{
v___y_331_ = v___y_359_;
v___y_332_ = v_lo_328_;
goto v___jp_330_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3___boxed(lean_object* v_g_360_, lean_object* v_lo_361_, lean_object* v_hi_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3(v_g_360_, v_lo_361_, v_hi_362_);
lean_dec(v_hi_362_);
lean_dec(v_lo_361_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0(lean_object* v_down_364_){
_start:
{
lean_inc_ref(v_down_364_);
return v_down_364_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0___boxed(lean_object* v_down_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__0(v_down_365_);
lean_dec_ref(v_down_365_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2(lean_object* v_n_368_, lean_object* v_x_369_){
_start:
{
lean_object* v___f_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___f_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___f_370_ = ((lean_object*)(lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___closed__0));
v___x_371_ = lean_unsigned_to_nat(1u);
v___x_372_ = lean_nat_add(v_n_368_, v___x_371_);
v___f_373_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___lam__1___boxed), 2, 1);
lean_closure_set(v___f_373_, 0, v___x_372_);
v___x_374_ = lean_unsigned_to_nat(0u);
v___x_375_ = lp_plausible_randNat___at___00Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2_spec__3(v_x_369_, v___x_374_, v_n_368_);
v___x_376_ = l_Prod_map___redArg(v___f_373_, v___f_370_, v___x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_n_377_, lean_object* v_x_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2(v_n_377_, v_x_378_);
lean_dec(v_n_377_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg(lean_object* v_lo_380_, lean_object* v_hi_381_, lean_object* v_a_382_){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v_fst_385_; lean_object* v_snd_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_394_; 
v___x_383_ = lean_nat_sub(v_hi_381_, v_lo_380_);
v___x_384_ = lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2(v___x_383_, v_a_382_);
lean_dec(v___x_383_);
v_fst_385_ = lean_ctor_get(v___x_384_, 0);
v_snd_386_ = lean_ctor_get(v___x_384_, 1);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_394_ == 0)
{
v___x_388_ = v___x_384_;
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_snd_386_);
lean_inc(v_fst_385_);
lean_dec(v___x_384_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_390_ = lean_nat_add(v_lo_380_, v_fst_385_);
lean_dec(v_fst_385_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_390_);
v___x_392_ = v___x_388_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_snd_386_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg___boxed(lean_object* v_lo_395_, lean_object* v_hi_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg(v_lo_395_, v_hi_396_, v_a_397_);
lean_dec(v_hi_396_);
lean_dec(v_lo_395_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(lean_object* v_lo_399_, lean_object* v_hi_400_, lean_object* v_a_401_){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_402_ = lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg(v_lo_399_, v_hi_400_, v_a_401_);
v___x_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg___boxed(lean_object* v_lo_404_, lean_object* v_hi_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v_lo_404_, v_hi_405_, v_a_406_);
lean_dec(v_hi_405_);
lean_dec(v_lo_404_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___redArg(lean_object* v_lo_408_, lean_object* v_hi_409_, lean_object* v_a_410_, lean_object* v_a_411_){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_432_; 
v___x_412_ = lean_unsigned_to_nat(1u);
v___x_413_ = lean_nat_add(v_lo_408_, v___x_412_);
v___x_414_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_413_, v_hi_409_, v_a_410_);
lean_dec(v___x_413_);
v_a_415_ = lean_ctor_get(v___x_414_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_432_ == 0)
{
v___x_417_ = v___x_414_;
v_isShared_418_ = v_isSharedCheck_432_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_414_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_432_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v_fst_419_; lean_object* v_snd_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_431_; 
v_fst_419_ = lean_ctor_get(v_a_415_, 0);
v_snd_420_ = lean_ctor_get(v_a_415_, 1);
v_isSharedCheck_431_ = !lean_is_exclusive(v_a_415_);
if (v_isSharedCheck_431_ == 0)
{
v___x_422_ = v_a_415_;
v_isShared_423_ = v_isSharedCheck_431_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_snd_420_);
lean_inc(v_fst_419_);
lean_dec(v_a_415_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_431_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___x_424_; lean_object* v___x_426_; 
v___x_424_ = lean_nat_sub(v_fst_419_, v___x_412_);
lean_dec(v_fst_419_);
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 0, v___x_424_);
v___x_426_ = v___x_422_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_424_);
lean_ctor_set(v_reuseFailAlloc_430_, 1, v_snd_420_);
v___x_426_ = v_reuseFailAlloc_430_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
lean_object* v___x_428_; 
if (v_isShared_418_ == 0)
{
lean_ctor_set(v___x_417_, 0, v___x_426_);
v___x_428_ = v___x_417_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v___x_426_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___redArg___boxed(lean_object* v_lo_433_, lean_object* v_hi_434_, lean_object* v_a_435_, lean_object* v_a_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_plausible_Plausible_Gen_chooseNatLt___redArg(v_lo_433_, v_hi_434_, v_a_435_, v_a_436_);
lean_dec(v_a_436_);
lean_dec(v_hi_434_);
lean_dec(v_lo_433_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt(lean_object* v_lo_438_, lean_object* v_hi_439_, lean_object* v_h_440_, lean_object* v_a_441_, lean_object* v_a_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_plausible_Plausible_Gen_chooseNatLt___redArg(v_lo_438_, v_hi_439_, v_a_441_, v_a_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNatLt___boxed(lean_object* v_lo_444_, lean_object* v_hi_445_, lean_object* v_h_446_, lean_object* v_a_447_, lean_object* v_a_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_plausible_Plausible_Gen_chooseNatLt(v_lo_444_, v_hi_445_, v_h_446_, v_a_447_, v_a_448_);
lean_dec(v_a_448_);
lean_dec(v_hi_445_);
lean_dec(v_lo_444_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0(lean_object* v_lo_450_, lean_object* v_hi_451_, lean_object* v_h_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v_lo_450_, v_hi_451_, v_a_453_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___boxed(lean_object* v_lo_456_, lean_object* v_hi_457_, lean_object* v_h_458_, lean_object* v_a_459_, lean_object* v_a_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0(v_lo_456_, v_hi_457_, v_h_458_, v_a_459_, v_a_460_);
lean_dec(v_a_460_);
lean_dec(v_hi_457_);
lean_dec(v_lo_456_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0(lean_object* v_lo_462_, lean_object* v_hi_463_, lean_object* v_h_464_, lean_object* v_a_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___redArg(v_lo_462_, v_hi_463_, v_a_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0___boxed(lean_object* v_lo_467_, lean_object* v_hi_468_, lean_object* v_h_469_, lean_object* v_a_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_plausible_Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0(v_lo_467_, v_hi_468_, v_h_469_, v_a_470_);
lean_dec(v_hi_468_);
lean_dec(v_lo_467_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1(lean_object* v___x_472_, lean_object* v_a_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_plausible_Plausible_Random_randFin___at___00Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1_spec__2(v___x_472_, v_a_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1___boxed(lean_object* v___x_475_, lean_object* v_a_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_plausible_Plausible_Random_rand___at___00Plausible_Random_randBound___at___00Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0_spec__0_spec__1(v___x_475_, v_a_476_);
lean_dec(v___x_475_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_getSize(lean_object* v_a_478_, lean_object* v_a_479_){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; 
lean_inc(v_a_479_);
v___x_480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_480_, 0, v_a_479_);
lean_ctor_set(v___x_480_, 1, v_a_478_);
v___x_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_481_, 0, v___x_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_getSize___boxed(lean_object* v_a_482_, lean_object* v_a_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_plausible_Plausible_Gen_getSize(v_a_482_, v_a_483_);
lean_dec(v_a_483_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___redArg(lean_object* v_f_485_, lean_object* v_x_486_, lean_object* v_a_487_, lean_object* v_a_488_){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; 
lean_inc(v_a_488_);
v___x_489_ = lean_apply_1(v_f_485_, v_a_488_);
v___x_490_ = lean_apply_2(v_x_486_, v_a_487_, v___x_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___redArg___boxed(lean_object* v_f_491_, lean_object* v_x_492_, lean_object* v_a_493_, lean_object* v_a_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_plausible_Plausible_Gen_resize___redArg(v_f_491_, v_x_492_, v_a_493_, v_a_494_);
lean_dec(v_a_494_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize(lean_object* v_00_u03b1_496_, lean_object* v_f_497_, lean_object* v_x_498_, lean_object* v_a_499_, lean_object* v_a_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_plausible_Plausible_Gen_resize___redArg(v_f_497_, v_x_498_, v_a_499_, v_a_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_resize___boxed(lean_object* v_00_u03b1_502_, lean_object* v_f_503_, lean_object* v_x_504_, lean_object* v_a_505_, lean_object* v_a_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_plausible_Plausible_Gen_resize(v_00_u03b1_502_, v_f_503_, v_x_504_, v_a_505_, v_a_506_);
lean_dec(v_a_506_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNat(lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
lean_object* v___x_510_; lean_object* v_a_511_; lean_object* v_fst_512_; lean_object* v_snd_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_532_; 
v___x_510_ = lp_plausible_Plausible_Gen_getSize(v_a_508_, v_a_509_);
v_a_511_ = lean_ctor_get(v___x_510_, 0);
lean_inc(v_a_511_);
lean_dec_ref(v___x_510_);
v_fst_512_ = lean_ctor_get(v_a_511_, 0);
lean_inc(v_fst_512_);
v_snd_513_ = lean_ctor_get(v_a_511_, 1);
lean_inc(v_snd_513_);
lean_dec(v_a_511_);
v___x_514_ = lean_unsigned_to_nat(0u);
v___x_515_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_514_, v_fst_512_, v_snd_513_);
lean_dec(v_fst_512_);
v_a_516_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_532_ == 0)
{
v___x_518_ = v___x_515_;
v_isShared_519_ = v_isSharedCheck_532_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_515_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_532_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v_fst_520_; lean_object* v_snd_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_531_; 
v_fst_520_ = lean_ctor_get(v_a_516_, 0);
v_snd_521_ = lean_ctor_get(v_a_516_, 1);
v_isSharedCheck_531_ = !lean_is_exclusive(v_a_516_);
if (v_isSharedCheck_531_ == 0)
{
v___x_523_ = v_a_516_;
v_isShared_524_ = v_isSharedCheck_531_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_snd_521_);
lean_inc(v_fst_520_);
lean_dec(v_a_516_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_531_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_524_ == 0)
{
v___x_526_ = v___x_523_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_fst_520_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v_snd_521_);
v___x_526_ = v_reuseFailAlloc_530_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
lean_object* v___x_528_; 
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 0, v___x_526_);
v___x_528_ = v___x_518_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_526_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_chooseNat___boxed(lean_object* v_a_533_, lean_object* v_a_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_plausible_Plausible_Gen_chooseNat(v_a_533_, v_a_534_);
lean_dec(v_a_534_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___redArg(lean_object* v_default_538_, lean_object* v_xs_539_, lean_object* v_n_540_){
_start:
{
if (lean_obj_tag(v_xs_539_) == 0)
{
lean_object* v___x_541_; lean_object* v___x_542_; 
lean_dec(v_n_540_);
v___x_541_ = lean_unsigned_to_nat(0u);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
lean_ctor_set(v___x_542_, 1, v_default_538_);
return v___x_542_;
}
else
{
lean_object* v_head_543_; lean_object* v_tail_544_; lean_object* v_fst_545_; uint8_t v___x_546_; 
v_head_543_ = lean_ctor_get(v_xs_539_, 0);
v_tail_544_ = lean_ctor_get(v_xs_539_, 1);
v_fst_545_ = lean_ctor_get(v_head_543_, 0);
v___x_546_ = lean_nat_dec_lt(v_n_540_, v_fst_545_);
if (v___x_546_ == 0)
{
lean_object* v___x_547_; 
v___x_547_ = lean_nat_sub(v_n_540_, v_fst_545_);
lean_dec(v_n_540_);
v_xs_539_ = v_tail_544_;
v_n_540_ = v___x_547_;
goto _start;
}
else
{
lean_dec(v_n_540_);
lean_dec_ref(v_default_538_);
lean_inc(v_head_543_);
return v_head_543_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___redArg___boxed(lean_object* v_default_549_, lean_object* v_xs_550_, lean_object* v_n_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_plausible_Plausible_Gen_pick___redArg(v_default_549_, v_xs_550_, v_n_551_);
lean_dec(v_xs_550_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick(lean_object* v_00_u03b1_553_, lean_object* v_default_554_, lean_object* v_xs_555_, lean_object* v_n_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_plausible_Plausible_Gen_pick___redArg(v_default_554_, v_xs_555_, v_n_556_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_pick___boxed(lean_object* v_00_u03b1_558_, lean_object* v_default_559_, lean_object* v_xs_560_, lean_object* v_n_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_plausible_Plausible_Gen_pick(v_00_u03b1_558_, v_default_559_, v_xs_560_, v_n_561_);
lean_dec(v_xs_560_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen___lam__0(lean_object* v___x_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v___x_566_; 
v___x_566_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_566_, 0, v___x_563_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen___lam__0___boxed(lean_object* v___x_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_plausible_Plausible_Gen_inhabitedGen___lam__0(v___x_567_, v___y_568_, v___y_569_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_inhabitedGen(lean_object* v_00_u03b1_574_){
_start:
{
lean_object* v___f_575_; 
v___f_575_ = ((lean_object*)(lp_plausible_Plausible_Gen_inhabitedGen___closed__1));
return v___f_575_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0(lean_object* v_fail_576_, lean_object* v___y_577_, lean_object* v___y_578_){
_start:
{
lean_object* v___x_579_; 
v___x_579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_579_, 0, v_fail_576_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0___boxed(lean_object* v_fail_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___lam__0(v_fail_580_, v___y_581_, v___y_582_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(lean_object* v_xs_587_, lean_object* v_n_588_){
_start:
{
if (lean_obj_tag(v_xs_587_) == 0)
{
lean_object* v___f_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; 
v___f_589_ = ((lean_object*)(lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___closed__1));
v___x_590_ = lean_unsigned_to_nat(0u);
v___x_591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_591_, 0, v___f_589_);
lean_ctor_set(v___x_591_, 1, v_xs_587_);
v___x_592_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_592_, 0, v___x_590_);
lean_ctor_set(v___x_592_, 1, v___x_591_);
return v___x_592_;
}
else
{
lean_object* v_head_593_; lean_object* v_tail_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_634_; 
v_head_593_ = lean_ctor_get(v_xs_587_, 0);
v_tail_594_ = lean_ctor_get(v_xs_587_, 1);
v_isSharedCheck_634_ = !lean_is_exclusive(v_xs_587_);
if (v_isSharedCheck_634_ == 0)
{
v___x_596_ = v_xs_587_;
v_isShared_597_ = v_isSharedCheck_634_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_tail_594_);
lean_inc(v_head_593_);
lean_dec(v_xs_587_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_634_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v_fst_598_; lean_object* v_snd_599_; uint8_t v___x_600_; 
v_fst_598_ = lean_ctor_get(v_head_593_, 0);
v_snd_599_ = lean_ctor_get(v_head_593_, 1);
v___x_600_ = lean_nat_dec_lt(v_n_588_, v_fst_598_);
if (v___x_600_ == 0)
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v_snd_603_; lean_object* v_fst_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_623_; 
v___x_601_ = lean_nat_sub(v_n_588_, v_fst_598_);
v___x_602_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(v_tail_594_, v___x_601_);
lean_dec(v___x_601_);
v_snd_603_ = lean_ctor_get(v___x_602_, 1);
v_fst_604_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_623_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_623_ == 0)
{
v___x_606_ = v___x_602_;
v_isShared_607_ = v_isSharedCheck_623_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_snd_603_);
lean_inc(v_fst_604_);
lean_dec(v___x_602_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_623_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
lean_object* v_fst_608_; lean_object* v_snd_609_; lean_object* v___x_611_; uint8_t v_isShared_612_; uint8_t v_isSharedCheck_622_; 
v_fst_608_ = lean_ctor_get(v_snd_603_, 0);
v_snd_609_ = lean_ctor_get(v_snd_603_, 1);
v_isSharedCheck_622_ = !lean_is_exclusive(v_snd_603_);
if (v_isSharedCheck_622_ == 0)
{
v___x_611_ = v_snd_603_;
v_isShared_612_ = v_isSharedCheck_622_;
goto v_resetjp_610_;
}
else
{
lean_inc(v_snd_609_);
lean_inc(v_fst_608_);
lean_dec(v_snd_603_);
v___x_611_ = lean_box(0);
v_isShared_612_ = v_isSharedCheck_622_;
goto v_resetjp_610_;
}
v_resetjp_610_:
{
lean_object* v___x_614_; 
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 1, v_snd_609_);
v___x_614_ = v___x_596_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_head_593_);
lean_ctor_set(v_reuseFailAlloc_621_, 1, v_snd_609_);
v___x_614_ = v_reuseFailAlloc_621_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
lean_object* v___x_616_; 
if (v_isShared_612_ == 0)
{
lean_ctor_set(v___x_611_, 1, v___x_614_);
v___x_616_ = v___x_611_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v_fst_608_);
lean_ctor_set(v_reuseFailAlloc_620_, 1, v___x_614_);
v___x_616_ = v_reuseFailAlloc_620_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
lean_object* v___x_618_; 
if (v_isShared_607_ == 0)
{
lean_ctor_set(v___x_606_, 1, v___x_616_);
v___x_618_ = v___x_606_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_fst_604_);
lean_ctor_set(v_reuseFailAlloc_619_, 1, v___x_616_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
}
}
else
{
lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_631_; 
lean_inc(v_snd_599_);
lean_inc(v_fst_598_);
lean_del_object(v___x_596_);
v_isSharedCheck_631_ = !lean_is_exclusive(v_head_593_);
if (v_isSharedCheck_631_ == 0)
{
lean_object* v_unused_632_; lean_object* v_unused_633_; 
v_unused_632_ = lean_ctor_get(v_head_593_, 1);
lean_dec(v_unused_632_);
v_unused_633_ = lean_ctor_get(v_head_593_, 0);
lean_dec(v_unused_633_);
v___x_625_ = v_head_593_;
v_isShared_626_ = v_isSharedCheck_631_;
goto v_resetjp_624_;
}
else
{
lean_dec(v_head_593_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_631_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 1, v_tail_594_);
lean_ctor_set(v___x_625_, 0, v_snd_599_);
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v_snd_599_);
lean_ctor_set(v_reuseFailAlloc_630_, 1, v_tail_594_);
v___x_628_ = v_reuseFailAlloc_630_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
lean_object* v___x_629_; 
v___x_629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_629_, 0, v_fst_598_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
return v___x_629_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg___boxed(lean_object* v_xs_635_, lean_object* v_n_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(v_xs_635_, v_n_636_);
lean_dec(v_n_636_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop(lean_object* v_00_u03b1_638_, lean_object* v_xs_639_, lean_object* v_n_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(v_xs_639_, v_n_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___boxed(lean_object* v_00_u03b1_642_, lean_object* v_xs_643_, lean_object* v_n_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop(v_00_u03b1_642_, v_xs_643_, v_n_644_);
lean_dec(v_n_644_);
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(lean_object* v_init_646_, lean_object* v_x_647_){
_start:
{
if (lean_obj_tag(v_x_647_) == 0)
{
lean_inc(v_init_646_);
return v_init_646_;
}
else
{
lean_object* v_head_648_; lean_object* v_tail_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v_head_648_ = lean_ctor_get(v_x_647_, 0);
v_tail_649_ = lean_ctor_get(v_x_647_, 1);
v___x_650_ = lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(v_init_646_, v_tail_649_);
v___x_651_ = lean_nat_add(v_head_648_, v___x_650_);
lean_dec(v___x_650_);
return v___x_651_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1___boxed(lean_object* v_init_652_, lean_object* v_x_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(v_init_652_, v_x_653_);
lean_dec(v_x_653_);
lean_dec(v_init_652_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1(lean_object* v_l_655_){
_start:
{
lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_656_ = lean_unsigned_to_nat(0u);
v___x_657_ = lp_plausible_List_foldr___at___00List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1_spec__1(v___x_656_, v_l_655_);
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1___boxed(lean_object* v_l_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1(v_l_658_);
lean_dec(v_l_658_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0___redArg(lean_object* v_a_660_, lean_object* v_a_661_){
_start:
{
if (lean_obj_tag(v_a_660_) == 0)
{
lean_object* v___x_662_; 
v___x_662_ = l_List_reverse___redArg(v_a_661_);
return v___x_662_;
}
else
{
lean_object* v_head_663_; lean_object* v_tail_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_673_; 
v_head_663_ = lean_ctor_get(v_a_660_, 0);
v_tail_664_ = lean_ctor_get(v_a_660_, 1);
v_isSharedCheck_673_ = !lean_is_exclusive(v_a_660_);
if (v_isSharedCheck_673_ == 0)
{
v___x_666_ = v_a_660_;
v_isShared_667_ = v_isSharedCheck_673_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_tail_664_);
lean_inc(v_head_663_);
lean_dec(v_a_660_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_673_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v_fst_668_; lean_object* v___x_670_; 
v_fst_668_ = lean_ctor_get(v_head_663_, 0);
lean_inc(v_fst_668_);
lean_dec(v_head_663_);
if (v_isShared_667_ == 0)
{
lean_ctor_set(v___x_666_, 1, v_a_661_);
lean_ctor_set(v___x_666_, 0, v_fst_668_);
v___x_670_ = v___x_666_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_fst_668_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_a_661_);
v___x_670_ = v_reuseFailAlloc_672_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
v_a_660_ = v_tail_664_;
v_a_661_ = v___x_670_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst___redArg(lean_object* v_gs_674_){
_start:
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_675_ = lean_box(0);
v___x_676_ = lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0___redArg(v_gs_674_, v___x_675_);
v___x_677_ = lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1(v___x_676_);
lean_dec(v___x_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst(lean_object* v_00_u03b1_678_, lean_object* v_gs_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst___redArg(v_gs_679_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0(lean_object* v_00_u03b1_681_, lean_object* v_a_682_, lean_object* v_a_683_){
_start:
{
lean_object* v___x_684_; 
v___x_684_ = lp_plausible_List_mapTR_loop___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__0___redArg(v_a_682_, v_a_683_);
return v___x_684_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg(lean_object* v_fuel_687_, lean_object* v_total_688_, lean_object* v_gs_689_, lean_object* v_a_690_, lean_object* v_a_691_){
_start:
{
lean_object* v_zero_692_; uint8_t v_isZero_693_; 
v_zero_692_ = lean_unsigned_to_nat(0u);
v_isZero_693_ = lean_nat_dec_eq(v_fuel_687_, v_zero_692_);
if (v_isZero_693_ == 1)
{
lean_object* v___x_694_; 
lean_dec_ref(v_a_690_);
lean_dec(v_gs_689_);
lean_dec(v_total_688_);
lean_dec(v_fuel_687_);
v___x_694_ = ((lean_object*)(lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___closed__0));
return v___x_694_;
}
else
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v_a_698_; lean_object* v_fst_699_; lean_object* v_snd_700_; lean_object* v___x_701_; lean_object* v_snd_702_; lean_object* v_fst_703_; lean_object* v_fst_704_; lean_object* v_snd_705_; lean_object* v___x_706_; 
v___x_695_ = lean_unsigned_to_nat(1u);
v___x_696_ = lean_nat_sub(v_total_688_, v___x_695_);
v___x_697_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v_zero_692_, v___x_696_, v_a_690_);
lean_dec(v___x_696_);
v_a_698_ = lean_ctor_get(v___x_697_, 0);
lean_inc(v_a_698_);
lean_dec_ref(v___x_697_);
v_fst_699_ = lean_ctor_get(v_a_698_, 0);
lean_inc(v_fst_699_);
v_snd_700_ = lean_ctor_get(v_a_698_, 1);
lean_inc_n(v_snd_700_, 2);
lean_dec(v_a_698_);
v___x_701_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_pickDrop___redArg(v_gs_689_, v_fst_699_);
lean_dec(v_fst_699_);
v_snd_702_ = lean_ctor_get(v___x_701_, 1);
lean_inc(v_snd_702_);
v_fst_703_ = lean_ctor_get(v___x_701_, 0);
lean_inc(v_fst_703_);
lean_dec_ref(v___x_701_);
v_fst_704_ = lean_ctor_get(v_snd_702_, 0);
lean_inc(v_fst_704_);
v_snd_705_ = lean_ctor_get(v_snd_702_, 1);
lean_inc(v_snd_705_);
lean_dec(v_snd_702_);
lean_inc(v_a_691_);
v___x_706_ = lean_apply_2(v_fst_704_, v_snd_700_, v_a_691_);
if (lean_obj_tag(v___x_706_) == 0)
{
lean_object* v_n_707_; lean_object* v___x_708_; 
lean_dec_ref_known(v___x_706_, 1);
v_n_707_ = lean_nat_sub(v_fuel_687_, v___x_695_);
lean_dec(v_fuel_687_);
v___x_708_ = lean_nat_sub(v_total_688_, v_fst_703_);
lean_dec(v_fst_703_);
lean_dec(v_total_688_);
v_fuel_687_ = v_n_707_;
v_total_688_ = v___x_708_;
v_gs_689_ = v_snd_705_;
v_a_690_ = v_snd_700_;
goto _start;
}
else
{
lean_dec(v_snd_705_);
lean_dec(v_fst_703_);
lean_dec(v_snd_700_);
lean_dec(v_total_688_);
lean_dec(v_fuel_687_);
return v___x_706_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg___boxed(lean_object* v_fuel_710_, lean_object* v_total_711_, lean_object* v_gs_712_, lean_object* v_a_713_, lean_object* v_a_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg(v_fuel_710_, v_total_711_, v_gs_712_, v_a_713_, v_a_714_);
lean_dec(v_a_714_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel(lean_object* v_00_u03b1_716_, lean_object* v_fuel_717_, lean_object* v_total_718_, lean_object* v_gs_719_, lean_object* v_a_720_, lean_object* v_a_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg(v_fuel_717_, v_total_718_, v_gs_719_, v_a_720_, v_a_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___boxed(lean_object* v_00_u03b1_723_, lean_object* v_fuel_724_, lean_object* v_total_725_, lean_object* v_gs_726_, lean_object* v_a_727_, lean_object* v_a_728_){
_start:
{
lean_object* v_res_729_; 
v_res_729_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel(v_00_u03b1_723_, v_fuel_724_, v_total_725_, v_gs_726_, v_a_727_, v_a_728_);
lean_dec(v_a_728_);
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___redArg(lean_object* v_gs_730_, lean_object* v_a_731_, lean_object* v_a_732_){
_start:
{
lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_733_ = l_List_lengthTR___redArg(v_gs_730_);
lean_inc(v_gs_730_);
v___x_734_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_sumFst___redArg(v_gs_730_);
v___x_735_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_backtrackFuel___redArg(v___x_733_, v___x_734_, v_gs_730_, v_a_731_, v_a_732_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___redArg___boxed(lean_object* v_gs_736_, lean_object* v_a_737_, lean_object* v_a_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_plausible_Plausible_Gen_backtrack___redArg(v_gs_736_, v_a_737_, v_a_738_);
lean_dec(v_a_738_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack(lean_object* v_00_u03b1_740_, lean_object* v_gs_741_, lean_object* v_a_742_, lean_object* v_a_743_){
_start:
{
lean_object* v___x_744_; 
v___x_744_ = lp_plausible_Plausible_Gen_backtrack___redArg(v_gs_741_, v_a_742_, v_a_743_);
return v___x_744_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_backtrack___boxed(lean_object* v_00_u03b1_745_, lean_object* v_gs_746_, lean_object* v_a_747_, lean_object* v_a_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_plausible_Plausible_Gen_backtrack(v_00_u03b1_745_, v_gs_746_, v_a_747_, v_a_748_);
lean_dec(v_a_748_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___redArg(lean_object* v_default_750_, lean_object* v_gs_751_, lean_object* v_a_752_, lean_object* v_a_753_){
_start:
{
if (lean_obj_tag(v_gs_751_) == 0)
{
lean_object* v___x_754_; 
lean_inc(v_a_753_);
v___x_754_ = lean_apply_2(v_default_750_, v_a_752_, v_a_753_);
return v___x_754_;
}
else
{
lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v_a_760_; lean_object* v_fst_761_; lean_object* v_snd_762_; lean_object* v___x_337__overap_763_; lean_object* v___x_764_; 
v___x_755_ = lean_unsigned_to_nat(0u);
v___x_756_ = l_List_lengthTR___redArg(v_gs_751_);
v___x_757_ = lean_unsigned_to_nat(1u);
v___x_758_ = lean_nat_sub(v___x_756_, v___x_757_);
lean_dec(v___x_756_);
v___x_759_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_755_, v___x_758_, v_a_752_);
lean_dec(v___x_758_);
v_a_760_ = lean_ctor_get(v___x_759_, 0);
lean_inc(v_a_760_);
lean_dec_ref(v___x_759_);
v_fst_761_ = lean_ctor_get(v_a_760_, 0);
lean_inc(v_fst_761_);
v_snd_762_ = lean_ctor_get(v_a_760_, 1);
lean_inc(v_snd_762_);
lean_dec(v_a_760_);
v___x_337__overap_763_ = l_List_getD___redArg(v_gs_751_, v_fst_761_, v_default_750_);
lean_dec_ref(v_default_750_);
lean_inc(v_a_753_);
v___x_764_ = lean_apply_2(v___x_337__overap_763_, v_snd_762_, v_a_753_);
return v___x_764_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___redArg___boxed(lean_object* v_default_765_, lean_object* v_gs_766_, lean_object* v_a_767_, lean_object* v_a_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_plausible_Plausible_Gen_oneOfWithDefault___redArg(v_default_765_, v_gs_766_, v_a_767_, v_a_768_);
lean_dec(v_a_768_);
lean_dec(v_gs_766_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault(lean_object* v_00_u03b1_770_, lean_object* v_default_771_, lean_object* v_gs_772_, lean_object* v_a_773_, lean_object* v_a_774_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lp_plausible_Plausible_Gen_oneOfWithDefault___redArg(v_default_771_, v_gs_772_, v_a_773_, v_a_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOfWithDefault___boxed(lean_object* v_00_u03b1_776_, lean_object* v_default_777_, lean_object* v_gs_778_, lean_object* v_a_779_, lean_object* v_a_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_plausible_Plausible_Gen_oneOfWithDefault(v_00_u03b1_776_, v_default_777_, v_gs_778_, v_a_779_, v_a_780_);
lean_dec(v_a_780_);
lean_dec(v_gs_778_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0___redArg(lean_object* v_a_782_, lean_object* v_a_783_){
_start:
{
if (lean_obj_tag(v_a_782_) == 0)
{
lean_object* v___x_784_; 
v___x_784_ = l_List_reverse___redArg(v_a_783_);
return v___x_784_;
}
else
{
lean_object* v_head_785_; lean_object* v_tail_786_; lean_object* v___x_788_; uint8_t v_isShared_789_; uint8_t v_isSharedCheck_795_; 
v_head_785_ = lean_ctor_get(v_a_782_, 0);
v_tail_786_ = lean_ctor_get(v_a_782_, 1);
v_isSharedCheck_795_ = !lean_is_exclusive(v_a_782_);
if (v_isSharedCheck_795_ == 0)
{
v___x_788_ = v_a_782_;
v_isShared_789_ = v_isSharedCheck_795_;
goto v_resetjp_787_;
}
else
{
lean_inc(v_tail_786_);
lean_inc(v_head_785_);
lean_dec(v_a_782_);
v___x_788_ = lean_box(0);
v_isShared_789_ = v_isSharedCheck_795_;
goto v_resetjp_787_;
}
v_resetjp_787_:
{
lean_object* v_fst_790_; lean_object* v___x_792_; 
v_fst_790_ = lean_ctor_get(v_head_785_, 0);
lean_inc(v_fst_790_);
lean_dec(v_head_785_);
if (v_isShared_789_ == 0)
{
lean_ctor_set(v___x_788_, 1, v_a_783_);
lean_ctor_set(v___x_788_, 0, v_fst_790_);
v___x_792_ = v___x_788_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_fst_790_);
lean_ctor_set(v_reuseFailAlloc_794_, 1, v_a_783_);
v___x_792_ = v_reuseFailAlloc_794_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
v_a_782_ = v_tail_786_;
v_a_783_ = v___x_792_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___redArg(lean_object* v_default_796_, lean_object* v_gs_797_, lean_object* v_a_798_, lean_object* v_a_799_){
_start:
{
lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v_total_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v_a_807_; lean_object* v_fst_808_; lean_object* v_snd_809_; lean_object* v___x_810_; lean_object* v_snd_811_; lean_object* v___x_812_; 
v___x_800_ = lean_unsigned_to_nat(0u);
v___x_801_ = lean_box(0);
lean_inc(v_gs_797_);
v___x_802_ = lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0___redArg(v_gs_797_, v___x_801_);
v_total_803_ = lp_plausible_List_sum___at___00__private_Plausible_Gen_0__Plausible_Gen_sumFst_spec__1(v___x_802_);
lean_dec(v___x_802_);
v___x_804_ = lean_unsigned_to_nat(1u);
v___x_805_ = lean_nat_sub(v_total_803_, v___x_804_);
lean_dec(v_total_803_);
v___x_806_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_800_, v___x_805_, v_a_798_);
lean_dec(v___x_805_);
v_a_807_ = lean_ctor_get(v___x_806_, 0);
lean_inc(v_a_807_);
lean_dec_ref(v___x_806_);
v_fst_808_ = lean_ctor_get(v_a_807_, 0);
lean_inc(v_fst_808_);
v_snd_809_ = lean_ctor_get(v_a_807_, 1);
lean_inc(v_snd_809_);
lean_dec(v_a_807_);
v___x_810_ = lp_plausible_Plausible_Gen_pick___redArg(v_default_796_, v_gs_797_, v_fst_808_);
lean_dec(v_gs_797_);
v_snd_811_ = lean_ctor_get(v___x_810_, 1);
lean_inc(v_snd_811_);
lean_dec_ref(v___x_810_);
lean_inc(v_a_799_);
v___x_812_ = lean_apply_2(v_snd_811_, v_snd_809_, v_a_799_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___redArg___boxed(lean_object* v_default_813_, lean_object* v_gs_814_, lean_object* v_a_815_, lean_object* v_a_816_){
_start:
{
lean_object* v_res_817_; 
v_res_817_ = lp_plausible_Plausible_Gen_frequency___redArg(v_default_813_, v_gs_814_, v_a_815_, v_a_816_);
lean_dec(v_a_816_);
return v_res_817_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency(lean_object* v_00_u03b1_818_, lean_object* v_default_819_, lean_object* v_gs_820_, lean_object* v_a_821_, lean_object* v_a_822_){
_start:
{
lean_object* v___x_823_; 
v___x_823_ = lp_plausible_Plausible_Gen_frequency___redArg(v_default_819_, v_gs_820_, v_a_821_, v_a_822_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_frequency___boxed(lean_object* v_00_u03b1_824_, lean_object* v_default_825_, lean_object* v_gs_826_, lean_object* v_a_827_, lean_object* v_a_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_plausible_Plausible_Gen_frequency(v_00_u03b1_824_, v_default_825_, v_gs_826_, v_a_827_, v_a_828_);
lean_dec(v_a_828_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0(lean_object* v_00_u03b1_830_, lean_object* v_a_831_, lean_object* v_a_832_){
_start:
{
lean_object* v___x_833_; 
v___x_833_ = lp_plausible_List_mapTR_loop___at___00Plausible_Gen_frequency_spec__0___redArg(v_a_831_, v_a_832_);
return v___x_833_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___redArg(lean_object* v_f_834_, lean_object* v_a_835_, lean_object* v_a_836_){
_start:
{
lean_object* v___x_837_; lean_object* v_a_838_; lean_object* v_fst_839_; lean_object* v_snd_840_; lean_object* v___x_841_; 
v___x_837_ = lp_plausible_Plausible_Gen_getSize(v_a_835_, v_a_836_);
v_a_838_ = lean_ctor_get(v___x_837_, 0);
lean_inc(v_a_838_);
lean_dec_ref(v___x_837_);
v_fst_839_ = lean_ctor_get(v_a_838_, 0);
lean_inc(v_fst_839_);
v_snd_840_ = lean_ctor_get(v_a_838_, 1);
lean_inc(v_snd_840_);
lean_dec(v_a_838_);
lean_inc(v_a_836_);
v___x_841_ = lean_apply_3(v_f_834_, v_fst_839_, v_snd_840_, v_a_836_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___redArg___boxed(lean_object* v_f_842_, lean_object* v_a_843_, lean_object* v_a_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_plausible_Plausible_Gen_sized___redArg(v_f_842_, v_a_843_, v_a_844_);
lean_dec(v_a_844_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized(lean_object* v_00_u03b1_846_, lean_object* v_f_847_, lean_object* v_a_848_, lean_object* v_a_849_){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lp_plausible_Plausible_Gen_sized___redArg(v_f_847_, v_a_848_, v_a_849_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_sized___boxed(lean_object* v_00_u03b1_851_, lean_object* v_f_852_, lean_object* v_a_853_, lean_object* v_a_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_plausible_Plausible_Gen_sized(v_00_u03b1_851_, v_f_852_, v_a_853_, v_a_854_);
lean_dec(v_a_854_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg(lean_object* v_x_856_, lean_object* v_range_857_, lean_object* v_b_858_, lean_object* v_i_859_, lean_object* v___y_860_, lean_object* v___y_861_){
_start:
{
lean_object* v_stop_862_; lean_object* v_step_863_; uint8_t v___x_864_; 
v_stop_862_ = lean_ctor_get(v_range_857_, 1);
v_step_863_ = lean_ctor_get(v_range_857_, 2);
v___x_864_ = lean_nat_dec_lt(v_i_859_, v_stop_862_);
if (v___x_864_ == 0)
{
lean_object* v___x_865_; lean_object* v___x_866_; 
lean_dec(v_i_859_);
lean_dec_ref(v_x_856_);
v___x_865_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_865_, 0, v_b_858_);
lean_ctor_set(v___x_865_, 1, v___y_860_);
v___x_866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_866_, 0, v___x_865_);
return v___x_866_;
}
else
{
lean_object* v___x_867_; 
lean_inc_ref(v_x_856_);
lean_inc(v___y_861_);
v___x_867_ = lean_apply_2(v_x_856_, v___y_860_, v___y_861_);
if (lean_obj_tag(v___x_867_) == 0)
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_875_; 
lean_dec(v_i_859_);
lean_dec_ref(v_b_858_);
lean_dec_ref(v_x_856_);
v_a_868_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_875_ == 0)
{
v___x_870_ = v___x_867_;
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_867_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_873_; 
if (v_isShared_871_ == 0)
{
v___x_873_ = v___x_870_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v_a_868_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
else
{
lean_object* v_a_876_; lean_object* v_fst_877_; lean_object* v_snd_878_; lean_object* v___x_879_; lean_object* v___x_880_; 
v_a_876_ = lean_ctor_get(v___x_867_, 0);
lean_inc(v_a_876_);
lean_dec_ref_known(v___x_867_, 1);
v_fst_877_ = lean_ctor_get(v_a_876_, 0);
lean_inc(v_fst_877_);
v_snd_878_ = lean_ctor_get(v_a_876_, 1);
lean_inc(v_snd_878_);
lean_dec(v_a_876_);
v___x_879_ = lean_array_push(v_b_858_, v_fst_877_);
v___x_880_ = lean_nat_add(v_i_859_, v_step_863_);
lean_dec(v_i_859_);
v_b_858_ = v___x_879_;
v_i_859_ = v___x_880_;
v___y_860_ = v_snd_878_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg___boxed(lean_object* v_x_882_, lean_object* v_range_883_, lean_object* v_b_884_, lean_object* v_i_885_, lean_object* v___y_886_, lean_object* v___y_887_){
_start:
{
lean_object* v_res_888_; 
v_res_888_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg(v_x_882_, v_range_883_, v_b_884_, v_i_885_, v___y_886_, v___y_887_);
lean_dec(v___y_887_);
lean_dec_ref(v_range_883_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___redArg(lean_object* v_x_889_, lean_object* v_a_890_, lean_object* v_a_891_){
_start:
{
lean_object* v___x_892_; 
v___x_892_ = lp_plausible_Plausible_Gen_chooseNat(v_a_890_, v_a_891_);
if (lean_obj_tag(v___x_892_) == 0)
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_900_; 
lean_dec_ref(v_x_889_);
v_a_893_ = lean_ctor_get(v___x_892_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_900_ == 0)
{
v___x_895_ = v___x_892_;
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_892_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_898_; 
if (v_isShared_896_ == 0)
{
v___x_898_ = v___x_895_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_893_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
else
{
lean_object* v_a_901_; lean_object* v_fst_902_; lean_object* v_snd_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v_a_901_ = lean_ctor_get(v___x_892_, 0);
lean_inc(v_a_901_);
lean_dec_ref_known(v___x_892_, 1);
v_fst_902_ = lean_ctor_get(v_a_901_, 0);
lean_inc(v_fst_902_);
v_snd_903_ = lean_ctor_get(v_a_901_, 1);
lean_inc(v_snd_903_);
lean_dec(v_a_901_);
v___x_904_ = lean_mk_empty_array_with_capacity(v_fst_902_);
v___x_905_ = lean_unsigned_to_nat(0u);
v___x_906_ = lean_unsigned_to_nat(1u);
v___x_907_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_907_, 0, v___x_905_);
lean_ctor_set(v___x_907_, 1, v_fst_902_);
lean_ctor_set(v___x_907_, 2, v___x_906_);
v___x_908_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg(v_x_889_, v___x_907_, v___x_904_, v___x_905_, v_snd_903_, v_a_891_);
lean_dec_ref_known(v___x_907_, 3);
return v___x_908_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___redArg___boxed(lean_object* v_x_909_, lean_object* v_a_910_, lean_object* v_a_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_plausible_Plausible_Gen_arrayOf___redArg(v_x_909_, v_a_910_, v_a_911_);
lean_dec(v_a_911_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf(lean_object* v_00_u03b1_913_, lean_object* v_x_914_, lean_object* v_a_915_, lean_object* v_a_916_){
_start:
{
lean_object* v___x_917_; 
v___x_917_ = lp_plausible_Plausible_Gen_arrayOf___redArg(v_x_914_, v_a_915_, v_a_916_);
return v___x_917_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_arrayOf___boxed(lean_object* v_00_u03b1_918_, lean_object* v_x_919_, lean_object* v_a_920_, lean_object* v_a_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_plausible_Plausible_Gen_arrayOf(v_00_u03b1_918_, v_x_919_, v_a_920_, v_a_921_);
lean_dec(v_a_921_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0(lean_object* v_00_u03b1_923_, lean_object* v_x_924_, lean_object* v_range_925_, lean_object* v_b_926_, lean_object* v_i_927_, lean_object* v_hs_928_, lean_object* v_hl_929_, lean_object* v___y_930_, lean_object* v___y_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___redArg(v_x_924_, v_range_925_, v_b_926_, v_i_927_, v___y_930_, v___y_931_);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0___boxed(lean_object* v_00_u03b1_933_, lean_object* v_x_934_, lean_object* v_range_935_, lean_object* v_b_936_, lean_object* v_i_937_, lean_object* v_hs_938_, lean_object* v_hl_939_, lean_object* v___y_940_, lean_object* v___y_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_plausible___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Plausible_Gen_arrayOf_spec__0(v_00_u03b1_933_, v_x_934_, v_range_935_, v_b_936_, v_i_937_, v_hs_938_, v_hl_939_, v___y_940_, v___y_941_);
lean_dec(v___y_941_);
lean_dec_ref(v_range_935_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___redArg(lean_object* v_x_943_, lean_object* v_a_944_, lean_object* v_a_945_){
_start:
{
lean_object* v___x_946_; 
v___x_946_ = lp_plausible_Plausible_Gen_arrayOf___redArg(v_x_943_, v_a_944_, v_a_945_);
if (lean_obj_tag(v___x_946_) == 0)
{
lean_object* v_a_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_954_; 
v_a_947_ = lean_ctor_get(v___x_946_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_946_);
if (v_isSharedCheck_954_ == 0)
{
v___x_949_ = v___x_946_;
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_a_947_);
lean_dec(v___x_946_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
lean_object* v___x_952_; 
if (v_isShared_950_ == 0)
{
v___x_952_ = v___x_949_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v_a_947_);
v___x_952_ = v_reuseFailAlloc_953_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
return v___x_952_;
}
}
}
else
{
lean_object* v_a_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_972_; 
v_a_955_ = lean_ctor_get(v___x_946_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v___x_946_);
if (v_isSharedCheck_972_ == 0)
{
v___x_957_ = v___x_946_;
v_isShared_958_ = v_isSharedCheck_972_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_a_955_);
lean_dec(v___x_946_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_972_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v_fst_959_; lean_object* v_snd_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_971_; 
v_fst_959_ = lean_ctor_get(v_a_955_, 0);
v_snd_960_ = lean_ctor_get(v_a_955_, 1);
v_isSharedCheck_971_ = !lean_is_exclusive(v_a_955_);
if (v_isSharedCheck_971_ == 0)
{
v___x_962_ = v_a_955_;
v_isShared_963_ = v_isSharedCheck_971_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_snd_960_);
lean_inc(v_fst_959_);
lean_dec(v_a_955_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_971_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_964_; lean_object* v___x_966_; 
v___x_964_ = lean_array_to_list(v_fst_959_);
if (v_isShared_963_ == 0)
{
lean_ctor_set(v___x_962_, 0, v___x_964_);
v___x_966_ = v___x_962_;
goto v_reusejp_965_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v___x_964_);
lean_ctor_set(v_reuseFailAlloc_970_, 1, v_snd_960_);
v___x_966_ = v_reuseFailAlloc_970_;
goto v_reusejp_965_;
}
v_reusejp_965_:
{
lean_object* v___x_968_; 
if (v_isShared_958_ == 0)
{
lean_ctor_set(v___x_957_, 0, v___x_966_);
v___x_968_ = v___x_957_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___redArg___boxed(lean_object* v_x_973_, lean_object* v_a_974_, lean_object* v_a_975_){
_start:
{
lean_object* v_res_976_; 
v_res_976_ = lp_plausible_Plausible_Gen_listOf___redArg(v_x_973_, v_a_974_, v_a_975_);
lean_dec(v_a_975_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf(lean_object* v_00_u03b1_977_, lean_object* v_x_978_, lean_object* v_a_979_, lean_object* v_a_980_){
_start:
{
lean_object* v___x_981_; 
v___x_981_ = lp_plausible_Plausible_Gen_listOf___redArg(v_x_978_, v_a_979_, v_a_980_);
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_listOf___boxed(lean_object* v_00_u03b1_982_, lean_object* v_x_983_, lean_object* v_a_984_, lean_object* v_a_985_){
_start:
{
lean_object* v_res_986_; 
v_res_986_ = lp_plausible_Plausible_Gen_listOf(v_00_u03b1_982_, v_x_983_, v_a_984_, v_a_985_);
lean_dec(v_a_985_);
return v_res_986_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12(void){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__10));
v___x_1014_ = l_Lean_mkAtom(v___x_1013_);
return v___x_1014_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13(void){
_start:
{
lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1015_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__12);
v___x_1016_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5));
v___x_1017_ = lean_array_push(v___x_1016_, v___x_1015_);
return v___x_1017_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17(void){
_start:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; 
v___x_1028_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__16));
v___x_1029_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5));
v___x_1030_ = lean_array_push(v___x_1029_, v___x_1028_);
return v___x_1030_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18(void){
_start:
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; 
v___x_1031_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__17);
v___x_1032_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__15));
v___x_1033_ = lean_box(2);
v___x_1034_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1033_);
lean_ctor_set(v___x_1034_, 1, v___x_1032_);
lean_ctor_set(v___x_1034_, 2, v___x_1031_);
return v___x_1034_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19(void){
_start:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; 
v___x_1035_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__18);
v___x_1036_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__13);
v___x_1037_ = lean_array_push(v___x_1036_, v___x_1035_);
return v___x_1037_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20(void){
_start:
{
lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1038_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__19);
v___x_1039_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__11));
v___x_1040_ = lean_box(2);
v___x_1041_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
lean_ctor_set(v___x_1041_, 1, v___x_1039_);
lean_ctor_set(v___x_1041_, 2, v___x_1038_);
return v___x_1041_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21(void){
_start:
{
lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1042_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__20);
v___x_1043_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5));
v___x_1044_ = lean_array_push(v___x_1043_, v___x_1042_);
return v___x_1044_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22(void){
_start:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1045_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__21);
v___x_1046_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__9));
v___x_1047_ = lean_box(2);
v___x_1048_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1047_);
lean_ctor_set(v___x_1048_, 1, v___x_1046_);
lean_ctor_set(v___x_1048_, 2, v___x_1045_);
return v___x_1048_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23(void){
_start:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; 
v___x_1049_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__22);
v___x_1050_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5));
v___x_1051_ = lean_array_push(v___x_1050_, v___x_1049_);
return v___x_1051_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24(void){
_start:
{
lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; 
v___x_1052_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__23);
v___x_1053_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__7));
v___x_1054_ = lean_box(2);
v___x_1055_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1055_, 0, v___x_1054_);
lean_ctor_set(v___x_1055_, 1, v___x_1053_);
lean_ctor_set(v___x_1055_, 2, v___x_1052_);
return v___x_1055_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25(void){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1056_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__24);
v___x_1057_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__5));
v___x_1058_ = lean_array_push(v___x_1057_, v___x_1056_);
return v___x_1058_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26(void){
_start:
{
lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1059_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__25);
v___x_1060_ = ((lean_object*)(lp_plausible_Plausible_Gen_oneOf___auto__1___closed__4));
v___x_1061_ = lean_box(2);
v___x_1062_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
lean_ctor_set(v___x_1062_, 1, v___x_1060_);
lean_ctor_set(v___x_1062_, 2, v___x_1059_);
return v___x_1062_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_oneOf___auto__1(void){
_start:
{
lean_object* v___x_1063_; 
v___x_1063_ = lean_obj_once(&lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26, &lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26_once, _init_lp_plausible_Plausible_Gen_oneOf___auto__1___closed__26);
return v___x_1063_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___redArg(lean_object* v_xs_1064_, lean_object* v_a_1065_, lean_object* v_a_1066_){
_start:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v_a_1070_; lean_object* v_fst_1071_; lean_object* v_snd_1072_; lean_object* v___x_307__overap_1073_; lean_object* v___x_1074_; 
v___x_1067_ = lean_unsigned_to_nat(0u);
v___x_1068_ = lean_array_get_size(v_xs_1064_);
v___x_1069_ = lp_plausible_Plausible_Gen_chooseNatLt___redArg(v___x_1067_, v___x_1068_, v_a_1065_, v_a_1066_);
v_a_1070_ = lean_ctor_get(v___x_1069_, 0);
lean_inc(v_a_1070_);
lean_dec_ref(v___x_1069_);
v_fst_1071_ = lean_ctor_get(v_a_1070_, 0);
lean_inc(v_fst_1071_);
v_snd_1072_ = lean_ctor_get(v_a_1070_, 1);
lean_inc(v_snd_1072_);
lean_dec(v_a_1070_);
v___x_307__overap_1073_ = lean_array_fget_borrowed(v_xs_1064_, v_fst_1071_);
lean_dec(v_fst_1071_);
lean_inc(v___x_307__overap_1073_);
lean_inc(v_a_1066_);
v___x_1074_ = lean_apply_2(v___x_307__overap_1073_, v_snd_1072_, v_a_1066_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___redArg___boxed(lean_object* v_xs_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_){
_start:
{
lean_object* v_res_1078_; 
v_res_1078_ = lp_plausible_Plausible_Gen_oneOf___redArg(v_xs_1075_, v_a_1076_, v_a_1077_);
lean_dec(v_a_1077_);
lean_dec_ref(v_xs_1075_);
return v_res_1078_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf(lean_object* v_00_u03b1_1079_, lean_object* v_xs_1080_, lean_object* v_pos_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lp_plausible_Plausible_Gen_oneOf___redArg(v_xs_1080_, v_a_1082_, v_a_1083_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_oneOf___boxed(lean_object* v_00_u03b1_1085_, lean_object* v_xs_1086_, lean_object* v_pos_1087_, lean_object* v_a_1088_, lean_object* v_a_1089_){
_start:
{
lean_object* v_res_1090_; 
v_res_1090_ = lp_plausible_Plausible_Gen_oneOf(v_00_u03b1_1085_, v_xs_1086_, v_pos_1087_, v_a_1088_, v_a_1089_);
lean_dec(v_a_1089_);
lean_dec_ref(v_xs_1086_);
return v_res_1090_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___redArg(lean_object* v_xs_1091_, lean_object* v_a_1092_, lean_object* v_a_1093_){
_start:
{
lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1114_; 
v___x_1094_ = lean_unsigned_to_nat(0u);
v___x_1095_ = l_List_lengthTR___redArg(v_xs_1091_);
v___x_1096_ = lp_plausible_Plausible_Gen_chooseNatLt___redArg(v___x_1094_, v___x_1095_, v_a_1092_, v_a_1093_);
lean_dec(v___x_1095_);
v_a_1097_ = lean_ctor_get(v___x_1096_, 0);
v_isSharedCheck_1114_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1114_ == 0)
{
v___x_1099_ = v___x_1096_;
v_isShared_1100_ = v_isSharedCheck_1114_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1096_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1114_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v_fst_1101_; lean_object* v_snd_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1113_; 
v_fst_1101_ = lean_ctor_get(v_a_1097_, 0);
v_snd_1102_ = lean_ctor_get(v_a_1097_, 1);
v_isSharedCheck_1113_ = !lean_is_exclusive(v_a_1097_);
if (v_isSharedCheck_1113_ == 0)
{
v___x_1104_ = v_a_1097_;
v_isShared_1105_ = v_isSharedCheck_1113_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_snd_1102_);
lean_inc(v_fst_1101_);
lean_dec(v_a_1097_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1113_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1106_; lean_object* v___x_1108_; 
v___x_1106_ = l_List_get___redArg(v_xs_1091_, v_fst_1101_);
if (v_isShared_1105_ == 0)
{
lean_ctor_set(v___x_1104_, 0, v___x_1106_);
v___x_1108_ = v___x_1104_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v___x_1106_);
lean_ctor_set(v_reuseFailAlloc_1112_, 1, v_snd_1102_);
v___x_1108_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
lean_object* v___x_1110_; 
if (v_isShared_1100_ == 0)
{
lean_ctor_set(v___x_1099_, 0, v___x_1108_);
v___x_1110_ = v___x_1099_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v___x_1108_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___redArg___boxed(lean_object* v_xs_1115_, lean_object* v_a_1116_, lean_object* v_a_1117_){
_start:
{
lean_object* v_res_1118_; 
v_res_1118_ = lp_plausible_Plausible_Gen_elements___redArg(v_xs_1115_, v_a_1116_, v_a_1117_);
lean_dec(v_a_1117_);
lean_dec(v_xs_1115_);
return v_res_1118_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements(lean_object* v_00_u03b1_1119_, lean_object* v_xs_1120_, lean_object* v_pos_1121_, lean_object* v_a_1122_, lean_object* v_a_1123_){
_start:
{
lean_object* v___x_1124_; 
v___x_1124_ = lp_plausible_Plausible_Gen_elements___redArg(v_xs_1120_, v_a_1122_, v_a_1123_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_elements___boxed(lean_object* v_00_u03b1_1125_, lean_object* v_xs_1126_, lean_object* v_pos_1127_, lean_object* v_a_1128_, lean_object* v_a_1129_){
_start:
{
lean_object* v_res_1130_; 
v_res_1130_ = lp_plausible_Plausible_Gen_elements(v_00_u03b1_1125_, v_xs_1126_, v_pos_1127_, v_a_1128_, v_a_1129_);
lean_dec(v_a_1129_);
lean_dec(v_xs_1126_);
return v_res_1130_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf___redArg(lean_object* v_x_1133_, lean_object* v_a_1134_){
_start:
{
if (lean_obj_tag(v_x_1133_) == 0)
{
lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1135_, 0, v_x_1133_);
lean_ctor_set(v___x_1135_, 1, v_a_1134_);
v___x_1136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1136_, 0, v___x_1135_);
return v___x_1136_;
}
else
{
lean_object* v_head_1137_; lean_object* v_tail_1138_; lean_object* v___x_1139_; 
v_head_1137_ = lean_ctor_get(v_x_1133_, 0);
lean_inc(v_head_1137_);
v_tail_1138_ = lean_ctor_get(v_x_1133_, 1);
lean_inc(v_tail_1138_);
lean_dec_ref_known(v_x_1133_, 2);
v___x_1139_ = lp_plausible_Plausible_Gen_permutationOf___redArg(v_tail_1138_, v_a_1134_);
if (lean_obj_tag(v___x_1139_) == 0)
{
lean_dec(v_head_1137_);
return v___x_1139_;
}
else
{
lean_object* v_a_1140_; lean_object* v_fst_1141_; lean_object* v_snd_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v_a_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1164_; 
v_a_1140_ = lean_ctor_get(v___x_1139_, 0);
lean_inc(v_a_1140_);
lean_dec_ref_known(v___x_1139_, 1);
v_fst_1141_ = lean_ctor_get(v_a_1140_, 0);
lean_inc(v_fst_1141_);
v_snd_1142_ = lean_ctor_get(v_a_1140_, 1);
lean_inc(v_snd_1142_);
lean_dec(v_a_1140_);
v___x_1143_ = lean_unsigned_to_nat(0u);
v___x_1144_ = l_List_lengthTR___redArg(v_fst_1141_);
v___x_1145_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_1143_, v___x_1144_, v_snd_1142_);
lean_dec(v___x_1144_);
v_a_1146_ = lean_ctor_get(v___x_1145_, 0);
v_isSharedCheck_1164_ = !lean_is_exclusive(v___x_1145_);
if (v_isSharedCheck_1164_ == 0)
{
v___x_1148_ = v___x_1145_;
v_isShared_1149_ = v_isSharedCheck_1164_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_a_1146_);
lean_dec(v___x_1145_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1164_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v_fst_1150_; lean_object* v_snd_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1163_; 
v_fst_1150_ = lean_ctor_get(v_a_1146_, 0);
v_snd_1151_ = lean_ctor_get(v_a_1146_, 1);
v_isSharedCheck_1163_ = !lean_is_exclusive(v_a_1146_);
if (v_isSharedCheck_1163_ == 0)
{
v___x_1153_ = v_a_1146_;
v_isShared_1154_ = v_isSharedCheck_1163_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_snd_1151_);
lean_inc(v_fst_1150_);
lean_dec(v_a_1146_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1163_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1158_; 
v___x_1155_ = ((lean_object*)(lp_plausible_Plausible_Gen_permutationOf___redArg___closed__0));
v___x_1156_ = l___private_Init_Data_List_Impl_0__List_insertIdxTR_go(lean_box(0), v_head_1137_, v_fst_1150_, v_fst_1141_, v___x_1155_);
if (v_isShared_1154_ == 0)
{
lean_ctor_set(v___x_1153_, 0, v___x_1156_);
v___x_1158_ = v___x_1153_;
goto v_reusejp_1157_;
}
else
{
lean_object* v_reuseFailAlloc_1162_; 
v_reuseFailAlloc_1162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1162_, 0, v___x_1156_);
lean_ctor_set(v_reuseFailAlloc_1162_, 1, v_snd_1151_);
v___x_1158_ = v_reuseFailAlloc_1162_;
goto v_reusejp_1157_;
}
v_reusejp_1157_:
{
lean_object* v___x_1160_; 
if (v_isShared_1149_ == 0)
{
lean_ctor_set(v___x_1148_, 0, v___x_1158_);
v___x_1160_ = v___x_1148_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v___x_1158_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf(lean_object* v_00_u03b1_1165_, lean_object* v_x_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_){
_start:
{
lean_object* v___x_1169_; 
v___x_1169_ = lp_plausible_Plausible_Gen_permutationOf___redArg(v_x_1166_, v_a_1167_);
return v___x_1169_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_permutationOf___boxed(lean_object* v_00_u03b1_1170_, lean_object* v_x_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_){
_start:
{
lean_object* v_res_1174_; 
v_res_1174_ = lp_plausible_Plausible_Gen_permutationOf(v_00_u03b1_1170_, v_x_1171_, v_a_1172_, v_a_1173_);
lean_dec(v_a_1173_);
return v_res_1174_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___redArg(lean_object* v_x_1175_, lean_object* v_y_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_){
_start:
{
lean_object* v___x_1179_; 
lean_inc(v_a_1178_);
v___x_1179_ = lean_apply_2(v_x_1175_, v_a_1177_, v_a_1178_);
if (lean_obj_tag(v___x_1179_) == 0)
{
lean_object* v_a_1180_; lean_object* v___x_1182_; uint8_t v_isShared_1183_; uint8_t v_isSharedCheck_1187_; 
lean_dec_ref(v_y_1176_);
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1187_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1187_ == 0)
{
v___x_1182_ = v___x_1179_;
v_isShared_1183_ = v_isSharedCheck_1187_;
goto v_resetjp_1181_;
}
else
{
lean_inc(v_a_1180_);
lean_dec(v___x_1179_);
v___x_1182_ = lean_box(0);
v_isShared_1183_ = v_isSharedCheck_1187_;
goto v_resetjp_1181_;
}
v_resetjp_1181_:
{
lean_object* v___x_1185_; 
if (v_isShared_1183_ == 0)
{
v___x_1185_ = v___x_1182_;
goto v_reusejp_1184_;
}
else
{
lean_object* v_reuseFailAlloc_1186_; 
v_reuseFailAlloc_1186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1186_, 0, v_a_1180_);
v___x_1185_ = v_reuseFailAlloc_1186_;
goto v_reusejp_1184_;
}
v_reusejp_1184_:
{
return v___x_1185_;
}
}
}
else
{
lean_object* v_a_1188_; lean_object* v_fst_1189_; lean_object* v_snd_1190_; lean_object* v___x_1192_; uint8_t v_isShared_1193_; uint8_t v_isSharedCheck_1223_; 
v_a_1188_ = lean_ctor_get(v___x_1179_, 0);
lean_inc(v_a_1188_);
lean_dec_ref_known(v___x_1179_, 1);
v_fst_1189_ = lean_ctor_get(v_a_1188_, 0);
v_snd_1190_ = lean_ctor_get(v_a_1188_, 1);
v_isSharedCheck_1223_ = !lean_is_exclusive(v_a_1188_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1192_ = v_a_1188_;
v_isShared_1193_ = v_isSharedCheck_1223_;
goto v_resetjp_1191_;
}
else
{
lean_inc(v_snd_1190_);
lean_inc(v_fst_1189_);
lean_dec(v_a_1188_);
v___x_1192_ = lean_box(0);
v_isShared_1193_ = v_isSharedCheck_1223_;
goto v_resetjp_1191_;
}
v_resetjp_1191_:
{
lean_object* v___x_1194_; 
lean_inc(v_a_1178_);
v___x_1194_ = lean_apply_2(v_y_1176_, v_snd_1190_, v_a_1178_);
if (lean_obj_tag(v___x_1194_) == 0)
{
lean_object* v_a_1195_; lean_object* v___x_1197_; uint8_t v_isShared_1198_; uint8_t v_isSharedCheck_1202_; 
lean_del_object(v___x_1192_);
lean_dec(v_fst_1189_);
v_a_1195_ = lean_ctor_get(v___x_1194_, 0);
v_isSharedCheck_1202_ = !lean_is_exclusive(v___x_1194_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1197_ = v___x_1194_;
v_isShared_1198_ = v_isSharedCheck_1202_;
goto v_resetjp_1196_;
}
else
{
lean_inc(v_a_1195_);
lean_dec(v___x_1194_);
v___x_1197_ = lean_box(0);
v_isShared_1198_ = v_isSharedCheck_1202_;
goto v_resetjp_1196_;
}
v_resetjp_1196_:
{
lean_object* v___x_1200_; 
if (v_isShared_1198_ == 0)
{
v___x_1200_ = v___x_1197_;
goto v_reusejp_1199_;
}
else
{
lean_object* v_reuseFailAlloc_1201_; 
v_reuseFailAlloc_1201_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1201_, 0, v_a_1195_);
v___x_1200_ = v_reuseFailAlloc_1201_;
goto v_reusejp_1199_;
}
v_reusejp_1199_:
{
return v___x_1200_;
}
}
}
else
{
lean_object* v_a_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1222_; 
v_a_1203_ = lean_ctor_get(v___x_1194_, 0);
v_isSharedCheck_1222_ = !lean_is_exclusive(v___x_1194_);
if (v_isSharedCheck_1222_ == 0)
{
v___x_1205_ = v___x_1194_;
v_isShared_1206_ = v_isSharedCheck_1222_;
goto v_resetjp_1204_;
}
else
{
lean_inc(v_a_1203_);
lean_dec(v___x_1194_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1222_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
lean_object* v_fst_1207_; lean_object* v_snd_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1221_; 
v_fst_1207_ = lean_ctor_get(v_a_1203_, 0);
v_snd_1208_ = lean_ctor_get(v_a_1203_, 1);
v_isSharedCheck_1221_ = !lean_is_exclusive(v_a_1203_);
if (v_isSharedCheck_1221_ == 0)
{
v___x_1210_ = v_a_1203_;
v_isShared_1211_ = v_isSharedCheck_1221_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_snd_1208_);
lean_inc(v_fst_1207_);
lean_dec(v_a_1203_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1221_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1213_; 
if (v_isShared_1211_ == 0)
{
lean_ctor_set(v___x_1210_, 1, v_fst_1207_);
lean_ctor_set(v___x_1210_, 0, v_fst_1189_);
v___x_1213_ = v___x_1210_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1220_; 
v_reuseFailAlloc_1220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1220_, 0, v_fst_1189_);
lean_ctor_set(v_reuseFailAlloc_1220_, 1, v_fst_1207_);
v___x_1213_ = v_reuseFailAlloc_1220_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
lean_object* v___x_1215_; 
if (v_isShared_1193_ == 0)
{
lean_ctor_set(v___x_1192_, 1, v_snd_1208_);
lean_ctor_set(v___x_1192_, 0, v___x_1213_);
v___x_1215_ = v___x_1192_;
goto v_reusejp_1214_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v___x_1213_);
lean_ctor_set(v_reuseFailAlloc_1219_, 1, v_snd_1208_);
v___x_1215_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1214_;
}
v_reusejp_1214_:
{
lean_object* v___x_1217_; 
if (v_isShared_1206_ == 0)
{
lean_ctor_set(v___x_1205_, 0, v___x_1215_);
v___x_1217_ = v___x_1205_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v___x_1215_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___redArg___boxed(lean_object* v_x_1224_, lean_object* v_y_1225_, lean_object* v_a_1226_, lean_object* v_a_1227_){
_start:
{
lean_object* v_res_1228_; 
v_res_1228_ = lp_plausible_Plausible_Gen_prodOf___redArg(v_x_1224_, v_y_1225_, v_a_1226_, v_a_1227_);
lean_dec(v_a_1227_);
return v_res_1228_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf(lean_object* v_00_u03b1_1229_, lean_object* v_00_u03b2_1230_, lean_object* v_x_1231_, lean_object* v_y_1232_, lean_object* v_a_1233_, lean_object* v_a_1234_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = lp_plausible_Plausible_Gen_prodOf___redArg(v_x_1231_, v_y_1232_, v_a_1233_, v_a_1234_);
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_prodOf___boxed(lean_object* v_00_u03b1_1236_, lean_object* v_00_u03b2_1237_, lean_object* v_x_1238_, lean_object* v_y_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_plausible_Plausible_Gen_prodOf(v_00_u03b1_1236_, v_00_u03b2_1237_, v_x_1238_, v_y_1239_, v_a_1240_, v_a_1241_);
lean_dec(v_a_1241_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(lean_object* v_m_1244_){
_start:
{
if (lean_obj_tag(v_m_1244_) == 0)
{
lean_object* v_a_1246_; lean_object* v___x_1248_; uint8_t v_isShared_1249_; uint8_t v_isSharedCheck_1256_; 
v_a_1246_ = lean_ctor_get(v_m_1244_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v_m_1244_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1248_ = v_m_1244_;
v_isShared_1249_ = v_isSharedCheck_1256_;
goto v_resetjp_1247_;
}
else
{
lean_inc(v_a_1246_);
lean_dec(v_m_1244_);
v___x_1248_ = lean_box(0);
v_isShared_1249_ = v_isSharedCheck_1256_;
goto v_resetjp_1247_;
}
v_resetjp_1247_:
{
lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1253_; 
v___x_1250_ = ((lean_object*)(lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___closed__0));
v___x_1251_ = lean_string_append(v___x_1250_, v_a_1246_);
lean_dec(v_a_1246_);
if (v_isShared_1249_ == 0)
{
lean_ctor_set_tag(v___x_1248_, 18);
lean_ctor_set(v___x_1248_, 0, v___x_1251_);
v___x_1253_ = v___x_1248_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(18, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v___x_1251_);
v___x_1253_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
lean_object* v___x_1254_; 
v___x_1254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1254_, 0, v___x_1253_);
return v___x_1254_;
}
}
}
else
{
lean_object* v_a_1257_; lean_object* v___x_1259_; uint8_t v_isShared_1260_; uint8_t v_isSharedCheck_1264_; 
v_a_1257_ = lean_ctor_get(v_m_1244_, 0);
v_isSharedCheck_1264_ = !lean_is_exclusive(v_m_1244_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1259_ = v_m_1244_;
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
else
{
lean_inc(v_a_1257_);
lean_dec(v_m_1244_);
v___x_1259_ = lean_box(0);
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
v_resetjp_1258_:
{
lean_object* v___x_1262_; 
if (v_isShared_1260_ == 0)
{
lean_ctor_set_tag(v___x_1259_, 0);
v___x_1262_ = v___x_1259_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v_a_1257_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg___boxed(lean_object* v_m_1265_, lean_object* v_a_1266_){
_start:
{
lean_object* v_res_1267_; 
v_res_1267_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v_m_1265_);
return v_res_1267_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError(lean_object* v_00_u03b1_1268_, lean_object* v_m_1269_){
_start:
{
lean_object* v___x_1271_; 
v___x_1271_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v_m_1269_);
return v___x_1271_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___boxed(lean_object* v_00_u03b1_1272_, lean_object* v_m_1273_, lean_object* v_a_1274_){
_start:
{
lean_object* v_res_1275_; 
v_res_1275_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError(v_00_u03b1_1272_, v_m_1273_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___redArg(lean_object* v_m_1276_){
_start:
{
lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; 
v___x_1278_ = lean_unsigned_to_nat(0u);
v___x_1279_ = lean_apply_1(v_m_1276_, v___x_1278_);
v___x_1280_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v___x_1279_);
return v___x_1280_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___redArg___boxed(lean_object* v_m_1281_, lean_object* v_a_1282_){
_start:
{
lean_object* v_res_1283_; 
v_res_1283_ = lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___redArg(v_m_1281_);
return v_res_1283_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1(lean_object* v_00_u03b1_1284_, lean_object* v_m_1285_){
_start:
{
lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; 
v___x_1287_ = lean_unsigned_to_nat(0u);
v___x_1288_ = lean_apply_1(v_m_1285_, v___x_1287_);
v___x_1289_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v___x_1288_);
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___private__1___boxed(lean_object* v_00_u03b1_1290_, lean_object* v_m_1291_, lean_object* v_a_1292_){
_start:
{
lean_object* v_res_1293_; 
v_res_1293_ = lp_plausible_Plausible_instMonadLiftStateIOGen___private__1(v_00_u03b1_1290_, v_m_1291_);
return v_res_1293_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0(lean_object* v_00_u03b1_1294_, lean_object* v_m_1295_){
_start:
{
lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; 
v___x_1297_ = lean_unsigned_to_nat(0u);
v___x_1298_ = lean_apply_1(v_m_1295_, v___x_1297_);
v___x_1299_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v___x_1298_);
return v___x_1299_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0___boxed(lean_object* v_00_u03b1_1300_, lean_object* v_m_1301_, lean_object* v___y_1302_){
_start:
{
lean_object* v_res_1303_; 
v_res_1303_ = lp_plausible_Plausible_instMonadLiftStateIOGen___lam__0(v_00_u03b1_1300_, v_m_1301_);
return v_res_1303_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg(lean_object* v_cmd_1306_){
_start:
{
lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
v___x_1308_ = l_IO_stdGenRef;
v___x_1309_ = lean_st_ref_get(v___x_1308_);
v___x_1310_ = lean_apply_2(v_cmd_1306_, v___x_1309_, lean_box(0));
if (lean_obj_tag(v___x_1310_) == 0)
{
lean_object* v_a_1311_; lean_object* v___x_1313_; uint8_t v_isShared_1314_; uint8_t v_isSharedCheck_1321_; 
v_a_1311_ = lean_ctor_get(v___x_1310_, 0);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1310_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1313_ = v___x_1310_;
v_isShared_1314_ = v_isSharedCheck_1321_;
goto v_resetjp_1312_;
}
else
{
lean_inc(v_a_1311_);
lean_dec(v___x_1310_);
v___x_1313_ = lean_box(0);
v_isShared_1314_ = v_isSharedCheck_1321_;
goto v_resetjp_1312_;
}
v_resetjp_1312_:
{
lean_object* v_fst_1315_; lean_object* v_snd_1316_; lean_object* v___x_1317_; lean_object* v___x_1319_; 
v_fst_1315_ = lean_ctor_get(v_a_1311_, 0);
lean_inc(v_fst_1315_);
v_snd_1316_ = lean_ctor_get(v_a_1311_, 1);
lean_inc(v_snd_1316_);
lean_dec(v_a_1311_);
v___x_1317_ = lean_st_ref_set(v___x_1308_, v_snd_1316_);
if (v_isShared_1314_ == 0)
{
lean_ctor_set(v___x_1313_, 0, v_fst_1315_);
v___x_1319_ = v___x_1313_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v_fst_1315_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
}
else
{
lean_object* v_a_1322_; lean_object* v___x_1324_; uint8_t v_isShared_1325_; uint8_t v_isSharedCheck_1329_; 
v_a_1322_ = lean_ctor_get(v___x_1310_, 0);
v_isSharedCheck_1329_ = !lean_is_exclusive(v___x_1310_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1324_ = v___x_1310_;
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
else
{
lean_inc(v_a_1322_);
lean_dec(v___x_1310_);
v___x_1324_ = lean_box(0);
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
v_resetjp_1323_:
{
lean_object* v___x_1327_; 
if (v_isShared_1325_ == 0)
{
v___x_1327_ = v___x_1324_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1328_; 
v_reuseFailAlloc_1328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1328_, 0, v_a_1322_);
v___x_1327_ = v_reuseFailAlloc_1328_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
return v___x_1327_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg___boxed(lean_object* v_cmd_1330_, lean_object* v___y_1331_){
_start:
{
lean_object* v_res_1332_; 
v_res_1332_ = lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg(v_cmd_1330_);
return v_res_1332_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0(lean_object* v_00_u03b1_1333_, lean_object* v_cmd_1334_){
_start:
{
lean_object* v___x_1336_; 
v___x_1336_ = lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg(v_cmd_1334_);
return v___x_1336_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___boxed(lean_object* v_00_u03b1_1337_, lean_object* v_cmd_1338_, lean_object* v___y_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0(v_00_u03b1_1337_, v_cmd_1338_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___lam__0(lean_object* v_x_1341_, lean_object* v_size_1342_, lean_object* v___y_1343_){
_start:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; 
v___x_1345_ = lean_apply_2(v_x_1341_, v___y_1343_, v_size_1342_);
v___x_1346_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError___redArg(v___x_1345_);
return v___x_1346_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___lam__0___boxed(lean_object* v_x_1347_, lean_object* v_size_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_){
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_plausible_Plausible_Gen_run___redArg___lam__0(v_x_1347_, v_size_1348_, v___y_1349_);
return v_res_1351_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg(lean_object* v_x_1352_, lean_object* v_size_1353_){
_start:
{
lean_object* v___f_1355_; lean_object* v___x_1356_; 
v___f_1355_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_run___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_1355_, 0, v_x_1352_);
lean_closure_set(v___f_1355_, 1, v_size_1353_);
v___x_1356_ = lp_plausible_Plausible_runRand___at___00Plausible_Gen_run_spec__0___redArg(v___f_1355_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___redArg___boxed(lean_object* v_x_1357_, lean_object* v_size_1358_, lean_object* v_a_1359_){
_start:
{
lean_object* v_res_1360_; 
v_res_1360_ = lp_plausible_Plausible_Gen_run___redArg(v_x_1357_, v_size_1358_);
return v_res_1360_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run(lean_object* v_00_u03b1_1361_, lean_object* v_x_1362_, lean_object* v_size_1363_){
_start:
{
lean_object* v___x_1365_; 
v___x_1365_ = lp_plausible_Plausible_Gen_run___redArg(v_x_1362_, v_size_1363_);
return v___x_1365_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_run___boxed(lean_object* v_00_u03b1_1366_, lean_object* v_x_1367_, lean_object* v_size_1368_, lean_object* v_a_1369_){
_start:
{
lean_object* v_res_1370_; 
v_res_1370_ = lp_plausible_Plausible_Gen_run(v_00_u03b1_1366_, v_x_1367_, v_size_1368_);
return v_res_1370_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___lam__0(lean_object* v___x_1371_, lean_object* v___f_1372_, lean_object* v_g_1373_, lean_object* v_inst_1374_, lean_object* v_a_1375_, lean_object* v_x_1376_, lean_object* v___y_1377_){
_start:
{
lean_object* v_a_1383_; lean_object* v___x_1395_; 
v___x_1395_ = lp_plausible_Plausible_Gen_run___redArg(v_g_1373_, v_a_1375_);
if (lean_obj_tag(v___x_1395_) == 0)
{
lean_object* v_a_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; 
v_a_1396_ = lean_ctor_get(v___x_1395_, 0);
lean_inc(v_a_1396_);
lean_dec_ref_known(v___x_1395_, 1);
v___x_1397_ = lean_unsigned_to_nat(0u);
v___x_1398_ = lean_apply_2(v_inst_1374_, v_a_1396_, v___x_1397_);
v___x_1399_ = l_Std_Format_defWidth;
v___x_1400_ = l_Std_Format_pretty(v___x_1398_, v___x_1399_, v___x_1397_, v___x_1397_);
lean_inc_ref(v___f_1372_);
v___x_1401_ = l_IO_println___redArg(v___f_1372_, v___x_1400_);
if (lean_obj_tag(v___x_1401_) == 0)
{
lean_dec_ref_known(v___x_1401_, 1);
lean_dec_ref(v___f_1372_);
goto v___jp_1379_;
}
else
{
lean_object* v_a_1402_; 
v_a_1402_ = lean_ctor_get(v___x_1401_, 0);
lean_inc(v_a_1402_);
lean_dec_ref_known(v___x_1401_, 1);
v_a_1383_ = v_a_1402_;
goto v___jp_1382_;
}
}
else
{
lean_object* v_a_1403_; 
lean_dec_ref(v_inst_1374_);
v_a_1403_ = lean_ctor_get(v___x_1395_, 0);
lean_inc(v_a_1403_);
lean_dec_ref_known(v___x_1395_, 1);
v_a_1383_ = v_a_1403_;
goto v___jp_1382_;
}
v___jp_1379_:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1380_, 0, v___x_1371_);
v___x_1381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1380_);
return v___x_1381_;
}
v___jp_1382_:
{
if (lean_obj_tag(v_a_1383_) == 18)
{
lean_object* v_msg_1384_; lean_object* v___x_1385_; 
v_msg_1384_ = lean_ctor_get(v_a_1383_, 0);
lean_inc_ref(v_msg_1384_);
lean_dec_ref_known(v_a_1383_, 1);
v___x_1385_ = l_IO_println___redArg(v___f_1372_, v_msg_1384_);
if (lean_obj_tag(v___x_1385_) == 0)
{
lean_dec_ref_known(v___x_1385_, 1);
goto v___jp_1379_;
}
else
{
lean_object* v_a_1386_; lean_object* v___x_1388_; uint8_t v_isShared_1389_; uint8_t v_isSharedCheck_1393_; 
v_a_1386_ = lean_ctor_get(v___x_1385_, 0);
v_isSharedCheck_1393_ = !lean_is_exclusive(v___x_1385_);
if (v_isSharedCheck_1393_ == 0)
{
v___x_1388_ = v___x_1385_;
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
else
{
lean_inc(v_a_1386_);
lean_dec(v___x_1385_);
v___x_1388_ = lean_box(0);
v_isShared_1389_ = v_isSharedCheck_1393_;
goto v_resetjp_1387_;
}
v_resetjp_1387_:
{
lean_object* v___x_1391_; 
if (v_isShared_1389_ == 0)
{
v___x_1391_ = v___x_1388_;
goto v_reusejp_1390_;
}
else
{
lean_object* v_reuseFailAlloc_1392_; 
v_reuseFailAlloc_1392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1392_, 0, v_a_1386_);
v___x_1391_ = v_reuseFailAlloc_1392_;
goto v_reusejp_1390_;
}
v_reusejp_1390_:
{
return v___x_1391_;
}
}
}
}
else
{
lean_object* v___x_1394_; 
lean_dec_ref(v___f_1372_);
v___x_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1394_, 0, v_a_1383_);
return v___x_1394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___lam__0___boxed(lean_object* v___x_1404_, lean_object* v___f_1405_, lean_object* v_g_1406_, lean_object* v_inst_1407_, lean_object* v_a_1408_, lean_object* v_x_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_){
_start:
{
lean_object* v_res_1412_; 
v_res_1412_ = lp_plausible_Plausible_Gen_printSamples___redArg___lam__0(v___x_1404_, v___f_1405_, v_g_1406_, v_inst_1407_, v_a_1408_, v_x_1409_, v___y_1410_);
return v_res_1412_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_printSamples___redArg___closed__0(void){
_start:
{
lean_object* v___x_1413_; 
v___x_1413_ = l_instMonadEIO(lean_box(0));
return v___x_1413_;
}
}
static lean_object* _init_lp_plausible_Plausible_Gen_printSamples___redArg___closed__2(void){
_start:
{
lean_object* v___x_1415_; lean_object* v_xs_1416_; 
v___x_1415_ = lean_unsigned_to_nat(10u);
v_xs_1416_ = l_List_range(v___x_1415_);
return v_xs_1416_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg(lean_object* v_inst_1417_, lean_object* v_g_1418_){
_start:
{
lean_object* v___x_1420_; lean_object* v___f_1421_; lean_object* v_xs_1422_; lean_object* v___x_1423_; lean_object* v___f_1424_; lean_object* v___x_294__overap_1425_; lean_object* v___x_1426_; 
v___x_1420_ = lean_obj_once(&lp_plausible_Plausible_Gen_printSamples___redArg___closed__0, &lp_plausible_Plausible_Gen_printSamples___redArg___closed__0_once, _init_lp_plausible_Plausible_Gen_printSamples___redArg___closed__0);
v___f_1421_ = ((lean_object*)(lp_plausible_Plausible_Gen_printSamples___redArg___closed__1));
v_xs_1422_ = lean_obj_once(&lp_plausible_Plausible_Gen_printSamples___redArg___closed__2, &lp_plausible_Plausible_Gen_printSamples___redArg___closed__2_once, _init_lp_plausible_Plausible_Gen_printSamples___redArg___closed__2);
v___x_1423_ = lean_box(0);
v___f_1424_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_printSamples___redArg___lam__0___boxed), 8, 4);
lean_closure_set(v___f_1424_, 0, v___x_1423_);
lean_closure_set(v___f_1424_, 1, v___f_1421_);
lean_closure_set(v___f_1424_, 2, v_g_1418_);
lean_closure_set(v___f_1424_, 3, v_inst_1417_);
v___x_294__overap_1425_ = l_List_forIn_x27_loop___redArg(v___x_1420_, v___f_1424_, v_xs_1422_, v___x_1423_);
v___x_1426_ = lean_apply_1(v___x_294__overap_1425_, lean_box(0));
if (lean_obj_tag(v___x_1426_) == 0)
{
lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1433_; 
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1426_);
if (v_isSharedCheck_1433_ == 0)
{
lean_object* v_unused_1434_; 
v_unused_1434_ = lean_ctor_get(v___x_1426_, 0);
lean_dec(v_unused_1434_);
v___x_1428_ = v___x_1426_;
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
else
{
lean_dec(v___x_1426_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1431_; 
if (v_isShared_1429_ == 0)
{
lean_ctor_set(v___x_1428_, 0, v___x_1423_);
v___x_1431_ = v___x_1428_;
goto v_reusejp_1430_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v___x_1423_);
v___x_1431_ = v_reuseFailAlloc_1432_;
goto v_reusejp_1430_;
}
v_reusejp_1430_:
{
return v___x_1431_;
}
}
}
else
{
return v___x_1426_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___redArg___boxed(lean_object* v_inst_1435_, lean_object* v_g_1436_, lean_object* v_a_1437_){
_start:
{
lean_object* v_res_1438_; 
v_res_1438_ = lp_plausible_Plausible_Gen_printSamples___redArg(v_inst_1435_, v_g_1436_);
return v_res_1438_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples(lean_object* v_t_1439_, lean_object* v_inst_1440_, lean_object* v_g_1441_){
_start:
{
lean_object* v___x_1443_; 
v___x_1443_ = lp_plausible_Plausible_Gen_printSamples___redArg(v_inst_1440_, v_g_1441_);
return v___x_1443_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_printSamples___boxed(lean_object* v_t_1444_, lean_object* v_inst_1445_, lean_object* v_g_1446_, lean_object* v_a_1447_){
_start:
{
lean_object* v_res_1448_; 
v_res_1448_ = lp_plausible_Plausible_Gen_printSamples(v_t_1444_, v_inst_1445_, v_g_1446_);
return v_res_1448_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_decr(lean_object* v_a_1449_){
_start:
{
if (lean_obj_tag(v_a_1449_) == 0)
{
return v_a_1449_;
}
else
{
lean_object* v_val_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1459_; 
v_val_1450_ = lean_ctor_get(v_a_1449_, 0);
v_isSharedCheck_1459_ = !lean_is_exclusive(v_a_1449_);
if (v_isSharedCheck_1459_ == 0)
{
v___x_1452_ = v_a_1449_;
v_isShared_1453_ = v_isSharedCheck_1459_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_val_1450_);
lean_dec(v_a_1449_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1459_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1457_; 
v___x_1454_ = lean_unsigned_to_nat(1u);
v___x_1455_ = lean_nat_sub(v_val_1450_, v___x_1454_);
lean_dec(v_val_1450_);
if (v_isShared_1453_ == 0)
{
lean_ctor_set(v___x_1452_, 0, v___x_1455_);
v___x_1457_ = v___x_1452_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v___x_1455_);
v___x_1457_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
return v___x_1457_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___redArg(lean_object* v_a_1460_){
_start:
{
lean_object* v___x_1461_; lean_object* v_fst_1462_; lean_object* v_snd_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1471_; 
v___x_1461_ = l_stdNext(v_a_1460_);
v_fst_1462_ = lean_ctor_get(v___x_1461_, 0);
v_snd_1463_ = lean_ctor_get(v___x_1461_, 1);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1461_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1465_ = v___x_1461_;
v_isShared_1466_ = v_isSharedCheck_1471_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_snd_1463_);
lean_inc(v_fst_1462_);
lean_dec(v___x_1461_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1471_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v___x_1468_; 
if (v_isShared_1466_ == 0)
{
v___x_1468_ = v___x_1465_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1470_; 
v_reuseFailAlloc_1470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1470_, 0, v_fst_1462_);
lean_ctor_set(v_reuseFailAlloc_1470_, 1, v_snd_1463_);
v___x_1468_ = v_reuseFailAlloc_1470_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
lean_object* v___x_1469_; 
v___x_1469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1469_, 0, v___x_1468_);
return v___x_1469_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0(lean_object* v_a_1472_, lean_object* v___y_1473_){
_start:
{
lean_object* v___x_1474_; 
v___x_1474_ = lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___redArg(v_a_1472_);
return v___x_1474_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___boxed(lean_object* v_a_1475_, lean_object* v___y_1476_){
_start:
{
lean_object* v_res_1477_; 
v_res_1477_ = lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0(v_a_1475_, v___y_1476_);
lean_dec(v___y_1476_);
return v_res_1477_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg(lean_object* v_attempts_1481_, lean_object* v_x_1482_, lean_object* v_a_1483_, lean_object* v_a_1484_){
_start:
{
lean_object* v___y_1486_; lean_object* v___y_1487_; 
if (lean_obj_tag(v_attempts_1481_) == 1)
{
lean_object* v_val_1502_; lean_object* v___x_1503_; uint8_t v___x_1504_; 
v_val_1502_ = lean_ctor_get(v_attempts_1481_, 0);
v___x_1503_ = lean_unsigned_to_nat(0u);
v___x_1504_ = lean_nat_dec_eq(v_val_1502_, v___x_1503_);
if (v___x_1504_ == 0)
{
v___y_1486_ = v_a_1483_;
v___y_1487_ = v_a_1484_;
goto v___jp_1485_;
}
else
{
lean_object* v___x_1505_; 
lean_dec_ref_known(v_attempts_1481_, 1);
lean_dec_ref(v_a_1483_);
lean_dec_ref(v_x_1482_);
v___x_1505_ = ((lean_object*)(lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___closed__1));
return v___x_1505_;
}
}
else
{
v___y_1486_ = v_a_1483_;
v___y_1487_ = v_a_1484_;
goto v___jp_1485_;
}
v___jp_1485_:
{
lean_object* v___x_1488_; 
lean_inc_ref(v_x_1482_);
lean_inc(v___y_1487_);
lean_inc_ref(v___y_1486_);
v___x_1488_ = lean_apply_2(v_x_1482_, v___y_1486_, v___y_1487_);
if (lean_obj_tag(v___x_1488_) == 0)
{
lean_object* v___x_1489_; 
lean_dec_ref_known(v___x_1488_, 1);
v___x_1489_ = lp_plausible_Plausible_Rand_next___at___00__private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen_spec__0___redArg(v___y_1486_);
if (lean_obj_tag(v___x_1489_) == 0)
{
lean_object* v_a_1490_; lean_object* v___x_1492_; uint8_t v_isShared_1493_; uint8_t v_isSharedCheck_1497_; 
lean_dec_ref(v_x_1482_);
lean_dec(v_attempts_1481_);
v_a_1490_ = lean_ctor_get(v___x_1489_, 0);
v_isSharedCheck_1497_ = !lean_is_exclusive(v___x_1489_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1492_ = v___x_1489_;
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
else
{
lean_inc(v_a_1490_);
lean_dec(v___x_1489_);
v___x_1492_ = lean_box(0);
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
v_resetjp_1491_:
{
lean_object* v___x_1495_; 
if (v_isShared_1493_ == 0)
{
v___x_1495_ = v___x_1492_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_a_1490_);
v___x_1495_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1494_;
}
v_reusejp_1494_:
{
return v___x_1495_;
}
}
}
else
{
lean_object* v_a_1498_; lean_object* v_snd_1499_; lean_object* v___x_1500_; 
v_a_1498_ = lean_ctor_get(v___x_1489_, 0);
lean_inc(v_a_1498_);
lean_dec_ref_known(v___x_1489_, 1);
v_snd_1499_ = lean_ctor_get(v_a_1498_, 1);
lean_inc(v_snd_1499_);
lean_dec(v_a_1498_);
v___x_1500_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_decr(v_attempts_1481_);
v_attempts_1481_ = v___x_1500_;
v_a_1483_ = v_snd_1499_;
v_a_1484_ = v___y_1487_;
goto _start;
}
}
else
{
lean_dec_ref(v___y_1486_);
lean_dec_ref(v_x_1482_);
lean_dec(v_attempts_1481_);
return v___x_1488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg___boxed(lean_object* v_attempts_1506_, lean_object* v_x_1507_, lean_object* v_a_1508_, lean_object* v_a_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg(v_attempts_1506_, v_x_1507_, v_a_1508_, v_a_1509_);
lean_dec(v_a_1509_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen(lean_object* v_00_u03b1_1511_, lean_object* v_attempts_1512_, lean_object* v_x_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_){
_start:
{
lean_object* v___x_1516_; 
v___x_1516_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___redArg(v_attempts_1512_, v_x_1513_, v_a_1514_, v_a_1515_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___boxed(lean_object* v_00_u03b1_1517_, lean_object* v_attempts_1518_, lean_object* v_x_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_){
_start:
{
lean_object* v_res_1522_; 
v_res_1522_ = lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen(v_00_u03b1_1517_, v_attempts_1518_, v_x_1519_, v_a_1520_, v_a_1521_);
lean_dec(v_a_1521_);
return v_res_1522_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___redArg(lean_object* v_attempts_1523_, lean_object* v_x_1524_, lean_object* v_size_1525_){
_start:
{
lean_object* v___x_1527_; lean_object* v___x_1528_; 
v___x_1527_ = lean_alloc_closure((void*)(lp_plausible___private_Plausible_Gen_0__Plausible_Gen_runUntil_repeatGen___boxed), 5, 3);
lean_closure_set(v___x_1527_, 0, lean_box(0));
lean_closure_set(v___x_1527_, 1, v_attempts_1523_);
lean_closure_set(v___x_1527_, 2, v_x_1524_);
v___x_1528_ = lp_plausible_Plausible_Gen_run___redArg(v___x_1527_, v_size_1525_);
return v___x_1528_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___redArg___boxed(lean_object* v_attempts_1529_, lean_object* v_x_1530_, lean_object* v_size_1531_, lean_object* v_a_1532_){
_start:
{
lean_object* v_res_1533_; 
v_res_1533_ = lp_plausible_Plausible_Gen_runUntil___redArg(v_attempts_1529_, v_x_1530_, v_size_1531_);
return v_res_1533_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil(lean_object* v_00_u03b1_1534_, lean_object* v_attempts_1535_, lean_object* v_x_1536_, lean_object* v_size_1537_){
_start:
{
lean_object* v___x_1539_; 
v___x_1539_ = lp_plausible_Plausible_Gen_runUntil___redArg(v_attempts_1535_, v_x_1536_, v_size_1537_);
return v___x_1539_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Gen_runUntil___boxed(lean_object* v_00_u03b1_1540_, lean_object* v_attempts_1541_, lean_object* v_x_1542_, lean_object* v_size_1543_, lean_object* v_a_1544_){
_start:
{
lean_object* v_res_1545_; 
v_res_1545_ = lp_plausible_Plausible_Gen_runUntil(v_00_u03b1_1540_, v_attempts_1541_, v_x_1542_, v_size_1543_);
return v_res_1545_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test(lean_object* v_n_1549_, lean_object* v_a_1550_, lean_object* v_a_1551_){
_start:
{
lean_object* v___x_1552_; lean_object* v_a_1553_; lean_object* v_fst_1554_; lean_object* v_snd_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v_a_1558_; lean_object* v___x_1560_; uint8_t v_isShared_1561_; uint8_t v_isSharedCheck_1577_; 
v___x_1552_ = lp_plausible_Plausible_Gen_getSize(v_a_1550_, v_a_1551_);
v_a_1553_ = lean_ctor_get(v___x_1552_, 0);
lean_inc(v_a_1553_);
lean_dec_ref(v___x_1552_);
v_fst_1554_ = lean_ctor_get(v_a_1553_, 0);
lean_inc(v_fst_1554_);
v_snd_1555_ = lean_ctor_get(v_a_1553_, 1);
lean_inc(v_snd_1555_);
lean_dec(v_a_1553_);
v___x_1556_ = lean_unsigned_to_nat(0u);
v___x_1557_ = lp_plausible_Plausible_Gen_choose___at___00Plausible_Gen_chooseNatLt_spec__0___redArg(v___x_1556_, v_fst_1554_, v_snd_1555_);
lean_dec(v_fst_1554_);
v_a_1558_ = lean_ctor_get(v___x_1557_, 0);
v_isSharedCheck_1577_ = !lean_is_exclusive(v___x_1557_);
if (v_isSharedCheck_1577_ == 0)
{
v___x_1560_ = v___x_1557_;
v_isShared_1561_ = v_isSharedCheck_1577_;
goto v_resetjp_1559_;
}
else
{
lean_inc(v_a_1558_);
lean_dec(v___x_1557_);
v___x_1560_ = lean_box(0);
v_isShared_1561_ = v_isSharedCheck_1577_;
goto v_resetjp_1559_;
}
v_resetjp_1559_:
{
lean_object* v_fst_1562_; lean_object* v_snd_1563_; lean_object* v___x_1565_; uint8_t v_isShared_1566_; uint8_t v_isSharedCheck_1576_; 
v_fst_1562_ = lean_ctor_get(v_a_1558_, 0);
v_snd_1563_ = lean_ctor_get(v_a_1558_, 1);
v_isSharedCheck_1576_ = !lean_is_exclusive(v_a_1558_);
if (v_isSharedCheck_1576_ == 0)
{
v___x_1565_ = v_a_1558_;
v_isShared_1566_ = v_isSharedCheck_1576_;
goto v_resetjp_1564_;
}
else
{
lean_inc(v_snd_1563_);
lean_inc(v_fst_1562_);
lean_dec(v_a_1558_);
v___x_1565_ = lean_box(0);
v_isShared_1566_ = v_isSharedCheck_1576_;
goto v_resetjp_1564_;
}
v_resetjp_1564_:
{
lean_object* v___x_1567_; uint8_t v___x_1568_; 
v___x_1567_ = lean_nat_mod(v_fst_1562_, v_n_1549_);
v___x_1568_ = lean_nat_dec_eq(v___x_1567_, v___x_1556_);
lean_dec(v___x_1567_);
if (v___x_1568_ == 0)
{
lean_object* v___x_1569_; 
lean_del_object(v___x_1565_);
lean_dec(v_snd_1563_);
lean_dec(v_fst_1562_);
lean_del_object(v___x_1560_);
v___x_1569_ = ((lean_object*)(lp_plausible___private_Plausible_Gen_0__Plausible_test___closed__1));
return v___x_1569_;
}
else
{
lean_object* v___x_1571_; 
if (v_isShared_1566_ == 0)
{
v___x_1571_ = v___x_1565_;
goto v_reusejp_1570_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v_fst_1562_);
lean_ctor_set(v_reuseFailAlloc_1575_, 1, v_snd_1563_);
v___x_1571_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1570_;
}
v_reusejp_1570_:
{
lean_object* v___x_1573_; 
if (v_isShared_1561_ == 0)
{
lean_ctor_set(v___x_1560_, 0, v___x_1571_);
v___x_1573_ = v___x_1560_;
goto v_reusejp_1572_;
}
else
{
lean_object* v_reuseFailAlloc_1574_; 
v_reuseFailAlloc_1574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1574_, 0, v___x_1571_);
v___x_1573_ = v_reuseFailAlloc_1574_;
goto v_reusejp_1572_;
}
v_reusejp_1572_:
{
return v___x_1573_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_test___boxed(lean_object* v_n_1578_, lean_object* v_a_1579_, lean_object* v_a_1580_){
_start:
{
lean_object* v_res_1581_; 
v_res_1581_ = lp_plausible___private_Plausible_Gen_0__Plausible_test(v_n_1578_, v_a_1579_, v_a_1580_);
lean_dec(v_a_1580_);
lean_dec(v_n_1578_);
return v_res_1581_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Random(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Gen(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Random(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_plausible_Plausible_instMonadErrorGen = _init_lp_plausible_Plausible_instMonadErrorGen();
lean_mark_persistent(lp_plausible_Plausible_instMonadErrorGen);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Gen(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_plausible_Plausible_Gen_oneOf___auto__1 = _init_lp_plausible_Plausible_Gen_oneOf___auto__1();
lean_mark_persistent(lp_plausible_Plausible_Gen_oneOf___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Random(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Gen(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Random(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Gen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Gen(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Gen(builtin);
}
#ifdef __cplusplus
}
#endif
