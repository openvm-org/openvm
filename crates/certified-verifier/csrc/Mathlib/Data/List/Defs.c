// Lean compiler output
// Module: Mathlib.Data.List.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Notation public import Mathlib.Control.Functor public import Mathlib.Data.SProd public import Batteries.Tactic.Lint.Basic public import Batteries.Data.List.Basic public import Batteries.Logic
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_foldlIdx___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_List_mapM_x27___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_firstM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_List_allM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_anyM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* lp_batteries_List_pwFilter___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_List_productTR(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_diff(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_batteries_List_toChunks___redArg(lean_object*, lean_object*);
lean_object* lean_task_spawn(lean_object*, lean_object*);
lean_object* lean_task_get_own(lean_object*);
lean_object* lp_batteries_List_takeDTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instSDiffOfDecidableEq__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instSDiffOfDecidableEq__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getI___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getI___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_headI___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_headI___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_headI(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_headI___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_takeI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_takeI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_orM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_List_orM___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_orM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_orM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_orM(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_andM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_andM(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxM_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_Forall_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_Forall_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux_rec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_permutationsAux_rec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_permutationsAux_rec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_permutationsAux___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_permutationsAux___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_permutationsAux___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_permutationsAux___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_List_permutationsAux___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_permutationsAux___redArg___lam__1, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_permutationsAux___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_permutationsAux___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutations___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutations(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27Aux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27Aux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_headI_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_headI_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_permutations_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_permutations_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_permutations_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_extractp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_extractp(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_instSProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_List_productTR, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_List_instSProd___closed__0 = (const lean_object*)&lp_mathlib_List_instSProd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instSProd(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_dedup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dedup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_destutter_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_destutter_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_destutter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_destutter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_map_u2082Left_x27_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_map_u2082Left_x27_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2___redArg(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_mapAsyncChunked___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_mapAsyncChunked___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_mapAsyncChunked___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterate(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_consecutivePairs___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_consecutivePairs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instSDiffOfDecidableEq__mathlib___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___f_2_; lean_object* v___x_3_; 
v___f_2_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2_, 0, v_inst_1_);
v___x_3_ = lean_alloc_closure((void*)(lp_batteries_List_diff), 4, 2);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v___f_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instSDiffOfDecidableEq__mathlib(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_List_instSDiffOfDecidableEq__mathlib___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getI___redArg(lean_object* v_inst_7_, lean_object* v_l_8_, lean_object* v_n_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = l_List_getD___redArg(v_l_8_, v_n_9_, v_inst_7_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getI___redArg___boxed(lean_object* v_inst_11_, lean_object* v_l_12_, lean_object* v_n_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_List_getI___redArg(v_inst_11_, v_l_12_, v_n_13_);
lean_dec(v_l_12_);
lean_dec(v_inst_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getI(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_l_17_, lean_object* v_n_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = l_List_getD___redArg(v_l_17_, v_n_18_, v_inst_16_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getI___boxed(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_, lean_object* v_l_22_, lean_object* v_n_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_List_getI(v_00_u03b1_20_, v_inst_21_, v_l_22_, v_n_23_);
lean_dec(v_l_22_);
lean_dec(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_headI___redArg(lean_object* v_inst_25_, lean_object* v_x_26_){
_start:
{
if (lean_obj_tag(v_x_26_) == 0)
{
lean_inc(v_inst_25_);
return v_inst_25_;
}
else
{
lean_object* v_head_27_; 
v_head_27_ = lean_ctor_get(v_x_26_, 0);
lean_inc(v_head_27_);
return v_head_27_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_headI___redArg___boxed(lean_object* v_inst_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_List_headI___redArg(v_inst_28_, v_x_29_);
lean_dec(v_x_29_);
lean_dec(v_inst_28_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_headI(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_x_33_){
_start:
{
if (lean_obj_tag(v_x_33_) == 0)
{
lean_inc(v_inst_32_);
return v_inst_32_;
}
else
{
lean_object* v_head_34_; 
v_head_34_ = lean_ctor_get(v_x_33_, 0);
lean_inc(v_head_34_);
return v_head_34_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_headI___boxed(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_x_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_List_headI(v_00_u03b1_35_, v_inst_36_, v_x_37_);
lean_dec(v_x_37_);
lean_dec(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___redArg(lean_object* v_inst_39_, lean_object* v_x_40_){
_start:
{
if (lean_obj_tag(v_x_40_) == 0)
{
lean_inc(v_inst_39_);
return v_inst_39_;
}
else
{
lean_object* v_tail_41_; 
v_tail_41_ = lean_ctor_get(v_x_40_, 1);
if (lean_obj_tag(v_tail_41_) == 0)
{
lean_object* v_head_42_; 
v_head_42_ = lean_ctor_get(v_x_40_, 0);
lean_inc(v_head_42_);
return v_head_42_;
}
else
{
lean_object* v_tail_43_; 
v_tail_43_ = lean_ctor_get(v_tail_41_, 1);
if (lean_obj_tag(v_tail_43_) == 0)
{
lean_object* v_head_44_; 
v_head_44_ = lean_ctor_get(v_tail_41_, 0);
lean_inc(v_head_44_);
return v_head_44_;
}
else
{
v_x_40_ = v_tail_43_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___redArg___boxed(lean_object* v_inst_46_, lean_object* v_x_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_List_getLastI___redArg(v_inst_46_, v_x_47_);
lean_dec(v_x_47_);
lean_dec(v_inst_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI(lean_object* v_00_u03b1_49_, lean_object* v_inst_50_, lean_object* v_x_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_List_getLastI___redArg(v_inst_50_, v_x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getLastI___boxed(lean_object* v_00_u03b1_53_, lean_object* v_inst_54_, lean_object* v_x_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_List_getLastI(v_00_u03b1_53_, v_inst_54_, v_x_55_);
lean_dec(v_x_55_);
lean_dec(v_inst_54_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_takeI___redArg(lean_object* v_inst_57_, lean_object* v_n_58_, lean_object* v_l_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_batteries_List_takeDTR___redArg(v_n_58_, v_l_59_, v_inst_57_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_takeI(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_n_63_, lean_object* v_l_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_batteries_List_takeDTR___redArg(v_n_63_, v_l_64_, v_inst_62_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM___redArg___lam__0(lean_object* v_toFunctor_66_, lean_object* v_tac_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_mapConst_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v_mapConst_69_ = lean_ctor_get(v_toFunctor_66_, 1);
lean_inc(v_mapConst_69_);
lean_dec_ref(v_toFunctor_66_);
lean_inc(v_a_68_);
v___x_70_ = lean_apply_1(v_tac_67_, v_a_68_);
v___x_71_ = lean_apply_4(v_mapConst_69_, lean_box(0), lean_box(0), v_a_68_, v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM___redArg(lean_object* v_inst_72_, lean_object* v_tac_73_, lean_object* v_a_74_){
_start:
{
lean_object* v_toApplicative_75_; lean_object* v_toFunctor_76_; lean_object* v___f_77_; lean_object* v___x_78_; 
v_toApplicative_75_ = lean_ctor_get(v_inst_72_, 0);
v_toFunctor_76_ = lean_ctor_get(v_toApplicative_75_, 0);
lean_inc_ref(v_toFunctor_76_);
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_List_findM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_77_, 0, v_toFunctor_76_);
lean_closure_set(v___f_77_, 1, v_tac_73_);
v___x_78_ = l_List_firstM___redArg(v_inst_72_, v___f_77_, v_a_74_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM(lean_object* v_00_u03b1_79_, lean_object* v_m_80_, lean_object* v_inst_81_, lean_object* v_tac_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_List_findM___redArg(v_inst_81_, v_tac_82_, v_a_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg___lam__0___boxed(lean_object* v_inst_85_, lean_object* v_p_86_, lean_object* v_tail_87_, lean_object* v_head_88_, lean_object* v_toPure_89_, lean_object* v_____x_90_){
_start:
{
uint8_t v_____x_109__boxed_91_; lean_object* v_res_92_; 
v_____x_109__boxed_91_ = lean_unbox(v_____x_90_);
v_res_92_ = lp_mathlib_List_findM_x3f_x27___redArg___lam__0(v_inst_85_, v_p_86_, v_tail_87_, v_head_88_, v_toPure_89_, v_____x_109__boxed_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg(lean_object* v_inst_93_, lean_object* v_p_94_, lean_object* v_x_95_){
_start:
{
if (lean_obj_tag(v_x_95_) == 0)
{
lean_object* v_toApplicative_96_; lean_object* v_toPure_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_toApplicative_96_ = lean_ctor_get(v_inst_93_, 0);
lean_inc_ref(v_toApplicative_96_);
lean_dec(v_p_94_);
lean_dec_ref(v_inst_93_);
v_toPure_97_ = lean_ctor_get(v_toApplicative_96_, 1);
lean_inc(v_toPure_97_);
lean_dec_ref(v_toApplicative_96_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_apply_2(v_toPure_97_, lean_box(0), v___x_98_);
return v___x_99_;
}
else
{
lean_object* v_toApplicative_100_; lean_object* v_toBind_101_; lean_object* v_toPure_102_; lean_object* v_head_103_; lean_object* v_tail_104_; lean_object* v___f_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_toApplicative_100_ = lean_ctor_get(v_inst_93_, 0);
v_toBind_101_ = lean_ctor_get(v_inst_93_, 1);
lean_inc(v_toBind_101_);
v_toPure_102_ = lean_ctor_get(v_toApplicative_100_, 1);
lean_inc(v_toPure_102_);
v_head_103_ = lean_ctor_get(v_x_95_, 0);
lean_inc_n(v_head_103_, 2);
v_tail_104_ = lean_ctor_get(v_x_95_, 1);
lean_inc(v_tail_104_);
lean_dec_ref_known(v_x_95_, 2);
lean_inc(v_p_94_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_List_findM_x3f_x27___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_105_, 0, v_inst_93_);
lean_closure_set(v___f_105_, 1, v_p_94_);
lean_closure_set(v___f_105_, 2, v_tail_104_);
lean_closure_set(v___f_105_, 3, v_head_103_);
lean_closure_set(v___f_105_, 4, v_toPure_102_);
v___x_106_ = lean_apply_1(v_p_94_, v_head_103_);
v___x_107_ = lean_apply_4(v_toBind_101_, lean_box(0), lean_box(0), v___x_106_, v___f_105_);
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27___redArg___lam__0(lean_object* v_inst_108_, lean_object* v_p_109_, lean_object* v_tail_110_, lean_object* v_head_111_, lean_object* v_toPure_112_, uint8_t v_____x_113_){
_start:
{
if (v_____x_113_ == 0)
{
lean_object* v___x_114_; 
lean_dec(v_toPure_112_);
lean_dec(v_head_111_);
v___x_114_ = lp_mathlib_List_findM_x3f_x27___redArg(v_inst_108_, v_p_109_, v_tail_110_);
return v___x_114_;
}
else
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v_tail_110_);
lean_dec(v_p_109_);
lean_dec_ref(v_inst_108_);
v___x_115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_115_, 0, v_head_111_);
v___x_116_ = lean_apply_2(v_toPure_112_, lean_box(0), v___x_115_);
return v___x_116_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f_x27(lean_object* v_m_117_, lean_object* v_inst_118_, lean_object* v_00_u03b1_119_, lean_object* v_p_120_, lean_object* v_x_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_List_findM_x3f_x27___redArg(v_inst_118_, v_p_120_, v_x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_orM___redArg(lean_object* v_inst_124_, lean_object* v_l_125_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib_List_orM___redArg___closed__0));
v___x_127_ = l_List_anyM___redArg(v_inst_124_, v___x_126_, v_l_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_orM(lean_object* v_m_128_, lean_object* v_inst_129_, lean_object* v_l_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_List_orM___redArg(v_inst_129_, v_l_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_andM___redArg(lean_object* v_inst_132_, lean_object* v_l_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib_List_orM___redArg___closed__0));
v___x_135_ = l_List_allM___redArg(v_inst_132_, v___x_134_, v_l_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_andM(lean_object* v_m_136_, lean_object* v_inst_137_, lean_object* v_l_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_List_andM___redArg(v_inst_137_, v_l_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg___lam__0(lean_object* v_f_140_, lean_object* v_i_141_, lean_object* v_b_142_, lean_object* v_a_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lean_apply_3(v_f_140_, v_i_141_, v_a_143_, v_b_142_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg___lam__1(lean_object* v_f_145_, lean_object* v_toBind_146_, lean_object* v_i_147_, lean_object* v_ma_148_, lean_object* v_b_149_){
_start:
{
lean_object* v___f_150_; lean_object* v___x_151_; 
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib_List_foldlIdxM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_150_, 0, v_f_145_);
lean_closure_set(v___f_150_, 1, v_i_147_);
lean_closure_set(v___f_150_, 2, v_b_149_);
v___x_151_ = lean_apply_4(v_toBind_146_, lean_box(0), lean_box(0), v_ma_148_, v___f_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM___redArg(lean_object* v_inst_152_, lean_object* v_f_153_, lean_object* v_b_154_, lean_object* v_as_155_){
_start:
{
lean_object* v_toApplicative_156_; lean_object* v_toBind_157_; lean_object* v_toPure_158_; lean_object* v___f_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
v_toApplicative_156_ = lean_ctor_get(v_inst_152_, 0);
lean_inc_ref(v_toApplicative_156_);
v_toBind_157_ = lean_ctor_get(v_inst_152_, 1);
lean_inc(v_toBind_157_);
lean_dec_ref(v_inst_152_);
v_toPure_158_ = lean_ctor_get(v_toApplicative_156_, 1);
lean_inc(v_toPure_158_);
lean_dec_ref(v_toApplicative_156_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_List_foldlIdxM___redArg___lam__1), 5, 2);
lean_closure_set(v___f_159_, 0, v_f_153_);
lean_closure_set(v___f_159_, 1, v_toBind_157_);
v___x_160_ = lean_apply_2(v_toPure_158_, lean_box(0), v_b_154_);
v___x_161_ = lean_unsigned_to_nat(0u);
v___x_162_ = lp_batteries_List_foldlIdx___redArg(v___f_159_, v___x_160_, v_as_155_, v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlIdxM(lean_object* v_m_163_, lean_object* v_inst_164_, lean_object* v_00_u03b1_165_, lean_object* v_00_u03b2_166_, lean_object* v_f_167_, lean_object* v_b_168_, lean_object* v_as_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_List_foldlIdxM___redArg(v_inst_164_, v_f_167_, v_b_168_, v_as_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg___lam__0(lean_object* v_f_171_, lean_object* v___x_172_, lean_object* v_a_173_, lean_object* v_b_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lean_apply_3(v_f_171_, v___x_172_, v_a_173_, v_b_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg___lam__1(lean_object* v_f_176_, lean_object* v_toBind_177_, lean_object* v_a_178_, lean_object* v_x_179_){
_start:
{
lean_object* v_fst_180_; lean_object* v_snd_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_192_; 
v_fst_180_ = lean_ctor_get(v_x_179_, 0);
v_snd_181_ = lean_ctor_get(v_x_179_, 1);
v_isSharedCheck_192_ = !lean_is_exclusive(v_x_179_);
if (v_isSharedCheck_192_ == 0)
{
v___x_183_ = v_x_179_;
v_isShared_184_ = v_isSharedCheck_192_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_snd_181_);
lean_inc(v_fst_180_);
lean_dec(v_x_179_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_192_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___f_187_; lean_object* v___x_188_; lean_object* v___x_190_; 
v___x_185_ = lean_unsigned_to_nat(1u);
v___x_186_ = lean_nat_sub(v_snd_181_, v___x_185_);
lean_dec(v_snd_181_);
lean_inc(v___x_186_);
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_List_foldrIdxM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_187_, 0, v_f_176_);
lean_closure_set(v___f_187_, 1, v___x_186_);
lean_closure_set(v___f_187_, 2, v_a_178_);
v___x_188_ = lean_apply_4(v_toBind_177_, lean_box(0), lean_box(0), v_fst_180_, v___f_187_);
if (v_isShared_184_ == 0)
{
lean_ctor_set(v___x_183_, 1, v___x_186_);
lean_ctor_set(v___x_183_, 0, v___x_188_);
v___x_190_ = v___x_183_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v___x_186_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM___redArg(lean_object* v_inst_193_, lean_object* v_f_194_, lean_object* v_b_195_, lean_object* v_as_196_){
_start:
{
lean_object* v_toApplicative_197_; lean_object* v_toBind_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_211_; 
v_toApplicative_197_ = lean_ctor_get(v_inst_193_, 0);
v_toBind_198_ = lean_ctor_get(v_inst_193_, 1);
v_isSharedCheck_211_ = !lean_is_exclusive(v_inst_193_);
if (v_isSharedCheck_211_ == 0)
{
v___x_200_ = v_inst_193_;
v_isShared_201_ = v_isSharedCheck_211_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_toBind_198_);
lean_inc(v_toApplicative_197_);
lean_dec(v_inst_193_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_211_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v_toPure_202_; lean_object* v___f_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_207_; 
v_toPure_202_ = lean_ctor_get(v_toApplicative_197_, 1);
lean_inc(v_toPure_202_);
lean_dec_ref(v_toApplicative_197_);
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_List_foldrIdxM___redArg___lam__1), 4, 2);
lean_closure_set(v___f_203_, 0, v_f_194_);
lean_closure_set(v___f_203_, 1, v_toBind_198_);
v___x_204_ = lean_apply_2(v_toPure_202_, lean_box(0), v_b_195_);
v___x_205_ = l_List_lengthTR___redArg(v_as_196_);
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 1, v___x_205_);
lean_ctor_set(v___x_200_, 0, v___x_204_);
v___x_207_ = v___x_200_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_204_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v___x_205_);
v___x_207_ = v_reuseFailAlloc_210_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
lean_object* v___x_208_; lean_object* v_fst_209_; 
v___x_208_ = l_List_foldrTR___redArg(v___f_203_, v___x_207_, v_as_196_);
v_fst_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc(v_fst_209_);
lean_dec(v___x_208_);
return v_fst_209_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrIdxM(lean_object* v_m_212_, lean_object* v_inst_213_, lean_object* v_00_u03b1_214_, lean_object* v_00_u03b2_215_, lean_object* v_f_216_, lean_object* v_b_217_, lean_object* v_as_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_List_foldrIdxM___redArg(v_inst_213_, v_f_216_, v_b_217_, v_as_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0___boxed(lean_object* v_x_220_, lean_object* v_inst_221_, lean_object* v_f_222_, lean_object* v_tail_223_, lean_object* v_x_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0(v_x_220_, v_inst_221_, v_f_222_, v_tail_223_, v_x_224_);
lean_dec(v_x_220_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg(lean_object* v_inst_226_, lean_object* v_f_227_, lean_object* v_x_228_, lean_object* v_x_229_){
_start:
{
if (lean_obj_tag(v_x_229_) == 0)
{
lean_object* v_toApplicative_230_; lean_object* v_toPure_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v_toApplicative_230_ = lean_ctor_get(v_inst_226_, 0);
lean_inc_ref(v_toApplicative_230_);
lean_dec(v_x_228_);
lean_dec(v_f_227_);
lean_dec_ref(v_inst_226_);
v_toPure_231_ = lean_ctor_get(v_toApplicative_230_, 1);
lean_inc(v_toPure_231_);
lean_dec_ref(v_toApplicative_230_);
v___x_232_ = lean_box(0);
v___x_233_ = lean_apply_2(v_toPure_231_, lean_box(0), v___x_232_);
return v___x_233_;
}
else
{
lean_object* v_toApplicative_234_; lean_object* v_toSeqRight_235_; lean_object* v_head_236_; lean_object* v_tail_237_; lean_object* v___f_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_toApplicative_234_ = lean_ctor_get(v_inst_226_, 0);
v_toSeqRight_235_ = lean_ctor_get(v_toApplicative_234_, 4);
lean_inc(v_toSeqRight_235_);
v_head_236_ = lean_ctor_get(v_x_229_, 0);
lean_inc(v_head_236_);
v_tail_237_ = lean_ctor_get(v_x_229_, 1);
lean_inc(v_tail_237_);
lean_dec_ref_known(v_x_229_, 2);
lean_inc(v_f_227_);
lean_inc(v_x_228_);
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_238_, 0, v_x_228_);
lean_closure_set(v___f_238_, 1, v_inst_226_);
lean_closure_set(v___f_238_, 2, v_f_227_);
lean_closure_set(v___f_238_, 3, v_tail_237_);
v___x_239_ = lean_apply_2(v_f_227_, v_x_228_, v_head_236_);
v___x_240_ = lean_apply_4(v_toSeqRight_235_, lean_box(0), lean_box(0), v___x_239_, v___f_238_);
return v___x_240_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27___redArg___lam__0(lean_object* v_x_241_, lean_object* v_inst_242_, lean_object* v_f_243_, lean_object* v_tail_244_, lean_object* v_x_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_246_ = lean_unsigned_to_nat(1u);
v___x_247_ = lean_nat_add(v_x_241_, v___x_246_);
v___x_248_ = lp_mathlib_List_mapIdxMAux_x27___redArg(v_inst_242_, v_f_243_, v___x_247_, v_tail_244_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxMAux_x27(lean_object* v_m_249_, lean_object* v_inst_250_, lean_object* v_00_u03b1_251_, lean_object* v_f_252_, lean_object* v_x_253_, lean_object* v_x_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_List_mapIdxMAux_x27___redArg(v_inst_250_, v_f_252_, v_x_253_, v_x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxM_x27___redArg(lean_object* v_inst_256_, lean_object* v_f_257_, lean_object* v_as_258_){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_259_ = lean_unsigned_to_nat(0u);
v___x_260_ = lp_mathlib_List_mapIdxMAux_x27___redArg(v_inst_256_, v_f_257_, v___x_259_, v_as_258_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapIdxM_x27(lean_object* v_m_261_, lean_object* v_inst_262_, lean_object* v_00_u03b1_263_, lean_object* v_f_264_, lean_object* v_as_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_mathlib_List_mapIdxM_x27___redArg(v_inst_262_, v_f_264_, v_as_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_Forall_match__1_splitter___redArg(lean_object* v_x_267_, lean_object* v_h__1_268_, lean_object* v_h__2_269_, lean_object* v_h__3_270_){
_start:
{
if (lean_obj_tag(v_x_267_) == 0)
{
lean_object* v___x_271_; lean_object* v___x_272_; 
lean_dec(v_h__3_270_);
lean_dec(v_h__2_269_);
v___x_271_ = lean_box(0);
v___x_272_ = lean_apply_1(v_h__1_268_, v___x_271_);
return v___x_272_;
}
else
{
lean_object* v_tail_273_; 
lean_dec(v_h__1_268_);
v_tail_273_ = lean_ctor_get(v_x_267_, 1);
if (lean_obj_tag(v_tail_273_) == 0)
{
lean_object* v_head_274_; lean_object* v___x_275_; 
lean_dec(v_h__3_270_);
v_head_274_ = lean_ctor_get(v_x_267_, 0);
lean_inc(v_head_274_);
lean_dec_ref_known(v_x_267_, 2);
v___x_275_ = lean_apply_1(v_h__2_269_, v_head_274_);
return v___x_275_;
}
else
{
lean_object* v_head_276_; lean_object* v___x_277_; 
lean_inc(v_tail_273_);
lean_dec(v_h__2_269_);
v_head_276_ = lean_ctor_get(v_x_267_, 0);
lean_inc(v_head_276_);
lean_dec_ref_known(v_x_267_, 2);
v___x_277_ = lean_apply_3(v_h__3_270_, v_head_276_, v_tail_273_, lean_box(0));
return v___x_277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_Forall_match__1_splitter(lean_object* v_00_u03b1_278_, lean_object* v_motive_279_, lean_object* v_x_280_, lean_object* v_h__1_281_, lean_object* v_h__2_282_, lean_object* v_h__3_283_){
_start:
{
if (lean_obj_tag(v_x_280_) == 0)
{
lean_object* v___x_284_; lean_object* v___x_285_; 
lean_dec(v_h__3_283_);
lean_dec(v_h__2_282_);
v___x_284_ = lean_box(0);
v___x_285_ = lean_apply_1(v_h__1_281_, v___x_284_);
return v___x_285_;
}
else
{
lean_object* v_tail_286_; 
lean_dec(v_h__1_281_);
v_tail_286_ = lean_ctor_get(v_x_280_, 1);
if (lean_obj_tag(v_tail_286_) == 0)
{
lean_object* v_head_287_; lean_object* v___x_288_; 
lean_dec(v_h__3_283_);
v_head_287_ = lean_ctor_get(v_x_280_, 0);
lean_inc(v_head_287_);
lean_dec_ref_known(v_x_280_, 2);
v___x_288_ = lean_apply_1(v_h__2_282_, v_head_287_);
return v___x_288_;
}
else
{
lean_object* v_head_289_; lean_object* v___x_290_; 
lean_inc(v_tail_286_);
lean_dec(v_h__2_282_);
v_head_289_ = lean_ctor_get(v_x_280_, 0);
lean_inc(v_head_289_);
lean_dec_ref_known(v_x_280_, 2);
v___x_290_ = lean_apply_3(v_h__3_283_, v_head_289_, v_tail_286_, lean_box(0));
return v___x_290_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2___redArg___lam__0(lean_object* v_head_291_, lean_object* v_x_292_, lean_object* v_x_293_){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_294_, 0, v_head_291_);
lean_ctor_set(v___x_294_, 1, v_x_293_);
v___x_295_ = lean_apply_1(v_x_292_, v___x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2___redArg(lean_object* v_t_296_, lean_object* v_ts_297_, lean_object* v_r_298_, lean_object* v_x_299_, lean_object* v_x_300_){
_start:
{
if (lean_obj_tag(v_x_299_) == 0)
{
lean_object* v___x_301_; 
lean_dec(v_x_300_);
lean_dec(v_t_296_);
v___x_301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_301_, 0, v_ts_297_);
lean_ctor_set(v___x_301_, 1, v_r_298_);
return v___x_301_;
}
else
{
lean_object* v_head_302_; lean_object* v_tail_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_324_; 
v_head_302_ = lean_ctor_get(v_x_299_, 0);
v_tail_303_ = lean_ctor_get(v_x_299_, 1);
v_isSharedCheck_324_ = !lean_is_exclusive(v_x_299_);
if (v_isSharedCheck_324_ == 0)
{
v___x_305_ = v_x_299_;
v_isShared_306_ = v_isSharedCheck_324_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_tail_303_);
lean_inc(v_head_302_);
lean_dec(v_x_299_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_324_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___f_307_; lean_object* v___x_308_; lean_object* v_fst_309_; lean_object* v_snd_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_323_; 
lean_inc(v_x_300_);
lean_inc(v_head_302_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_List_permutationsAux2___redArg___lam__0), 3, 2);
lean_closure_set(v___f_307_, 0, v_head_302_);
lean_closure_set(v___f_307_, 1, v_x_300_);
lean_inc(v_t_296_);
v___x_308_ = lp_mathlib_List_permutationsAux2___redArg(v_t_296_, v_ts_297_, v_r_298_, v_tail_303_, v___f_307_);
v_fst_309_ = lean_ctor_get(v___x_308_, 0);
v_snd_310_ = lean_ctor_get(v___x_308_, 1);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_323_ == 0)
{
v___x_312_ = v___x_308_;
v_isShared_313_ = v_isSharedCheck_323_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_snd_310_);
lean_inc(v_fst_309_);
lean_dec(v___x_308_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_323_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_315_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 1, v_fst_309_);
v___x_315_ = v___x_305_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_head_302_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v_fst_309_);
v___x_315_ = v_reuseFailAlloc_322_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_320_; 
lean_inc_ref(v___x_315_);
v___x_316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_316_, 0, v_t_296_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = lean_apply_1(v_x_300_, v___x_316_);
v___x_318_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_317_);
lean_ctor_set(v___x_318_, 1, v_snd_310_);
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 1, v___x_318_);
lean_ctor_set(v___x_312_, 0, v___x_315_);
v___x_320_ = v___x_312_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___x_315_);
lean_ctor_set(v_reuseFailAlloc_321_, 1, v___x_318_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux2(lean_object* v_00_u03b1_325_, lean_object* v_00_u03b2_326_, lean_object* v_t_327_, lean_object* v_ts_328_, lean_object* v_r_329_, lean_object* v_x_330_, lean_object* v_x_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_List_permutationsAux2___redArg(v_t_327_, v_ts_328_, v_r_329_, v_x_330_, v_x_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux_rec___redArg(lean_object* v_H0_333_, lean_object* v_H1_334_, lean_object* v_x_335_, lean_object* v_x_336_){
_start:
{
if (lean_obj_tag(v_x_335_) == 0)
{
lean_object* v___x_337_; 
lean_dec(v_H1_334_);
v___x_337_ = lean_apply_1(v_H0_333_, v_x_336_);
return v___x_337_;
}
else
{
lean_object* v_head_338_; lean_object* v_tail_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_350_; 
v_head_338_ = lean_ctor_get(v_x_335_, 0);
v_tail_339_ = lean_ctor_get(v_x_335_, 1);
v_isSharedCheck_350_ = !lean_is_exclusive(v_x_335_);
if (v_isSharedCheck_350_ == 0)
{
v___x_341_ = v_x_335_;
v_isShared_342_ = v_isSharedCheck_350_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_tail_339_);
lean_inc(v_head_338_);
lean_dec(v_x_335_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_350_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_344_; 
lean_inc(v_x_336_);
lean_inc(v_head_338_);
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 1, v_x_336_);
v___x_344_ = v___x_341_;
goto v_reusejp_343_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_head_338_);
lean_ctor_set(v_reuseFailAlloc_349_, 1, v_x_336_);
v___x_344_ = v_reuseFailAlloc_349_;
goto v_reusejp_343_;
}
v_reusejp_343_:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
lean_inc(v_tail_339_);
lean_inc_n(v_H1_334_, 2);
lean_inc(v_H0_333_);
v___x_345_ = lp_mathlib_List_permutationsAux_rec___redArg(v_H0_333_, v_H1_334_, v_tail_339_, v___x_344_);
v___x_346_ = lean_box(0);
lean_inc(v_x_336_);
v___x_347_ = lp_mathlib_List_permutationsAux_rec___redArg(v_H0_333_, v_H1_334_, v_x_336_, v___x_346_);
v___x_348_ = lean_apply_5(v_H1_334_, v_head_338_, v_tail_339_, v_x_336_, v___x_345_, v___x_347_);
return v___x_348_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux_rec(lean_object* v_00_u03b1_351_, lean_object* v_C_352_, lean_object* v_H0_353_, lean_object* v_H1_354_, lean_object* v_x_355_, lean_object* v_x_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_List_permutationsAux_rec___redArg(v_H0_353_, v_H1_354_, v_x_355_, v_x_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_permutationsAux_rec_match__1_splitter___redArg(lean_object* v_x_358_, lean_object* v_x_359_, lean_object* v_h__1_360_, lean_object* v_h__2_361_){
_start:
{
if (lean_obj_tag(v_x_358_) == 0)
{
lean_object* v___x_362_; 
lean_dec(v_h__2_361_);
v___x_362_ = lean_apply_1(v_h__1_360_, v_x_359_);
return v___x_362_;
}
else
{
lean_object* v_head_363_; lean_object* v_tail_364_; lean_object* v___x_365_; 
lean_dec(v_h__1_360_);
v_head_363_ = lean_ctor_get(v_x_358_, 0);
lean_inc(v_head_363_);
v_tail_364_ = lean_ctor_get(v_x_358_, 1);
lean_inc(v_tail_364_);
lean_dec_ref_known(v_x_358_, 2);
v___x_365_ = lean_apply_3(v_h__2_361_, v_head_363_, v_tail_364_, v_x_359_);
return v___x_365_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_permutationsAux_rec_match__1_splitter(lean_object* v_00_u03b1_366_, lean_object* v_motive_367_, lean_object* v_x_368_, lean_object* v_x_369_, lean_object* v_h__1_370_, lean_object* v_h__2_371_){
_start:
{
if (lean_obj_tag(v_x_368_) == 0)
{
lean_object* v___x_372_; 
lean_dec(v_h__2_371_);
v___x_372_ = lean_apply_1(v_h__1_370_, v_x_369_);
return v___x_372_;
}
else
{
lean_object* v_head_373_; lean_object* v_tail_374_; lean_object* v___x_375_; 
lean_dec(v_h__1_370_);
v_head_373_ = lean_ctor_get(v_x_368_, 0);
lean_inc(v_head_373_);
v_tail_374_ = lean_ctor_get(v_x_368_, 1);
lean_inc(v_tail_374_);
lean_dec_ref_known(v_x_368_, 2);
v___x_375_ = lean_apply_3(v_h__2_371_, v_head_373_, v_tail_374_, v_x_369_);
return v___x_375_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__0(lean_object* v_x_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lean_box(0);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__0___boxed(lean_object* v_x_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_List_permutationsAux___redArg___lam__0(v_x_378_);
lean_dec(v_x_378_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0(lean_object* v___y_380_){
_start:
{
lean_inc(v___y_380_);
return v___y_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v___y_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___lam__0(v___y_381_);
lean_dec(v___y_381_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg(lean_object* v_t_384_, lean_object* v_ts_385_, lean_object* v_as_386_, size_t v_i_387_, size_t v_stop_388_, lean_object* v_b_389_){
_start:
{
uint8_t v___x_390_; 
v___x_390_ = lean_usize_dec_eq(v_i_387_, v_stop_388_);
if (v___x_390_ == 0)
{
lean_object* v___f_391_; size_t v___x_392_; size_t v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_snd_396_; 
v___f_391_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___closed__0));
v___x_392_ = ((size_t)1ULL);
v___x_393_ = lean_usize_sub(v_i_387_, v___x_392_);
v___x_394_ = lean_array_uget_borrowed(v_as_386_, v___x_393_);
lean_inc(v___x_394_);
lean_inc(v_ts_385_);
lean_inc(v_t_384_);
v___x_395_ = lp_mathlib_List_permutationsAux2___redArg(v_t_384_, v_ts_385_, v_b_389_, v___x_394_, v___f_391_);
v_snd_396_ = lean_ctor_get(v___x_395_, 1);
lean_inc(v_snd_396_);
lean_dec_ref(v___x_395_);
v_i_387_ = v___x_393_;
v_b_389_ = v_snd_396_;
goto _start;
}
else
{
lean_dec(v_ts_385_);
lean_dec(v_t_384_);
return v_b_389_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg___boxed(lean_object* v_t_398_, lean_object* v_ts_399_, lean_object* v_as_400_, lean_object* v_i_401_, lean_object* v_stop_402_, lean_object* v_b_403_){
_start:
{
size_t v_i_boxed_404_; size_t v_stop_boxed_405_; lean_object* v_res_406_; 
v_i_boxed_404_ = lean_unbox_usize(v_i_401_);
lean_dec(v_i_401_);
v_stop_boxed_405_ = lean_unbox_usize(v_stop_402_);
lean_dec(v_stop_402_);
v_res_406_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg(v_t_398_, v_ts_399_, v_as_400_, v_i_boxed_404_, v_stop_boxed_405_, v_b_403_);
lean_dec_ref(v_as_400_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0___redArg(lean_object* v_t_407_, lean_object* v_ts_408_, lean_object* v_init_409_, lean_object* v_l_410_){
_start:
{
lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; uint8_t v___x_414_; 
v___x_411_ = lean_array_mk(v_l_410_);
v___x_412_ = lean_array_get_size(v___x_411_);
v___x_413_ = lean_unsigned_to_nat(0u);
v___x_414_ = lean_nat_dec_lt(v___x_413_, v___x_412_);
if (v___x_414_ == 0)
{
lean_dec_ref(v___x_411_);
lean_dec(v_ts_408_);
lean_dec(v_t_407_);
return v_init_409_;
}
else
{
size_t v___x_415_; size_t v___x_416_; lean_object* v___x_417_; 
v___x_415_ = lean_usize_of_nat(v___x_412_);
v___x_416_ = ((size_t)0ULL);
v___x_417_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg(v_t_407_, v_ts_408_, v___x_411_, v___x_415_, v___x_416_, v_init_409_);
lean_dec_ref(v___x_411_);
return v___x_417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg___lam__1(lean_object* v_t_418_, lean_object* v_ts_419_, lean_object* v_is_420_, lean_object* v_IH1_421_, lean_object* v_IH2_422_){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_423_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_423_, 0, v_is_420_);
lean_ctor_set(v___x_423_, 1, v_IH2_422_);
v___x_424_ = lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0___redArg(v_t_418_, v_ts_419_, v_IH1_421_, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux___redArg(lean_object* v_l_u2081_427_, lean_object* v_l_u2082_428_){
_start:
{
lean_object* v___f_429_; lean_object* v___f_430_; lean_object* v___x_431_; 
v___f_429_ = ((lean_object*)(lp_mathlib_List_permutationsAux___redArg___closed__0));
v___f_430_ = ((lean_object*)(lp_mathlib_List_permutationsAux___redArg___closed__1));
v___x_431_ = lp_mathlib_List_permutationsAux_rec___redArg(v___f_429_, v___f_430_, v_l_u2081_427_, v_l_u2082_428_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutationsAux(lean_object* v_00_u03b1_432_, lean_object* v_l_u2081_433_, lean_object* v_l_u2082_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_List_permutationsAux___redArg(v_l_u2081_433_, v_l_u2082_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0(lean_object* v_00_u03b1_436_, lean_object* v_t_437_, lean_object* v_ts_438_, lean_object* v_init_439_, lean_object* v_l_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_List_foldrTR___at___00List_permutationsAux_spec__0___redArg(v_t_437_, v_ts_438_, v_init_439_, v_l_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0(lean_object* v_00_u03b1_442_, lean_object* v_t_443_, lean_object* v_ts_444_, lean_object* v_as_445_, size_t v_i_446_, size_t v_stop_447_, lean_object* v_b_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___redArg(v_t_443_, v_ts_444_, v_as_445_, v_i_446_, v_stop_447_, v_b_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0___boxed(lean_object* v_00_u03b1_450_, lean_object* v_t_451_, lean_object* v_ts_452_, lean_object* v_as_453_, lean_object* v_i_454_, lean_object* v_stop_455_, lean_object* v_b_456_){
_start:
{
size_t v_i_boxed_457_; size_t v_stop_boxed_458_; lean_object* v_res_459_; 
v_i_boxed_457_ = lean_unbox_usize(v_i_454_);
lean_dec(v_i_454_);
v_stop_boxed_458_ = lean_unbox_usize(v_stop_455_);
lean_dec(v_stop_455_);
v_res_459_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_permutationsAux_spec__0_spec__0(v_00_u03b1_450_, v_t_451_, v_ts_452_, v_as_453_, v_i_boxed_457_, v_stop_boxed_458_, v_b_456_);
lean_dec_ref(v_as_453_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations___redArg(lean_object* v_l_460_){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_461_ = lean_box(0);
lean_inc(v_l_460_);
v___x_462_ = lp_mathlib_List_permutationsAux___redArg(v_l_460_, v___x_461_);
v___x_463_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_463_, 0, v_l_460_);
lean_ctor_set(v___x_463_, 1, v___x_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations(lean_object* v_00_u03b1_464_, lean_object* v_l_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_List_permutations___redArg(v_l_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0___redArg(lean_object* v_head_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
if (lean_obj_tag(v_a_468_) == 0)
{
lean_object* v___x_470_; 
lean_dec(v_head_467_);
v___x_470_ = l_List_reverse___redArg(v_a_469_);
return v___x_470_;
}
else
{
lean_object* v_head_471_; lean_object* v_tail_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_481_; 
v_head_471_ = lean_ctor_get(v_a_468_, 0);
v_tail_472_ = lean_ctor_get(v_a_468_, 1);
v_isSharedCheck_481_ = !lean_is_exclusive(v_a_468_);
if (v_isSharedCheck_481_ == 0)
{
v___x_474_ = v_a_468_;
v_isShared_475_ = v_isSharedCheck_481_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_tail_472_);
lean_inc(v_head_471_);
lean_dec(v_a_468_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_481_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_477_; 
lean_inc(v_head_467_);
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 1, v_head_471_);
lean_ctor_set(v___x_474_, 0, v_head_467_);
v___x_477_ = v___x_474_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_head_467_);
lean_ctor_set(v_reuseFailAlloc_480_, 1, v_head_471_);
v___x_477_ = v_reuseFailAlloc_480_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_478_; 
v___x_478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v_a_469_);
v_a_468_ = v_tail_472_;
v_a_469_ = v___x_478_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27Aux___redArg(lean_object* v_t_482_, lean_object* v_x_483_){
_start:
{
if (lean_obj_tag(v_x_483_) == 0)
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_484_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_484_, 0, v_t_482_);
lean_ctor_set(v___x_484_, 1, v_x_483_);
v___x_485_ = lean_box(0);
v___x_486_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_484_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
return v___x_486_;
}
else
{
lean_object* v_head_487_; lean_object* v_tail_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_head_487_ = lean_ctor_get(v_x_483_, 0);
lean_inc(v_head_487_);
v_tail_488_ = lean_ctor_get(v_x_483_, 1);
lean_inc(v_tail_488_);
lean_inc(v_t_482_);
v___x_489_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_489_, 0, v_t_482_);
lean_ctor_set(v___x_489_, 1, v_x_483_);
v___x_490_ = lp_mathlib_List_permutations_x27Aux___redArg(v_t_482_, v_tail_488_);
v___x_491_ = lean_box(0);
v___x_492_ = lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0___redArg(v_head_487_, v___x_490_, v___x_491_);
v___x_493_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_489_);
lean_ctor_set(v___x_493_, 1, v___x_492_);
return v___x_493_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27Aux(lean_object* v_00_u03b1_494_, lean_object* v_t_495_, lean_object* v_x_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_List_permutations_x27Aux___redArg(v_t_495_, v_x_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0(lean_object* v_00_u03b1_498_, lean_object* v_head_499_, lean_object* v_a_500_, lean_object* v_a_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_List_mapTR_loop___at___00List_permutations_x27Aux_spec__0___redArg(v_head_499_, v_a_500_, v_a_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_headI_match__1_splitter___redArg(lean_object* v_x_503_, lean_object* v_h__1_504_, lean_object* v_h__2_505_){
_start:
{
if (lean_obj_tag(v_x_503_) == 0)
{
lean_object* v___x_506_; lean_object* v___x_507_; 
lean_dec(v_h__2_505_);
v___x_506_ = lean_box(0);
v___x_507_ = lean_apply_1(v_h__1_504_, v___x_506_);
return v___x_507_;
}
else
{
lean_object* v_head_508_; lean_object* v_tail_509_; lean_object* v___x_510_; 
lean_dec(v_h__1_504_);
v_head_508_ = lean_ctor_get(v_x_503_, 0);
lean_inc(v_head_508_);
v_tail_509_ = lean_ctor_get(v_x_503_, 1);
lean_inc(v_tail_509_);
lean_dec_ref_known(v_x_503_, 2);
v___x_510_ = lean_apply_2(v_h__2_505_, v_head_508_, v_tail_509_);
return v___x_510_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_headI_match__1_splitter(lean_object* v_00_u03b1_511_, lean_object* v_motive_512_, lean_object* v_x_513_, lean_object* v_h__1_514_, lean_object* v_h__2_515_){
_start:
{
if (lean_obj_tag(v_x_513_) == 0)
{
lean_object* v___x_516_; lean_object* v___x_517_; 
lean_dec(v_h__2_515_);
v___x_516_ = lean_box(0);
v___x_517_ = lean_apply_1(v_h__1_514_, v___x_516_);
return v___x_517_;
}
else
{
lean_object* v_head_518_; lean_object* v_tail_519_; lean_object* v___x_520_; 
lean_dec(v_h__1_514_);
v_head_518_ = lean_ctor_get(v_x_513_, 0);
lean_inc(v_head_518_);
v_tail_519_ = lean_ctor_get(v_x_513_, 1);
lean_inc(v_tail_519_);
lean_dec_ref_known(v_x_513_, 2);
v___x_520_ = lean_apply_2(v_h__2_515_, v_head_518_, v_tail_519_);
return v___x_520_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0___redArg(lean_object* v_head_521_, lean_object* v_a_522_, lean_object* v_a_523_){
_start:
{
if (lean_obj_tag(v_a_522_) == 0)
{
lean_object* v___x_524_; 
lean_dec(v_head_521_);
v___x_524_ = lean_array_to_list(v_a_523_);
return v___x_524_;
}
else
{
lean_object* v_head_525_; lean_object* v_tail_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v_head_525_ = lean_ctor_get(v_a_522_, 0);
lean_inc(v_head_525_);
v_tail_526_ = lean_ctor_get(v_a_522_, 1);
lean_inc(v_tail_526_);
lean_dec_ref_known(v_a_522_, 2);
lean_inc(v_head_521_);
v___x_527_ = lp_mathlib_List_permutations_x27Aux___redArg(v_head_521_, v_head_525_);
v___x_528_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_523_, v___x_527_);
v_a_522_ = v_tail_526_;
v_a_523_ = v___x_528_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27___redArg(lean_object* v_x_532_){
_start:
{
if (lean_obj_tag(v_x_532_) == 0)
{
lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_533_ = lean_box(0);
v___x_534_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_534_, 0, v_x_532_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
return v___x_534_;
}
else
{
lean_object* v_head_535_; lean_object* v_tail_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v_head_535_ = lean_ctor_get(v_x_532_, 0);
lean_inc(v_head_535_);
v_tail_536_ = lean_ctor_get(v_x_532_, 1);
lean_inc(v_tail_536_);
lean_dec_ref_known(v_x_532_, 2);
v___x_537_ = lp_mathlib_List_permutations_x27___redArg(v_tail_536_);
v___x_538_ = ((lean_object*)(lp_mathlib_List_permutations_x27___redArg___closed__0));
v___x_539_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0___redArg(v_head_535_, v___x_537_, v___x_538_);
return v___x_539_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_permutations_x27(lean_object* v_00_u03b1_540_, lean_object* v_x_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_List_permutations_x27___redArg(v_x_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0(lean_object* v_00_u03b1_543_, lean_object* v_head_544_, lean_object* v_a_545_, lean_object* v_a_546_){
_start:
{
lean_object* v___x_547_; 
v___x_547_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_permutations_x27_spec__0___redArg(v_head_544_, v_a_545_, v_a_546_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_extractp___redArg(lean_object* v_inst_548_, lean_object* v_x_549_){
_start:
{
if (lean_obj_tag(v_x_549_) == 0)
{
lean_object* v___x_550_; lean_object* v___x_551_; 
lean_dec_ref(v_inst_548_);
v___x_550_ = lean_box(0);
v___x_551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_551_, 0, v___x_550_);
lean_ctor_set(v___x_551_, 1, v_x_549_);
return v___x_551_;
}
else
{
lean_object* v_head_552_; lean_object* v_tail_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_574_; 
v_head_552_ = lean_ctor_get(v_x_549_, 0);
v_tail_553_ = lean_ctor_get(v_x_549_, 1);
v_isSharedCheck_574_ = !lean_is_exclusive(v_x_549_);
if (v_isSharedCheck_574_ == 0)
{
v___x_555_ = v_x_549_;
v_isShared_556_ = v_isSharedCheck_574_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_tail_553_);
lean_inc(v_head_552_);
lean_dec(v_x_549_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_574_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; uint8_t v___x_558_; 
lean_inc_ref(v_inst_548_);
lean_inc(v_head_552_);
v___x_557_ = lean_apply_1(v_inst_548_, v_head_552_);
v___x_558_ = lean_unbox(v___x_557_);
if (v___x_558_ == 0)
{
lean_object* v___x_559_; lean_object* v_fst_560_; lean_object* v_snd_561_; lean_object* v___x_563_; uint8_t v_isShared_564_; uint8_t v_isSharedCheck_571_; 
v___x_559_ = lp_mathlib_List_extractp___redArg(v_inst_548_, v_tail_553_);
v_fst_560_ = lean_ctor_get(v___x_559_, 0);
v_snd_561_ = lean_ctor_get(v___x_559_, 1);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_571_ == 0)
{
v___x_563_ = v___x_559_;
v_isShared_564_ = v_isSharedCheck_571_;
goto v_resetjp_562_;
}
else
{
lean_inc(v_snd_561_);
lean_inc(v_fst_560_);
lean_dec(v___x_559_);
v___x_563_ = lean_box(0);
v_isShared_564_ = v_isSharedCheck_571_;
goto v_resetjp_562_;
}
v_resetjp_562_:
{
lean_object* v___x_566_; 
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 1, v_snd_561_);
v___x_566_ = v___x_555_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_head_552_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v_snd_561_);
v___x_566_ = v_reuseFailAlloc_570_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
lean_object* v___x_568_; 
if (v_isShared_564_ == 0)
{
lean_ctor_set(v___x_563_, 1, v___x_566_);
v___x_568_ = v___x_563_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_fst_560_);
lean_ctor_set(v_reuseFailAlloc_569_, 1, v___x_566_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
else
{
lean_object* v___x_572_; lean_object* v___x_573_; 
lean_del_object(v___x_555_);
lean_dec_ref(v_inst_548_);
v___x_572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_572_, 0, v_head_552_);
v___x_573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v_tail_553_);
return v___x_573_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_extractp(lean_object* v_00_u03b1_575_, lean_object* v_p_576_, lean_object* v_inst_577_, lean_object* v_x_578_){
_start:
{
lean_object* v___x_579_; 
v___x_579_ = lp_mathlib_List_extractp___redArg(v_inst_577_, v_x_578_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instSProd(lean_object* v_00_u03b1_581_, lean_object* v_00_u03b2_582_){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = ((lean_object*)(lp_mathlib_List_instSProd___closed__0));
return v___x_583_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_dedup___redArg___lam__0(lean_object* v_inst_584_, lean_object* v_a_585_, lean_object* v_b_586_){
_start:
{
lean_object* v___x_587_; uint8_t v___x_588_; 
v___x_587_ = lean_apply_2(v_inst_584_, v_a_585_, v_b_586_);
v___x_588_ = lean_unbox(v___x_587_);
if (v___x_588_ == 0)
{
uint8_t v___x_589_; 
v___x_589_ = 1;
return v___x_589_;
}
else
{
uint8_t v___x_590_; 
v___x_590_ = 0;
return v___x_590_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___redArg___lam__0___boxed(lean_object* v_inst_591_, lean_object* v_a_592_, lean_object* v_b_593_){
_start:
{
uint8_t v_res_594_; lean_object* v_r_595_; 
v_res_594_ = lp_mathlib_List_dedup___redArg___lam__0(v_inst_591_, v_a_592_, v_b_593_);
v_r_595_ = lean_box(v_res_594_);
return v_r_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___redArg(lean_object* v_inst_596_, lean_object* v_l_597_){
_start:
{
lean_object* v___f_598_; lean_object* v___x_599_; 
v___f_598_ = lean_alloc_closure((void*)(lp_mathlib_List_dedup___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_598_, 0, v_inst_596_);
v___x_599_ = lp_batteries_List_pwFilter___redArg(v___f_598_, v_l_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dedup(lean_object* v_00_u03b1_600_, lean_object* v_inst_601_, lean_object* v_l_602_){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = lp_mathlib_List_dedup___redArg(v_inst_601_, v_l_602_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_destutter_x27___redArg(lean_object* v_inst_604_, lean_object* v_x_605_, lean_object* v_x_606_){
_start:
{
if (lean_obj_tag(v_x_606_) == 0)
{
lean_object* v___x_607_; 
lean_dec_ref(v_inst_604_);
v___x_607_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_607_, 0, v_x_605_);
lean_ctor_set(v___x_607_, 1, v_x_606_);
return v___x_607_;
}
else
{
lean_object* v_head_608_; lean_object* v_tail_609_; lean_object* v___x_611_; uint8_t v_isShared_612_; uint8_t v_isSharedCheck_620_; 
v_head_608_ = lean_ctor_get(v_x_606_, 0);
v_tail_609_ = lean_ctor_get(v_x_606_, 1);
v_isSharedCheck_620_ = !lean_is_exclusive(v_x_606_);
if (v_isSharedCheck_620_ == 0)
{
v___x_611_ = v_x_606_;
v_isShared_612_ = v_isSharedCheck_620_;
goto v_resetjp_610_;
}
else
{
lean_inc(v_tail_609_);
lean_inc(v_head_608_);
lean_dec(v_x_606_);
v___x_611_ = lean_box(0);
v_isShared_612_ = v_isSharedCheck_620_;
goto v_resetjp_610_;
}
v_resetjp_610_:
{
lean_object* v___x_613_; uint8_t v___x_614_; 
lean_inc_ref(v_inst_604_);
lean_inc(v_head_608_);
lean_inc(v_x_605_);
v___x_613_ = lean_apply_2(v_inst_604_, v_x_605_, v_head_608_);
v___x_614_ = lean_unbox(v___x_613_);
if (v___x_614_ == 0)
{
lean_del_object(v___x_611_);
lean_dec(v_head_608_);
v_x_606_ = v_tail_609_;
goto _start;
}
else
{
lean_object* v___x_616_; lean_object* v___x_618_; 
v___x_616_ = lp_mathlib_List_destutter_x27___redArg(v_inst_604_, v_head_608_, v_tail_609_);
if (v_isShared_612_ == 0)
{
lean_ctor_set(v___x_611_, 1, v___x_616_);
lean_ctor_set(v___x_611_, 0, v_x_605_);
v___x_618_ = v___x_611_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_x_605_);
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
LEAN_EXPORT lean_object* lp_mathlib_List_destutter_x27(lean_object* v_00_u03b1_621_, lean_object* v_R_622_, lean_object* v_inst_623_, lean_object* v_x_624_, lean_object* v_x_625_){
_start:
{
lean_object* v___x_626_; 
v___x_626_ = lp_mathlib_List_destutter_x27___redArg(v_inst_623_, v_x_624_, v_x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_destutter___redArg(lean_object* v_inst_627_, lean_object* v_x_628_){
_start:
{
if (lean_obj_tag(v_x_628_) == 0)
{
lean_dec_ref(v_inst_627_);
return v_x_628_;
}
else
{
lean_object* v_head_629_; lean_object* v_tail_630_; lean_object* v___x_631_; 
v_head_629_ = lean_ctor_get(v_x_628_, 0);
lean_inc(v_head_629_);
v_tail_630_ = lean_ctor_get(v_x_628_, 1);
lean_inc(v_tail_630_);
lean_dec_ref_known(v_x_628_, 2);
v___x_631_ = lp_mathlib_List_destutter_x27___redArg(v_inst_627_, v_head_629_, v_tail_630_);
return v___x_631_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_destutter(lean_object* v_00_u03b1_632_, lean_object* v_R_633_, lean_object* v_inst_634_, lean_object* v_x_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_mathlib_List_destutter___redArg(v_inst_634_, v_x_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_chooseX___redArg(lean_object* v_inst_637_, lean_object* v_x_638_){
_start:
{
lean_object* v_head_639_; lean_object* v_tail_640_; lean_object* v___x_641_; uint8_t v___x_642_; 
v_head_639_ = lean_ctor_get(v_x_638_, 0);
lean_inc_n(v_head_639_, 2);
v_tail_640_ = lean_ctor_get(v_x_638_, 1);
lean_inc(v_tail_640_);
lean_dec(v_x_638_);
lean_inc_ref(v_inst_637_);
v___x_641_ = lean_apply_1(v_inst_637_, v_head_639_);
v___x_642_ = lean_unbox(v___x_641_);
if (v___x_642_ == 0)
{
lean_dec(v_head_639_);
v_x_638_ = v_tail_640_;
goto _start;
}
else
{
lean_dec(v_tail_640_);
lean_dec_ref(v_inst_637_);
return v_head_639_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_chooseX(lean_object* v_00_u03b1_644_, lean_object* v_p_645_, lean_object* v_inst_646_, lean_object* v_x_647_, lean_object* v_x_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_List_chooseX___redArg(v_inst_646_, v_x_647_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_choose___redArg(lean_object* v_inst_650_, lean_object* v_l_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_mathlib_List_chooseX___redArg(v_inst_650_, v_l_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_choose(lean_object* v_00_u03b1_653_, lean_object* v_p_654_, lean_object* v_inst_655_, lean_object* v_l_656_, lean_object* v_hp_657_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lp_mathlib_List_chooseX___redArg(v_inst_655_, v_l_656_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__1(lean_object* v_f_659_, lean_object* v_head_660_, lean_object* v_inst_661_, lean_object* v_tail_662_, lean_object* v_toBind_663_, lean_object* v___f_664_, lean_object* v_____x_665_){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_666_ = lean_apply_1(v_f_659_, v_head_660_);
v___x_667_ = l_List_mapM_x27___redArg(v_inst_661_, v___x_666_, v_tail_662_);
v___x_668_ = lean_apply_4(v_toBind_663_, lean_box(0), lean_box(0), v___x_667_, v___f_664_);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__0___boxed(lean_object* v_inst_669_, lean_object* v_f_670_, lean_object* v_tail_671_, lean_object* v_____x_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_List_mapDiagM_x27___redArg___lam__0(v_inst_669_, v_f_670_, v_tail_671_, v_____x_672_);
lean_dec(v_____x_672_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg(lean_object* v_inst_674_, lean_object* v_f_675_, lean_object* v_x_676_){
_start:
{
if (lean_obj_tag(v_x_676_) == 0)
{
lean_object* v_toApplicative_677_; lean_object* v_toPure_678_; lean_object* v___x_679_; lean_object* v___x_680_; 
v_toApplicative_677_ = lean_ctor_get(v_inst_674_, 0);
lean_inc_ref(v_toApplicative_677_);
lean_dec(v_f_675_);
lean_dec_ref(v_inst_674_);
v_toPure_678_ = lean_ctor_get(v_toApplicative_677_, 1);
lean_inc(v_toPure_678_);
lean_dec_ref(v_toApplicative_677_);
v___x_679_ = lean_box(0);
v___x_680_ = lean_apply_2(v_toPure_678_, lean_box(0), v___x_679_);
return v___x_680_;
}
else
{
lean_object* v_toBind_681_; lean_object* v_head_682_; lean_object* v_tail_683_; lean_object* v___f_684_; lean_object* v___f_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v_toBind_681_ = lean_ctor_get(v_inst_674_, 1);
lean_inc_n(v_toBind_681_, 2);
v_head_682_ = lean_ctor_get(v_x_676_, 0);
lean_inc_n(v_head_682_, 3);
v_tail_683_ = lean_ctor_get(v_x_676_, 1);
lean_inc_n(v_tail_683_, 2);
lean_dec_ref_known(v_x_676_, 2);
lean_inc_n(v_f_675_, 2);
lean_inc_ref(v_inst_674_);
v___f_684_ = lean_alloc_closure((void*)(lp_mathlib_List_mapDiagM_x27___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_684_, 0, v_inst_674_);
lean_closure_set(v___f_684_, 1, v_f_675_);
lean_closure_set(v___f_684_, 2, v_tail_683_);
v___f_685_ = lean_alloc_closure((void*)(lp_mathlib_List_mapDiagM_x27___redArg___lam__1), 7, 6);
lean_closure_set(v___f_685_, 0, v_f_675_);
lean_closure_set(v___f_685_, 1, v_head_682_);
lean_closure_set(v___f_685_, 2, v_inst_674_);
lean_closure_set(v___f_685_, 3, v_tail_683_);
lean_closure_set(v___f_685_, 4, v_toBind_681_);
lean_closure_set(v___f_685_, 5, v___f_684_);
v___x_686_ = lean_apply_2(v_f_675_, v_head_682_, v_head_682_);
v___x_687_ = lean_apply_4(v_toBind_681_, lean_box(0), lean_box(0), v___x_686_, v___f_685_);
return v___x_687_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27___redArg___lam__0(lean_object* v_inst_688_, lean_object* v_f_689_, lean_object* v_tail_690_, lean_object* v_____x_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib_List_mapDiagM_x27___redArg(v_inst_688_, v_f_689_, v_tail_690_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapDiagM_x27(lean_object* v_m_693_, lean_object* v_inst_694_, lean_object* v_00_u03b1_695_, lean_object* v_f_696_, lean_object* v_x_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_List_mapDiagM_x27___redArg(v_inst_694_, v_f_696_, v_x_697_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0___redArg(lean_object* v_f_699_, lean_object* v_a_700_, lean_object* v_a_701_){
_start:
{
if (lean_obj_tag(v_a_700_) == 0)
{
lean_object* v___x_702_; 
lean_dec(v_f_699_);
v___x_702_ = l_List_reverse___redArg(v_a_701_);
return v___x_702_;
}
else
{
lean_object* v_head_703_; lean_object* v_tail_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_714_; 
v_head_703_ = lean_ctor_get(v_a_700_, 0);
v_tail_704_ = lean_ctor_get(v_a_700_, 1);
v_isSharedCheck_714_ = !lean_is_exclusive(v_a_700_);
if (v_isSharedCheck_714_ == 0)
{
v___x_706_ = v_a_700_;
v_isShared_707_ = v_isSharedCheck_714_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_tail_704_);
lean_inc(v_head_703_);
lean_dec(v_a_700_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_714_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_711_; 
v___x_708_ = lean_box(0);
lean_inc(v_f_699_);
v___x_709_ = lean_apply_2(v_f_699_, v_head_703_, v___x_708_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 1, v_a_701_);
lean_ctor_set(v___x_706_, 0, v___x_709_);
v___x_711_ = v___x_706_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v___x_709_);
lean_ctor_set(v_reuseFailAlloc_713_, 1, v_a_701_);
v___x_711_ = v_reuseFailAlloc_713_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
v_a_700_ = v_tail_704_;
v_a_701_ = v___x_711_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left_x27___redArg(lean_object* v_f_715_, lean_object* v_x_716_, lean_object* v_x_717_){
_start:
{
if (lean_obj_tag(v_x_716_) == 0)
{
lean_object* v___x_718_; lean_object* v___x_719_; 
lean_dec(v_f_715_);
v___x_718_ = lean_box(0);
v___x_719_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v_x_717_);
return v___x_719_;
}
else
{
if (lean_obj_tag(v_x_717_) == 0)
{
lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_720_ = lean_box(0);
v___x_721_ = lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0___redArg(v_f_715_, v_x_716_, v___x_720_);
v___x_722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_722_, 0, v___x_721_);
lean_ctor_set(v___x_722_, 1, v_x_717_);
return v___x_722_;
}
else
{
lean_object* v_head_723_; lean_object* v_tail_724_; lean_object* v_head_725_; lean_object* v_tail_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_745_; 
v_head_723_ = lean_ctor_get(v_x_716_, 0);
lean_inc(v_head_723_);
v_tail_724_ = lean_ctor_get(v_x_716_, 1);
lean_inc(v_tail_724_);
lean_dec_ref_known(v_x_716_, 2);
v_head_725_ = lean_ctor_get(v_x_717_, 0);
v_tail_726_ = lean_ctor_get(v_x_717_, 1);
v_isSharedCheck_745_ = !lean_is_exclusive(v_x_717_);
if (v_isSharedCheck_745_ == 0)
{
v___x_728_ = v_x_717_;
v_isShared_729_ = v_isSharedCheck_745_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_tail_726_);
lean_inc(v_head_725_);
lean_dec(v_x_717_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_745_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v_rec_x27_730_; lean_object* v_fst_731_; lean_object* v_snd_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_744_; 
lean_inc(v_f_715_);
v_rec_x27_730_ = lp_mathlib_List_map_u2082Left_x27___redArg(v_f_715_, v_tail_724_, v_tail_726_);
v_fst_731_ = lean_ctor_get(v_rec_x27_730_, 0);
v_snd_732_ = lean_ctor_get(v_rec_x27_730_, 1);
v_isSharedCheck_744_ = !lean_is_exclusive(v_rec_x27_730_);
if (v_isSharedCheck_744_ == 0)
{
v___x_734_ = v_rec_x27_730_;
v_isShared_735_ = v_isSharedCheck_744_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_snd_732_);
lean_inc(v_fst_731_);
lean_dec(v_rec_x27_730_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_744_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_739_; 
v___x_736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_736_, 0, v_head_725_);
v___x_737_ = lean_apply_2(v_f_715_, v_head_723_, v___x_736_);
if (v_isShared_729_ == 0)
{
lean_ctor_set(v___x_728_, 1, v_fst_731_);
lean_ctor_set(v___x_728_, 0, v___x_737_);
v___x_739_ = v___x_728_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v___x_737_);
lean_ctor_set(v_reuseFailAlloc_743_, 1, v_fst_731_);
v___x_739_ = v_reuseFailAlloc_743_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
lean_object* v___x_741_; 
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 0, v___x_739_);
v___x_741_ = v___x_734_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v___x_739_);
lean_ctor_set(v_reuseFailAlloc_742_, 1, v_snd_732_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left_x27(lean_object* v_00_u03b1_746_, lean_object* v_00_u03b2_747_, lean_object* v_00_u03b3_748_, lean_object* v_f_749_, lean_object* v_x_750_, lean_object* v_x_751_){
_start:
{
lean_object* v___x_752_; 
v___x_752_ = lp_mathlib_List_map_u2082Left_x27___redArg(v_f_749_, v_x_750_, v_x_751_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0(lean_object* v_00_u03b1_753_, lean_object* v_00_u03b3_754_, lean_object* v_00_u03b2_755_, lean_object* v_f_756_, lean_object* v_a_757_, lean_object* v_a_758_){
_start:
{
lean_object* v___x_759_; 
v___x_759_ = lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0___redArg(v_f_756_, v_a_757_, v_a_758_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_map_u2082Left_x27_match__1_splitter___redArg(lean_object* v_x_760_, lean_object* v_x_761_, lean_object* v_h__1_762_, lean_object* v_h__2_763_, lean_object* v_h__3_764_){
_start:
{
if (lean_obj_tag(v_x_760_) == 0)
{
lean_object* v___x_765_; 
lean_dec(v_h__3_764_);
lean_dec(v_h__2_763_);
v___x_765_ = lean_apply_1(v_h__1_762_, v_x_761_);
return v___x_765_;
}
else
{
lean_dec(v_h__1_762_);
if (lean_obj_tag(v_x_761_) == 0)
{
lean_object* v_head_766_; lean_object* v_tail_767_; lean_object* v___x_768_; 
lean_dec(v_h__3_764_);
v_head_766_ = lean_ctor_get(v_x_760_, 0);
lean_inc(v_head_766_);
v_tail_767_ = lean_ctor_get(v_x_760_, 1);
lean_inc(v_tail_767_);
lean_dec_ref_known(v_x_760_, 2);
v___x_768_ = lean_apply_2(v_h__2_763_, v_head_766_, v_tail_767_);
return v___x_768_;
}
else
{
lean_object* v_head_769_; lean_object* v_tail_770_; lean_object* v_head_771_; lean_object* v_tail_772_; lean_object* v___x_773_; 
lean_dec(v_h__2_763_);
v_head_769_ = lean_ctor_get(v_x_760_, 0);
lean_inc(v_head_769_);
v_tail_770_ = lean_ctor_get(v_x_760_, 1);
lean_inc(v_tail_770_);
lean_dec_ref_known(v_x_760_, 2);
v_head_771_ = lean_ctor_get(v_x_761_, 0);
lean_inc(v_head_771_);
v_tail_772_ = lean_ctor_get(v_x_761_, 1);
lean_inc(v_tail_772_);
lean_dec_ref_known(v_x_761_, 2);
v___x_773_ = lean_apply_4(v_h__3_764_, v_head_769_, v_tail_770_, v_head_771_, v_tail_772_);
return v___x_773_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_map_u2082Left_x27_match__1_splitter(lean_object* v_00_u03b1_774_, lean_object* v_00_u03b2_775_, lean_object* v_motive_776_, lean_object* v_x_777_, lean_object* v_x_778_, lean_object* v_h__1_779_, lean_object* v_h__2_780_, lean_object* v_h__3_781_){
_start:
{
if (lean_obj_tag(v_x_777_) == 0)
{
lean_object* v___x_782_; 
lean_dec(v_h__3_781_);
lean_dec(v_h__2_780_);
v___x_782_ = lean_apply_1(v_h__1_779_, v_x_778_);
return v___x_782_;
}
else
{
lean_dec(v_h__1_779_);
if (lean_obj_tag(v_x_778_) == 0)
{
lean_object* v_head_783_; lean_object* v_tail_784_; lean_object* v___x_785_; 
lean_dec(v_h__3_781_);
v_head_783_ = lean_ctor_get(v_x_777_, 0);
lean_inc(v_head_783_);
v_tail_784_ = lean_ctor_get(v_x_777_, 1);
lean_inc(v_tail_784_);
lean_dec_ref_known(v_x_777_, 2);
v___x_785_ = lean_apply_2(v_h__2_780_, v_head_783_, v_tail_784_);
return v___x_785_;
}
else
{
lean_object* v_head_786_; lean_object* v_tail_787_; lean_object* v_head_788_; lean_object* v_tail_789_; lean_object* v___x_790_; 
lean_dec(v_h__2_780_);
v_head_786_ = lean_ctor_get(v_x_777_, 0);
lean_inc(v_head_786_);
v_tail_787_ = lean_ctor_get(v_x_777_, 1);
lean_inc(v_tail_787_);
lean_dec_ref_known(v_x_777_, 2);
v_head_788_ = lean_ctor_get(v_x_778_, 0);
lean_inc(v_head_788_);
v_tail_789_ = lean_ctor_get(v_x_778_, 1);
lean_inc(v_tail_789_);
lean_dec_ref_known(v_x_778_, 2);
v___x_790_ = lean_apply_4(v_h__3_781_, v_head_786_, v_tail_787_, v_head_788_, v_tail_789_);
return v___x_790_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27___redArg___lam__0(lean_object* v_f_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v___x_794_; 
v___x_794_ = lean_apply_2(v_f_791_, v___y_793_, v___y_792_);
return v___x_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27___redArg(lean_object* v_f_795_, lean_object* v_as_796_, lean_object* v_bs_797_){
_start:
{
lean_object* v___f_798_; lean_object* v___x_799_; 
v___f_798_ = lean_alloc_closure((void*)(lp_mathlib_List_map_u2082Right_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_798_, 0, v_f_795_);
v___x_799_ = lp_mathlib_List_map_u2082Left_x27___redArg(v___f_798_, v_bs_797_, v_as_796_);
return v___x_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right_x27(lean_object* v_00_u03b1_800_, lean_object* v_00_u03b2_801_, lean_object* v_00_u03b3_802_, lean_object* v_f_803_, lean_object* v_as_804_, lean_object* v_bs_805_){
_start:
{
lean_object* v___x_806_; 
v___x_806_ = lp_mathlib_List_map_u2082Right_x27___redArg(v_f_803_, v_as_804_, v_bs_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left___redArg(lean_object* v_f_807_, lean_object* v_x_808_, lean_object* v_x_809_){
_start:
{
if (lean_obj_tag(v_x_808_) == 0)
{
lean_object* v___x_810_; 
lean_dec(v_x_809_);
lean_dec(v_f_807_);
v___x_810_ = lean_box(0);
return v___x_810_;
}
else
{
if (lean_obj_tag(v_x_809_) == 0)
{
lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_811_ = lean_box(0);
v___x_812_ = lp_mathlib_List_mapTR_loop___at___00List_map_u2082Left_x27_spec__0___redArg(v_f_807_, v_x_808_, v___x_811_);
return v___x_812_;
}
else
{
lean_object* v_head_813_; lean_object* v_tail_814_; lean_object* v_head_815_; lean_object* v_tail_816_; lean_object* v___x_818_; uint8_t v_isShared_819_; uint8_t v_isSharedCheck_826_; 
v_head_813_ = lean_ctor_get(v_x_808_, 0);
lean_inc(v_head_813_);
v_tail_814_ = lean_ctor_get(v_x_808_, 1);
lean_inc(v_tail_814_);
lean_dec_ref_known(v_x_808_, 2);
v_head_815_ = lean_ctor_get(v_x_809_, 0);
v_tail_816_ = lean_ctor_get(v_x_809_, 1);
v_isSharedCheck_826_ = !lean_is_exclusive(v_x_809_);
if (v_isSharedCheck_826_ == 0)
{
v___x_818_ = v_x_809_;
v_isShared_819_ = v_isSharedCheck_826_;
goto v_resetjp_817_;
}
else
{
lean_inc(v_tail_816_);
lean_inc(v_head_815_);
lean_dec(v_x_809_);
v___x_818_ = lean_box(0);
v_isShared_819_ = v_isSharedCheck_826_;
goto v_resetjp_817_;
}
v_resetjp_817_:
{
lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_824_; 
v___x_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_820_, 0, v_head_815_);
lean_inc(v_f_807_);
v___x_821_ = lean_apply_2(v_f_807_, v_head_813_, v___x_820_);
v___x_822_ = lp_mathlib_List_map_u2082Left___redArg(v_f_807_, v_tail_814_, v_tail_816_);
if (v_isShared_819_ == 0)
{
lean_ctor_set(v___x_818_, 1, v___x_822_);
lean_ctor_set(v___x_818_, 0, v___x_821_);
v___x_824_ = v___x_818_;
goto v_reusejp_823_;
}
else
{
lean_object* v_reuseFailAlloc_825_; 
v_reuseFailAlloc_825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_825_, 0, v___x_821_);
lean_ctor_set(v_reuseFailAlloc_825_, 1, v___x_822_);
v___x_824_ = v_reuseFailAlloc_825_;
goto v_reusejp_823_;
}
v_reusejp_823_:
{
return v___x_824_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Left(lean_object* v_00_u03b1_827_, lean_object* v_00_u03b2_828_, lean_object* v_00_u03b3_829_, lean_object* v_f_830_, lean_object* v_x_831_, lean_object* v_x_832_){
_start:
{
lean_object* v___x_833_; 
v___x_833_ = lp_mathlib_List_map_u2082Left___redArg(v_f_830_, v_x_831_, v_x_832_);
return v___x_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right___redArg(lean_object* v_f_834_, lean_object* v_as_835_, lean_object* v_bs_836_){
_start:
{
lean_object* v___f_837_; lean_object* v___x_838_; 
v___f_837_ = lean_alloc_closure((void*)(lp_mathlib_List_map_u2082Right_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_837_, 0, v_f_834_);
v___x_838_ = lp_mathlib_List_map_u2082Left___redArg(v___f_837_, v_bs_836_, v_as_835_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_map_u2082Right(lean_object* v_00_u03b1_839_, lean_object* v_00_u03b2_840_, lean_object* v_00_u03b3_841_, lean_object* v_f_842_, lean_object* v_as_843_, lean_object* v_bs_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_mathlib_List_map_u2082Right___redArg(v_f_842_, v_as_843_, v_bs_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0___redArg(lean_object* v_f_846_, lean_object* v_a_847_, lean_object* v_a_848_){
_start:
{
if (lean_obj_tag(v_a_847_) == 0)
{
lean_object* v___x_849_; 
lean_dec(v_f_846_);
v___x_849_ = l_List_reverse___redArg(v_a_848_);
return v___x_849_;
}
else
{
lean_object* v_head_850_; lean_object* v_tail_851_; lean_object* v___x_853_; uint8_t v_isShared_854_; uint8_t v_isSharedCheck_860_; 
v_head_850_ = lean_ctor_get(v_a_847_, 0);
v_tail_851_ = lean_ctor_get(v_a_847_, 1);
v_isSharedCheck_860_ = !lean_is_exclusive(v_a_847_);
if (v_isSharedCheck_860_ == 0)
{
v___x_853_ = v_a_847_;
v_isShared_854_ = v_isSharedCheck_860_;
goto v_resetjp_852_;
}
else
{
lean_inc(v_tail_851_);
lean_inc(v_head_850_);
lean_dec(v_a_847_);
v___x_853_ = lean_box(0);
v_isShared_854_ = v_isSharedCheck_860_;
goto v_resetjp_852_;
}
v_resetjp_852_:
{
lean_object* v___x_855_; lean_object* v___x_857_; 
lean_inc(v_f_846_);
v___x_855_ = lean_apply_1(v_f_846_, v_head_850_);
if (v_isShared_854_ == 0)
{
lean_ctor_set(v___x_853_, 1, v_a_848_);
lean_ctor_set(v___x_853_, 0, v___x_855_);
v___x_857_ = v___x_853_;
goto v_reusejp_856_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v___x_855_);
lean_ctor_set(v_reuseFailAlloc_859_, 1, v_a_848_);
v___x_857_ = v_reuseFailAlloc_859_;
goto v_reusejp_856_;
}
v_reusejp_856_:
{
v_a_847_ = v_tail_851_;
v_a_848_ = v___x_857_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg___lam__0(lean_object* v_f_861_, lean_object* v_head_862_, lean_object* v_x_863_){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; 
v___x_864_ = lean_box(0);
v___x_865_ = lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0___redArg(v_f_861_, v_head_862_, v___x_864_);
return v___x_865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg(lean_object* v_f_866_, lean_object* v_a_867_, lean_object* v_a_868_){
_start:
{
if (lean_obj_tag(v_a_867_) == 0)
{
lean_object* v___x_869_; 
lean_dec(v_f_866_);
v___x_869_ = l_List_reverse___redArg(v_a_868_);
return v___x_869_;
}
else
{
lean_object* v_head_870_; lean_object* v_tail_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_882_; 
v_head_870_ = lean_ctor_get(v_a_867_, 0);
v_tail_871_ = lean_ctor_get(v_a_867_, 1);
v_isSharedCheck_882_ = !lean_is_exclusive(v_a_867_);
if (v_isSharedCheck_882_ == 0)
{
v___x_873_ = v_a_867_;
v_isShared_874_ = v_isSharedCheck_882_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_tail_871_);
lean_inc(v_head_870_);
lean_dec(v_a_867_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_882_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___f_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_879_; 
lean_inc(v_f_866_);
v___f_875_ = lean_alloc_closure((void*)(lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg___lam__0), 3, 2);
lean_closure_set(v___f_875_, 0, v_f_866_);
lean_closure_set(v___f_875_, 1, v_head_870_);
v___x_876_ = lean_unsigned_to_nat(0u);
v___x_877_ = lean_task_spawn(v___f_875_, v___x_876_);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v_a_868_);
lean_ctor_set(v___x_873_, 0, v___x_877_);
v___x_879_ = v___x_873_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v___x_877_);
lean_ctor_set(v_reuseFailAlloc_881_, 1, v_a_868_);
v___x_879_ = v_reuseFailAlloc_881_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
v_a_867_ = v_tail_871_;
v_a_868_ = v___x_879_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2___redArg(lean_object* v_a_883_, lean_object* v_a_884_){
_start:
{
if (lean_obj_tag(v_a_883_) == 0)
{
lean_object* v___x_885_; 
v___x_885_ = lean_array_to_list(v_a_884_);
return v___x_885_;
}
else
{
lean_object* v_head_886_; lean_object* v_tail_887_; lean_object* v___x_888_; lean_object* v___x_889_; 
v_head_886_ = lean_ctor_get(v_a_883_, 0);
lean_inc(v_head_886_);
v_tail_887_ = lean_ctor_get(v_a_883_, 1);
lean_inc(v_tail_887_);
lean_dec_ref_known(v_a_883_, 2);
v___x_888_ = lean_task_get_own(v_head_886_);
v___x_889_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_884_, v___x_888_);
v_a_883_ = v_tail_887_;
v_a_884_ = v___x_889_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___redArg(lean_object* v_f_893_, lean_object* v_xs_894_, lean_object* v_chunk__size_895_){
_start:
{
lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; 
v___x_896_ = lp_batteries_List_toChunks___redArg(v_chunk__size_895_, v_xs_894_);
v___x_897_ = lean_box(0);
v___x_898_ = lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg(v_f_893_, v___x_896_, v___x_897_);
v___x_899_ = ((lean_object*)(lp_mathlib_List_mapAsyncChunked___redArg___closed__0));
v___x_900_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2___redArg(v___x_898_, v___x_899_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___redArg___boxed(lean_object* v_f_901_, lean_object* v_xs_902_, lean_object* v_chunk__size_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_mathlib_List_mapAsyncChunked___redArg(v_f_901_, v_xs_902_, v_chunk__size_903_);
lean_dec(v_chunk__size_903_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked(lean_object* v_00_u03b1_905_, lean_object* v_00_u03b2_906_, lean_object* v_f_907_, lean_object* v_xs_908_, lean_object* v_chunk__size_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lp_mathlib_List_mapAsyncChunked___redArg(v_f_907_, v_xs_908_, v_chunk__size_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAsyncChunked___boxed(lean_object* v_00_u03b1_911_, lean_object* v_00_u03b2_912_, lean_object* v_f_913_, lean_object* v_xs_914_, lean_object* v_chunk__size_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_mathlib_List_mapAsyncChunked(v_00_u03b1_911_, v_00_u03b2_912_, v_f_913_, v_xs_914_, v_chunk__size_915_);
lean_dec(v_chunk__size_915_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0(lean_object* v_00_u03b1_917_, lean_object* v_00_u03b2_918_, lean_object* v_f_919_, lean_object* v_a_920_, lean_object* v_a_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__0___redArg(v_f_919_, v_a_920_, v_a_921_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1(lean_object* v_00_u03b1_923_, lean_object* v_00_u03b2_924_, lean_object* v_f_925_, lean_object* v_a_926_, lean_object* v_a_927_){
_start:
{
lean_object* v___x_928_; 
v___x_928_ = lp_mathlib_List_mapTR_loop___at___00List_mapAsyncChunked_spec__1___redArg(v_f_925_, v_a_926_, v_a_927_);
return v___x_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2(lean_object* v_00_u03b2_929_, lean_object* v_a_930_, lean_object* v_a_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_mapAsyncChunked_spec__2___redArg(v_a_930_, v_a_931_);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith3___redArg(lean_object* v_f_933_, lean_object* v_x_934_, lean_object* v_x_935_, lean_object* v_x_936_){
_start:
{
if (lean_obj_tag(v_x_934_) == 1)
{
if (lean_obj_tag(v_x_935_) == 1)
{
if (lean_obj_tag(v_x_936_) == 1)
{
lean_object* v_head_937_; lean_object* v_tail_938_; lean_object* v_head_939_; lean_object* v_tail_940_; lean_object* v_head_941_; lean_object* v_tail_942_; lean_object* v___x_944_; uint8_t v_isShared_945_; uint8_t v_isSharedCheck_951_; 
v_head_937_ = lean_ctor_get(v_x_934_, 0);
lean_inc(v_head_937_);
v_tail_938_ = lean_ctor_get(v_x_934_, 1);
lean_inc(v_tail_938_);
lean_dec_ref_known(v_x_934_, 2);
v_head_939_ = lean_ctor_get(v_x_935_, 0);
lean_inc(v_head_939_);
v_tail_940_ = lean_ctor_get(v_x_935_, 1);
lean_inc(v_tail_940_);
lean_dec_ref_known(v_x_935_, 2);
v_head_941_ = lean_ctor_get(v_x_936_, 0);
v_tail_942_ = lean_ctor_get(v_x_936_, 1);
v_isSharedCheck_951_ = !lean_is_exclusive(v_x_936_);
if (v_isSharedCheck_951_ == 0)
{
v___x_944_ = v_x_936_;
v_isShared_945_ = v_isSharedCheck_951_;
goto v_resetjp_943_;
}
else
{
lean_inc(v_tail_942_);
lean_inc(v_head_941_);
lean_dec(v_x_936_);
v___x_944_ = lean_box(0);
v_isShared_945_ = v_isSharedCheck_951_;
goto v_resetjp_943_;
}
v_resetjp_943_:
{
lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_949_; 
lean_inc(v_f_933_);
v___x_946_ = lean_apply_3(v_f_933_, v_head_937_, v_head_939_, v_head_941_);
v___x_947_ = lp_mathlib_List_zipWith3___redArg(v_f_933_, v_tail_938_, v_tail_940_, v_tail_942_);
if (v_isShared_945_ == 0)
{
lean_ctor_set(v___x_944_, 1, v___x_947_);
lean_ctor_set(v___x_944_, 0, v___x_946_);
v___x_949_ = v___x_944_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v___x_946_);
lean_ctor_set(v_reuseFailAlloc_950_, 1, v___x_947_);
v___x_949_ = v_reuseFailAlloc_950_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
return v___x_949_;
}
}
}
else
{
lean_object* v___x_952_; 
lean_dec_ref_known(v_x_935_, 2);
lean_dec_ref_known(v_x_934_, 2);
lean_dec(v_x_936_);
lean_dec(v_f_933_);
v___x_952_ = lean_box(0);
return v___x_952_;
}
}
else
{
lean_object* v___x_953_; 
lean_dec_ref_known(v_x_934_, 2);
lean_dec(v_x_936_);
lean_dec(v_x_935_);
lean_dec(v_f_933_);
v___x_953_ = lean_box(0);
return v___x_953_;
}
}
else
{
lean_object* v___x_954_; 
lean_dec(v_x_936_);
lean_dec(v_x_935_);
lean_dec(v_x_934_);
lean_dec(v_f_933_);
v___x_954_ = lean_box(0);
return v___x_954_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith3(lean_object* v_00_u03b1_955_, lean_object* v_00_u03b2_956_, lean_object* v_00_u03b3_957_, lean_object* v_00_u03b4_958_, lean_object* v_f_959_, lean_object* v_x_960_, lean_object* v_x_961_, lean_object* v_x_962_){
_start:
{
lean_object* v___x_963_; 
v___x_963_ = lp_mathlib_List_zipWith3___redArg(v_f_959_, v_x_960_, v_x_961_, v_x_962_);
return v___x_963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith4___redArg(lean_object* v_f_964_, lean_object* v_x_965_, lean_object* v_x_966_, lean_object* v_x_967_, lean_object* v_x_968_){
_start:
{
if (lean_obj_tag(v_x_965_) == 1)
{
if (lean_obj_tag(v_x_966_) == 1)
{
if (lean_obj_tag(v_x_967_) == 1)
{
if (lean_obj_tag(v_x_968_) == 1)
{
lean_object* v_head_969_; lean_object* v_tail_970_; lean_object* v_head_971_; lean_object* v_tail_972_; lean_object* v_head_973_; lean_object* v_tail_974_; lean_object* v_head_975_; lean_object* v_tail_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_985_; 
v_head_969_ = lean_ctor_get(v_x_965_, 0);
lean_inc(v_head_969_);
v_tail_970_ = lean_ctor_get(v_x_965_, 1);
lean_inc(v_tail_970_);
lean_dec_ref_known(v_x_965_, 2);
v_head_971_ = lean_ctor_get(v_x_966_, 0);
lean_inc(v_head_971_);
v_tail_972_ = lean_ctor_get(v_x_966_, 1);
lean_inc(v_tail_972_);
lean_dec_ref_known(v_x_966_, 2);
v_head_973_ = lean_ctor_get(v_x_967_, 0);
lean_inc(v_head_973_);
v_tail_974_ = lean_ctor_get(v_x_967_, 1);
lean_inc(v_tail_974_);
lean_dec_ref_known(v_x_967_, 2);
v_head_975_ = lean_ctor_get(v_x_968_, 0);
v_tail_976_ = lean_ctor_get(v_x_968_, 1);
v_isSharedCheck_985_ = !lean_is_exclusive(v_x_968_);
if (v_isSharedCheck_985_ == 0)
{
v___x_978_ = v_x_968_;
v_isShared_979_ = v_isSharedCheck_985_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_tail_976_);
lean_inc(v_head_975_);
lean_dec(v_x_968_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_985_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_983_; 
lean_inc(v_f_964_);
v___x_980_ = lean_apply_4(v_f_964_, v_head_969_, v_head_971_, v_head_973_, v_head_975_);
v___x_981_ = lp_mathlib_List_zipWith4___redArg(v_f_964_, v_tail_970_, v_tail_972_, v_tail_974_, v_tail_976_);
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 1, v___x_981_);
lean_ctor_set(v___x_978_, 0, v___x_980_);
v___x_983_ = v___x_978_;
goto v_reusejp_982_;
}
else
{
lean_object* v_reuseFailAlloc_984_; 
v_reuseFailAlloc_984_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_984_, 0, v___x_980_);
lean_ctor_set(v_reuseFailAlloc_984_, 1, v___x_981_);
v___x_983_ = v_reuseFailAlloc_984_;
goto v_reusejp_982_;
}
v_reusejp_982_:
{
return v___x_983_;
}
}
}
else
{
lean_object* v___x_986_; 
lean_dec_ref_known(v_x_967_, 2);
lean_dec_ref_known(v_x_966_, 2);
lean_dec_ref_known(v_x_965_, 2);
lean_dec(v_x_968_);
lean_dec(v_f_964_);
v___x_986_ = lean_box(0);
return v___x_986_;
}
}
else
{
lean_object* v___x_987_; 
lean_dec_ref_known(v_x_966_, 2);
lean_dec_ref_known(v_x_965_, 2);
lean_dec(v_x_968_);
lean_dec(v_x_967_);
lean_dec(v_f_964_);
v___x_987_ = lean_box(0);
return v___x_987_;
}
}
else
{
lean_object* v___x_988_; 
lean_dec_ref_known(v_x_965_, 2);
lean_dec(v_x_968_);
lean_dec(v_x_967_);
lean_dec(v_x_966_);
lean_dec(v_f_964_);
v___x_988_ = lean_box(0);
return v___x_988_;
}
}
else
{
lean_object* v___x_989_; 
lean_dec(v_x_968_);
lean_dec(v_x_967_);
lean_dec(v_x_966_);
lean_dec(v_x_965_);
lean_dec(v_f_964_);
v___x_989_ = lean_box(0);
return v___x_989_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith4(lean_object* v_00_u03b1_990_, lean_object* v_00_u03b2_991_, lean_object* v_00_u03b3_992_, lean_object* v_00_u03b4_993_, lean_object* v_00_u03b5_994_, lean_object* v_f_995_, lean_object* v_x_996_, lean_object* v_x_997_, lean_object* v_x_998_, lean_object* v_x_999_){
_start:
{
lean_object* v___x_1000_; 
v___x_1000_ = lp_mathlib_List_zipWith4___redArg(v_f_995_, v_x_996_, v_x_997_, v_x_998_, v_x_999_);
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith5___redArg(lean_object* v_f_1001_, lean_object* v_x_1002_, lean_object* v_x_1003_, lean_object* v_x_1004_, lean_object* v_x_1005_, lean_object* v_x_1006_){
_start:
{
if (lean_obj_tag(v_x_1002_) == 1)
{
if (lean_obj_tag(v_x_1003_) == 1)
{
if (lean_obj_tag(v_x_1004_) == 1)
{
if (lean_obj_tag(v_x_1005_) == 1)
{
if (lean_obj_tag(v_x_1006_) == 1)
{
lean_object* v_head_1007_; lean_object* v_tail_1008_; lean_object* v_head_1009_; lean_object* v_tail_1010_; lean_object* v_head_1011_; lean_object* v_tail_1012_; lean_object* v_head_1013_; lean_object* v_tail_1014_; lean_object* v_head_1015_; lean_object* v_tail_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1025_; 
v_head_1007_ = lean_ctor_get(v_x_1002_, 0);
lean_inc(v_head_1007_);
v_tail_1008_ = lean_ctor_get(v_x_1002_, 1);
lean_inc(v_tail_1008_);
lean_dec_ref_known(v_x_1002_, 2);
v_head_1009_ = lean_ctor_get(v_x_1003_, 0);
lean_inc(v_head_1009_);
v_tail_1010_ = lean_ctor_get(v_x_1003_, 1);
lean_inc(v_tail_1010_);
lean_dec_ref_known(v_x_1003_, 2);
v_head_1011_ = lean_ctor_get(v_x_1004_, 0);
lean_inc(v_head_1011_);
v_tail_1012_ = lean_ctor_get(v_x_1004_, 1);
lean_inc(v_tail_1012_);
lean_dec_ref_known(v_x_1004_, 2);
v_head_1013_ = lean_ctor_get(v_x_1005_, 0);
lean_inc(v_head_1013_);
v_tail_1014_ = lean_ctor_get(v_x_1005_, 1);
lean_inc(v_tail_1014_);
lean_dec_ref_known(v_x_1005_, 2);
v_head_1015_ = lean_ctor_get(v_x_1006_, 0);
v_tail_1016_ = lean_ctor_get(v_x_1006_, 1);
v_isSharedCheck_1025_ = !lean_is_exclusive(v_x_1006_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1018_ = v_x_1006_;
v_isShared_1019_ = v_isSharedCheck_1025_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_tail_1016_);
lean_inc(v_head_1015_);
lean_dec(v_x_1006_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1025_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1023_; 
lean_inc(v_f_1001_);
v___x_1020_ = lean_apply_5(v_f_1001_, v_head_1007_, v_head_1009_, v_head_1011_, v_head_1013_, v_head_1015_);
v___x_1021_ = lp_mathlib_List_zipWith5___redArg(v_f_1001_, v_tail_1008_, v_tail_1010_, v_tail_1012_, v_tail_1014_, v_tail_1016_);
if (v_isShared_1019_ == 0)
{
lean_ctor_set(v___x_1018_, 1, v___x_1021_);
lean_ctor_set(v___x_1018_, 0, v___x_1020_);
v___x_1023_ = v___x_1018_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v___x_1020_);
lean_ctor_set(v_reuseFailAlloc_1024_, 1, v___x_1021_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
}
else
{
lean_object* v___x_1026_; 
lean_dec_ref_known(v_x_1005_, 2);
lean_dec_ref_known(v_x_1004_, 2);
lean_dec_ref_known(v_x_1003_, 2);
lean_dec_ref_known(v_x_1002_, 2);
lean_dec(v_x_1006_);
lean_dec(v_f_1001_);
v___x_1026_ = lean_box(0);
return v___x_1026_;
}
}
else
{
lean_object* v___x_1027_; 
lean_dec_ref_known(v_x_1004_, 2);
lean_dec_ref_known(v_x_1003_, 2);
lean_dec_ref_known(v_x_1002_, 2);
lean_dec(v_x_1006_);
lean_dec(v_x_1005_);
lean_dec(v_f_1001_);
v___x_1027_ = lean_box(0);
return v___x_1027_;
}
}
else
{
lean_object* v___x_1028_; 
lean_dec_ref_known(v_x_1003_, 2);
lean_dec_ref_known(v_x_1002_, 2);
lean_dec(v_x_1006_);
lean_dec(v_x_1005_);
lean_dec(v_x_1004_);
lean_dec(v_f_1001_);
v___x_1028_ = lean_box(0);
return v___x_1028_;
}
}
else
{
lean_object* v___x_1029_; 
lean_dec_ref_known(v_x_1002_, 2);
lean_dec(v_x_1006_);
lean_dec(v_x_1005_);
lean_dec(v_x_1004_);
lean_dec(v_x_1003_);
lean_dec(v_f_1001_);
v___x_1029_ = lean_box(0);
return v___x_1029_;
}
}
else
{
lean_object* v___x_1030_; 
lean_dec(v_x_1006_);
lean_dec(v_x_1005_);
lean_dec(v_x_1004_);
lean_dec(v_x_1003_);
lean_dec(v_x_1002_);
lean_dec(v_f_1001_);
v___x_1030_ = lean_box(0);
return v___x_1030_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_zipWith5(lean_object* v_00_u03b1_1031_, lean_object* v_00_u03b2_1032_, lean_object* v_00_u03b3_1033_, lean_object* v_00_u03b4_1034_, lean_object* v_00_u03b5_1035_, lean_object* v_00_u03b6_1036_, lean_object* v_f_1037_, lean_object* v_x_1038_, lean_object* v_x_1039_, lean_object* v_x_1040_, lean_object* v_x_1041_, lean_object* v_x_1042_){
_start:
{
lean_object* v___x_1043_; 
v___x_1043_ = lp_mathlib_List_zipWith5___redArg(v_f_1037_, v_x_1038_, v_x_1039_, v_x_1040_, v_x_1041_, v_x_1042_);
return v___x_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___redArg(lean_object* v_x_1044_, lean_object* v_x_1045_, lean_object* v_x_1046_){
_start:
{
if (lean_obj_tag(v_x_1046_) == 0)
{
lean_dec(v_x_1045_);
lean_inc(v_x_1044_);
return v_x_1044_;
}
else
{
if (lean_obj_tag(v_x_1044_) == 0)
{
lean_dec(v_x_1045_);
return v_x_1044_;
}
else
{
if (lean_obj_tag(v_x_1045_) == 0)
{
lean_inc_ref(v_x_1044_);
return v_x_1044_;
}
else
{
lean_object* v_head_1047_; uint8_t v___x_1048_; 
v_head_1047_ = lean_ctor_get(v_x_1045_, 0);
v___x_1048_ = lean_unbox(v_head_1047_);
if (v___x_1048_ == 0)
{
lean_object* v_head_1049_; lean_object* v_tail_1050_; lean_object* v_tail_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1059_; 
v_head_1049_ = lean_ctor_get(v_x_1044_, 0);
v_tail_1050_ = lean_ctor_get(v_x_1044_, 1);
v_tail_1051_ = lean_ctor_get(v_x_1045_, 1);
v_isSharedCheck_1059_ = !lean_is_exclusive(v_x_1045_);
if (v_isSharedCheck_1059_ == 0)
{
lean_object* v_unused_1060_; 
v_unused_1060_ = lean_ctor_get(v_x_1045_, 0);
lean_dec(v_unused_1060_);
v___x_1053_ = v_x_1045_;
v_isShared_1054_ = v_isSharedCheck_1059_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_tail_1051_);
lean_dec(v_x_1045_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1059_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1055_; lean_object* v___x_1057_; 
v___x_1055_ = lp_mathlib_List_replaceIf___redArg(v_tail_1050_, v_tail_1051_, v_x_1046_);
lean_inc(v_head_1049_);
if (v_isShared_1054_ == 0)
{
lean_ctor_set(v___x_1053_, 1, v___x_1055_);
lean_ctor_set(v___x_1053_, 0, v_head_1049_);
v___x_1057_ = v___x_1053_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v_head_1049_);
lean_ctor_set(v_reuseFailAlloc_1058_, 1, v___x_1055_);
v___x_1057_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
return v___x_1057_;
}
}
}
else
{
lean_object* v_head_1061_; lean_object* v_tail_1062_; lean_object* v_tail_1063_; lean_object* v_tail_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1072_; 
v_head_1061_ = lean_ctor_get(v_x_1046_, 0);
v_tail_1062_ = lean_ctor_get(v_x_1046_, 1);
v_tail_1063_ = lean_ctor_get(v_x_1044_, 1);
v_tail_1064_ = lean_ctor_get(v_x_1045_, 1);
v_isSharedCheck_1072_ = !lean_is_exclusive(v_x_1045_);
if (v_isSharedCheck_1072_ == 0)
{
lean_object* v_unused_1073_; 
v_unused_1073_ = lean_ctor_get(v_x_1045_, 0);
lean_dec(v_unused_1073_);
v___x_1066_ = v_x_1045_;
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_tail_1064_);
lean_dec(v_x_1045_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1068_; lean_object* v___x_1070_; 
v___x_1068_ = lp_mathlib_List_replaceIf___redArg(v_tail_1063_, v_tail_1064_, v_tail_1062_);
lean_inc(v_head_1061_);
if (v_isShared_1067_ == 0)
{
lean_ctor_set(v___x_1066_, 1, v___x_1068_);
lean_ctor_set(v___x_1066_, 0, v_head_1061_);
v___x_1070_ = v___x_1066_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v_head_1061_);
lean_ctor_set(v_reuseFailAlloc_1071_, 1, v___x_1068_);
v___x_1070_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
return v___x_1070_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___redArg___boxed(lean_object* v_x_1074_, lean_object* v_x_1075_, lean_object* v_x_1076_){
_start:
{
lean_object* v_res_1077_; 
v_res_1077_ = lp_mathlib_List_replaceIf___redArg(v_x_1074_, v_x_1075_, v_x_1076_);
lean_dec(v_x_1076_);
lean_dec(v_x_1074_);
return v_res_1077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf(lean_object* v_00_u03b1_1078_, lean_object* v_x_1079_, lean_object* v_x_1080_, lean_object* v_x_1081_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_mathlib_List_replaceIf___redArg(v_x_1079_, v_x_1080_, v_x_1081_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_replaceIf___boxed(lean_object* v_00_u03b1_1083_, lean_object* v_x_1084_, lean_object* v_x_1085_, lean_object* v_x_1086_){
_start:
{
lean_object* v_res_1087_; 
v_res_1087_ = lp_mathlib_List_replaceIf(v_00_u03b1_1083_, v_x_1084_, v_x_1085_, v_x_1086_);
lean_dec(v_x_1086_);
lean_dec(v_x_1084_);
return v_res_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___redArg(lean_object* v_f_1088_, lean_object* v_a_1089_, lean_object* v_x_1090_){
_start:
{
lean_object* v_zero_1091_; uint8_t v_isZero_1092_; 
v_zero_1091_ = lean_unsigned_to_nat(0u);
v_isZero_1092_ = lean_nat_dec_eq(v_x_1090_, v_zero_1091_);
if (v_isZero_1092_ == 1)
{
lean_object* v___x_1093_; 
lean_dec(v_a_1089_);
lean_dec(v_f_1088_);
v___x_1093_ = lean_box(0);
return v___x_1093_;
}
else
{
lean_object* v_one_1094_; lean_object* v_n_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; 
v_one_1094_ = lean_unsigned_to_nat(1u);
v_n_1095_ = lean_nat_sub(v_x_1090_, v_one_1094_);
lean_inc(v_f_1088_);
lean_inc(v_a_1089_);
v___x_1096_ = lean_apply_1(v_f_1088_, v_a_1089_);
v___x_1097_ = lp_mathlib_List_iterate___redArg(v_f_1088_, v___x_1096_, v_n_1095_);
lean_dec(v_n_1095_);
v___x_1098_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1098_, 0, v_a_1089_);
lean_ctor_set(v___x_1098_, 1, v___x_1097_);
return v___x_1098_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___redArg___boxed(lean_object* v_f_1099_, lean_object* v_a_1100_, lean_object* v_x_1101_){
_start:
{
lean_object* v_res_1102_; 
v_res_1102_ = lp_mathlib_List_iterate___redArg(v_f_1099_, v_a_1100_, v_x_1101_);
lean_dec(v_x_1101_);
return v_res_1102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterate(lean_object* v_00_u03b1_1103_, lean_object* v_f_1104_, lean_object* v_a_1105_, lean_object* v_x_1106_){
_start:
{
lean_object* v___x_1107_; 
v___x_1107_ = lp_mathlib_List_iterate___redArg(v_f_1104_, v_a_1105_, v_x_1106_);
return v___x_1107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterate___boxed(lean_object* v_00_u03b1_1108_, lean_object* v_f_1109_, lean_object* v_a_1110_, lean_object* v_x_1111_){
_start:
{
lean_object* v_res_1112_; 
v_res_1112_ = lp_mathlib_List_iterate(v_00_u03b1_1108_, v_f_1109_, v_a_1110_, v_x_1111_);
lean_dec(v_x_1111_);
return v_res_1112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___redArg(lean_object* v_x_1113_, lean_object* v_h__1_1114_, lean_object* v_h__2_1115_){
_start:
{
lean_object* v_zero_1116_; uint8_t v_isZero_1117_; 
v_zero_1116_ = lean_unsigned_to_nat(0u);
v_isZero_1117_ = lean_nat_dec_eq(v_x_1113_, v_zero_1116_);
if (v_isZero_1117_ == 1)
{
lean_object* v___x_1118_; lean_object* v___x_1119_; 
lean_dec(v_h__2_1115_);
v___x_1118_ = lean_box(0);
v___x_1119_ = lean_apply_1(v_h__1_1114_, v___x_1118_);
return v___x_1119_;
}
else
{
lean_object* v_one_1120_; lean_object* v_n_1121_; lean_object* v___x_1122_; 
lean_dec(v_h__1_1114_);
v_one_1120_ = lean_unsigned_to_nat(1u);
v_n_1121_ = lean_nat_sub(v_x_1113_, v_one_1120_);
v___x_1122_ = lean_apply_1(v_h__2_1115_, v_n_1121_);
return v___x_1122_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___redArg___boxed(lean_object* v_x_1123_, lean_object* v_h__1_1124_, lean_object* v_h__2_1125_){
_start:
{
lean_object* v_res_1126_; 
v_res_1126_ = lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___redArg(v_x_1123_, v_h__1_1124_, v_h__2_1125_);
lean_dec(v_x_1123_);
return v_res_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter(lean_object* v_motive_1127_, lean_object* v_x_1128_, lean_object* v_h__1_1129_, lean_object* v_h__2_1130_){
_start:
{
lean_object* v_zero_1131_; uint8_t v_isZero_1132_; 
v_zero_1131_ = lean_unsigned_to_nat(0u);
v_isZero_1132_ = lean_nat_dec_eq(v_x_1128_, v_zero_1131_);
if (v_isZero_1132_ == 1)
{
lean_object* v___x_1133_; lean_object* v___x_1134_; 
lean_dec(v_h__2_1130_);
v___x_1133_ = lean_box(0);
v___x_1134_ = lean_apply_1(v_h__1_1129_, v___x_1133_);
return v___x_1134_;
}
else
{
lean_object* v_one_1135_; lean_object* v_n_1136_; lean_object* v___x_1137_; 
lean_dec(v_h__1_1129_);
v_one_1135_ = lean_unsigned_to_nat(1u);
v_n_1136_ = lean_nat_sub(v_x_1128_, v_one_1135_);
v___x_1137_ = lean_apply_1(v_h__2_1130_, v_n_1136_);
return v___x_1137_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter___boxed(lean_object* v_motive_1138_, lean_object* v_x_1139_, lean_object* v_h__1_1140_, lean_object* v_h__2_1141_){
_start:
{
lean_object* v_res_1142_; 
v_res_1142_ = lp_mathlib___private_Mathlib_Data_List_Defs_0__List_iterate_match__1_splitter(v_motive_1138_, v_x_1139_, v_h__1_1140_, v_h__2_1141_);
lean_dec(v_x_1139_);
return v_res_1142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR_loop___redArg(lean_object* v_f_1143_, lean_object* v_a_1144_, lean_object* v_n_1145_, lean_object* v_l_1146_){
_start:
{
lean_object* v_zero_1147_; uint8_t v_isZero_1148_; 
v_zero_1147_ = lean_unsigned_to_nat(0u);
v_isZero_1148_ = lean_nat_dec_eq(v_n_1145_, v_zero_1147_);
if (v_isZero_1148_ == 1)
{
lean_object* v___x_1149_; 
lean_dec(v_n_1145_);
lean_dec(v_a_1144_);
lean_dec(v_f_1143_);
v___x_1149_ = l_List_reverse___redArg(v_l_1146_);
return v___x_1149_;
}
else
{
lean_object* v_one_1150_; lean_object* v_n_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v_one_1150_ = lean_unsigned_to_nat(1u);
v_n_1151_ = lean_nat_sub(v_n_1145_, v_one_1150_);
lean_dec(v_n_1145_);
lean_inc(v_f_1143_);
lean_inc(v_a_1144_);
v___x_1152_ = lean_apply_1(v_f_1143_, v_a_1144_);
v___x_1153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1153_, 0, v_a_1144_);
lean_ctor_set(v___x_1153_, 1, v_l_1146_);
v_a_1144_ = v___x_1152_;
v_n_1145_ = v_n_1151_;
v_l_1146_ = v___x_1153_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR_loop(lean_object* v_00_u03b1_1155_, lean_object* v_f_1156_, lean_object* v_a_1157_, lean_object* v_n_1158_, lean_object* v_l_1159_){
_start:
{
lean_object* v___x_1160_; 
v___x_1160_ = lp_mathlib_List_iterateTR_loop___redArg(v_f_1156_, v_a_1157_, v_n_1158_, v_l_1159_);
return v___x_1160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR___redArg(lean_object* v_f_1161_, lean_object* v_a_1162_, lean_object* v_n_1163_){
_start:
{
lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1164_ = lean_box(0);
v___x_1165_ = lp_mathlib_List_iterateTR_loop___redArg(v_f_1161_, v_a_1162_, v_n_1163_, v___x_1164_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_iterateTR(lean_object* v_00_u03b1_1166_, lean_object* v_f_1167_, lean_object* v_a_1168_, lean_object* v_n_1169_){
_start:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1170_ = lean_box(0);
v___x_1171_ = lp_mathlib_List_iterateTR_loop___redArg(v_f_1167_, v_a_1168_, v_n_1169_, v___x_1170_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr___redArg(lean_object* v_f_1172_, lean_object* v_x_1173_, lean_object* v_x_1174_){
_start:
{
if (lean_obj_tag(v_x_1173_) == 0)
{
lean_object* v___x_1175_; lean_object* v___x_1176_; 
lean_dec_ref(v_f_1172_);
v___x_1175_ = lean_box(0);
v___x_1176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1176_, 0, v_x_1174_);
lean_ctor_set(v___x_1176_, 1, v___x_1175_);
return v___x_1176_;
}
else
{
lean_object* v_head_1177_; lean_object* v_tail_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1198_; 
v_head_1177_ = lean_ctor_get(v_x_1173_, 0);
v_tail_1178_ = lean_ctor_get(v_x_1173_, 1);
v_isSharedCheck_1198_ = !lean_is_exclusive(v_x_1173_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1180_ = v_x_1173_;
v_isShared_1181_ = v_isSharedCheck_1198_;
goto v_resetjp_1179_;
}
else
{
lean_inc(v_tail_1178_);
lean_inc(v_head_1177_);
lean_dec(v_x_1173_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1198_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v_r_1182_; lean_object* v_fst_1183_; lean_object* v_snd_1184_; lean_object* v_z_1185_; lean_object* v_fst_1186_; lean_object* v_snd_1187_; lean_object* v___x_1189_; uint8_t v_isShared_1190_; uint8_t v_isSharedCheck_1197_; 
lean_inc_ref(v_f_1172_);
v_r_1182_ = lp_mathlib_List_mapAccumr___redArg(v_f_1172_, v_tail_1178_, v_x_1174_);
v_fst_1183_ = lean_ctor_get(v_r_1182_, 0);
lean_inc(v_fst_1183_);
v_snd_1184_ = lean_ctor_get(v_r_1182_, 1);
lean_inc(v_snd_1184_);
lean_dec_ref(v_r_1182_);
v_z_1185_ = lean_apply_2(v_f_1172_, v_head_1177_, v_fst_1183_);
v_fst_1186_ = lean_ctor_get(v_z_1185_, 0);
v_snd_1187_ = lean_ctor_get(v_z_1185_, 1);
v_isSharedCheck_1197_ = !lean_is_exclusive(v_z_1185_);
if (v_isSharedCheck_1197_ == 0)
{
v___x_1189_ = v_z_1185_;
v_isShared_1190_ = v_isSharedCheck_1197_;
goto v_resetjp_1188_;
}
else
{
lean_inc(v_snd_1187_);
lean_inc(v_fst_1186_);
lean_dec(v_z_1185_);
v___x_1189_ = lean_box(0);
v_isShared_1190_ = v_isSharedCheck_1197_;
goto v_resetjp_1188_;
}
v_resetjp_1188_:
{
lean_object* v___x_1192_; 
if (v_isShared_1181_ == 0)
{
lean_ctor_set(v___x_1180_, 1, v_snd_1184_);
lean_ctor_set(v___x_1180_, 0, v_snd_1187_);
v___x_1192_ = v___x_1180_;
goto v_reusejp_1191_;
}
else
{
lean_object* v_reuseFailAlloc_1196_; 
v_reuseFailAlloc_1196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1196_, 0, v_snd_1187_);
lean_ctor_set(v_reuseFailAlloc_1196_, 1, v_snd_1184_);
v___x_1192_ = v_reuseFailAlloc_1196_;
goto v_reusejp_1191_;
}
v_reusejp_1191_:
{
lean_object* v___x_1194_; 
if (v_isShared_1190_ == 0)
{
lean_ctor_set(v___x_1189_, 1, v___x_1192_);
v___x_1194_ = v___x_1189_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_fst_1186_);
lean_ctor_set(v_reuseFailAlloc_1195_, 1, v___x_1192_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr(lean_object* v_00_u03b1_1199_, lean_object* v_00_u03b2_1200_, lean_object* v_00_u03b3_1201_, lean_object* v_f_1202_, lean_object* v_x_1203_, lean_object* v_x_1204_){
_start:
{
lean_object* v___x_1205_; 
v___x_1205_ = lp_mathlib_List_mapAccumr___redArg(v_f_1202_, v_x_1203_, v_x_1204_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr_u2082___redArg(lean_object* v_f_1206_, lean_object* v_x_1207_, lean_object* v_x_1208_, lean_object* v_x_1209_){
_start:
{
if (lean_obj_tag(v_x_1207_) == 0)
{
lean_object* v___x_1210_; lean_object* v___x_1211_; 
lean_dec(v_x_1208_);
lean_dec_ref(v_f_1206_);
v___x_1210_ = lean_box(0);
v___x_1211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1211_, 0, v_x_1209_);
lean_ctor_set(v___x_1211_, 1, v___x_1210_);
return v___x_1211_;
}
else
{
if (lean_obj_tag(v_x_1208_) == 0)
{
lean_object* v___x_1213_; uint8_t v_isShared_1214_; uint8_t v_isSharedCheck_1219_; 
lean_dec_ref(v_f_1206_);
v_isSharedCheck_1219_ = !lean_is_exclusive(v_x_1207_);
if (v_isSharedCheck_1219_ == 0)
{
lean_object* v_unused_1220_; lean_object* v_unused_1221_; 
v_unused_1220_ = lean_ctor_get(v_x_1207_, 1);
lean_dec(v_unused_1220_);
v_unused_1221_ = lean_ctor_get(v_x_1207_, 0);
lean_dec(v_unused_1221_);
v___x_1213_ = v_x_1207_;
v_isShared_1214_ = v_isSharedCheck_1219_;
goto v_resetjp_1212_;
}
else
{
lean_dec(v_x_1207_);
v___x_1213_ = lean_box(0);
v_isShared_1214_ = v_isSharedCheck_1219_;
goto v_resetjp_1212_;
}
v_resetjp_1212_:
{
lean_object* v___x_1215_; lean_object* v___x_1217_; 
v___x_1215_ = lean_box(0);
if (v_isShared_1214_ == 0)
{
lean_ctor_set_tag(v___x_1213_, 0);
lean_ctor_set(v___x_1213_, 1, v___x_1215_);
lean_ctor_set(v___x_1213_, 0, v_x_1209_);
v___x_1217_ = v___x_1213_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v_x_1209_);
lean_ctor_set(v_reuseFailAlloc_1218_, 1, v___x_1215_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
}
else
{
lean_object* v_head_1222_; lean_object* v_tail_1223_; lean_object* v_head_1224_; lean_object* v_tail_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1245_; 
v_head_1222_ = lean_ctor_get(v_x_1207_, 0);
lean_inc(v_head_1222_);
v_tail_1223_ = lean_ctor_get(v_x_1207_, 1);
lean_inc(v_tail_1223_);
lean_dec_ref_known(v_x_1207_, 2);
v_head_1224_ = lean_ctor_get(v_x_1208_, 0);
v_tail_1225_ = lean_ctor_get(v_x_1208_, 1);
v_isSharedCheck_1245_ = !lean_is_exclusive(v_x_1208_);
if (v_isSharedCheck_1245_ == 0)
{
v___x_1227_ = v_x_1208_;
v_isShared_1228_ = v_isSharedCheck_1245_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_tail_1225_);
lean_inc(v_head_1224_);
lean_dec(v_x_1208_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1245_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v_r_1229_; lean_object* v_fst_1230_; lean_object* v_snd_1231_; lean_object* v_q_1232_; lean_object* v_fst_1233_; lean_object* v_snd_1234_; lean_object* v___x_1236_; uint8_t v_isShared_1237_; uint8_t v_isSharedCheck_1244_; 
lean_inc_ref(v_f_1206_);
v_r_1229_ = lp_mathlib_List_mapAccumr_u2082___redArg(v_f_1206_, v_tail_1223_, v_tail_1225_, v_x_1209_);
v_fst_1230_ = lean_ctor_get(v_r_1229_, 0);
lean_inc(v_fst_1230_);
v_snd_1231_ = lean_ctor_get(v_r_1229_, 1);
lean_inc(v_snd_1231_);
lean_dec_ref(v_r_1229_);
v_q_1232_ = lean_apply_3(v_f_1206_, v_head_1222_, v_head_1224_, v_fst_1230_);
v_fst_1233_ = lean_ctor_get(v_q_1232_, 0);
v_snd_1234_ = lean_ctor_get(v_q_1232_, 1);
v_isSharedCheck_1244_ = !lean_is_exclusive(v_q_1232_);
if (v_isSharedCheck_1244_ == 0)
{
v___x_1236_ = v_q_1232_;
v_isShared_1237_ = v_isSharedCheck_1244_;
goto v_resetjp_1235_;
}
else
{
lean_inc(v_snd_1234_);
lean_inc(v_fst_1233_);
lean_dec(v_q_1232_);
v___x_1236_ = lean_box(0);
v_isShared_1237_ = v_isSharedCheck_1244_;
goto v_resetjp_1235_;
}
v_resetjp_1235_:
{
lean_object* v___x_1239_; 
if (v_isShared_1228_ == 0)
{
lean_ctor_set(v___x_1227_, 1, v_snd_1231_);
lean_ctor_set(v___x_1227_, 0, v_snd_1234_);
v___x_1239_ = v___x_1227_;
goto v_reusejp_1238_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v_snd_1234_);
lean_ctor_set(v_reuseFailAlloc_1243_, 1, v_snd_1231_);
v___x_1239_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1238_;
}
v_reusejp_1238_:
{
lean_object* v___x_1241_; 
if (v_isShared_1237_ == 0)
{
lean_ctor_set(v___x_1236_, 1, v___x_1239_);
v___x_1241_ = v___x_1236_;
goto v_reusejp_1240_;
}
else
{
lean_object* v_reuseFailAlloc_1242_; 
v_reuseFailAlloc_1242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1242_, 0, v_fst_1233_);
lean_ctor_set(v_reuseFailAlloc_1242_, 1, v___x_1239_);
v___x_1241_ = v_reuseFailAlloc_1242_;
goto v_reusejp_1240_;
}
v_reusejp_1240_:
{
return v___x_1241_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapAccumr_u2082(lean_object* v_00_u03b1_1246_, lean_object* v_00_u03b2_1247_, lean_object* v_00_u03b3_1248_, lean_object* v_00_u03b4_1249_, lean_object* v_f_1250_, lean_object* v_x_1251_, lean_object* v_x_1252_, lean_object* v_x_1253_){
_start:
{
lean_object* v___x_1254_; 
v___x_1254_ = lp_mathlib_List_mapAccumr_u2082___redArg(v_f_1250_, v_x_1251_, v_x_1252_, v_x_1253_);
return v___x_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_consecutivePairs___redArg(lean_object* v_l_1255_){
_start:
{
if (lean_obj_tag(v_l_1255_) == 0)
{
lean_object* v___x_1256_; 
v___x_1256_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_l_1255_, v_l_1255_);
return v___x_1256_;
}
else
{
lean_object* v_tail_1257_; lean_object* v___x_1258_; 
v_tail_1257_ = lean_ctor_get(v_l_1255_, 1);
lean_inc(v_tail_1257_);
v___x_1258_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_l_1255_, v_tail_1257_);
return v___x_1258_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_consecutivePairs(lean_object* v_00_u03b1_1259_, lean_object* v_l_1260_){
_start:
{
if (lean_obj_tag(v_l_1260_) == 0)
{
lean_object* v___x_1261_; 
v___x_1261_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_l_1260_, v_l_1260_);
return v___x_1261_;
}
else
{
lean_object* v_tail_1262_; lean_object* v___x_1263_; 
v_tail_1262_ = lean_ctor_get(v_l_1260_, 1);
lean_inc(v_tail_1262_);
v___x_1263_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_l_1260_, v_tail_1262_);
return v___x_1263_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
