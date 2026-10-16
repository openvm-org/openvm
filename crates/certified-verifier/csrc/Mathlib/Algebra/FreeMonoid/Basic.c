// Lean compiler output
// Module: Mathlib.Algebra.FreeMonoid.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Algebra.Group.Units.Defs public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.BigOperators.Group.List.Defs public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Data.List.Basic public import Mathlib.Tactic.ToDual public import Mathlib.Util.CompileInductive
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Units_instInhabited___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_mathlib_AddUnits_instInhabited___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_FreeMonoid_toList___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_toList___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_toList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_toList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_ofList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_ofList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeMonoid_instCancelMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeMonoid_instCancelMonoid___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___closed__0 = (const lean_object*)&lp_mathlib_FreeMonoid_instCancelMonoid___closed__0_value;
static lean_once_cell_t lp_mathlib_FreeMonoid_instCancelMonoid___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___closed__1;
static lean_once_cell_t lp_mathlib_FreeMonoid_instCancelMonoid___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___closed__2;
static lean_once_cell_t lp_mathlib_FreeMonoid_instCancelMonoid___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___closed__3;
static lean_once_cell_t lp_mathlib_FreeMonoid_instCancelMonoid___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__0 = (const lean_object*)&lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__0_value;
static lean_once_cell_t lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instAddCancelMonoid(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instInhabited(lean_object*);
static lean_once_cell_t lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instUniqueOfIsEmpty(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_of(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_of(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_length(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_length(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instMembership(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instMembership(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux_match__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeMonoid_prodAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeMonoid_prodAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeAddMonoid_sumAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeAddMonoid_sumAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeMonoid_lift___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeMonoid_lift___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeMonoid_lift___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeMonoid_lift___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAddMonoid_lift___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAddMonoid_lift___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeAddMonoid_lift___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_mkAddAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_mkAddAction(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_map(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeMonoid_uniqueUnits___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_uniqueUnits___closed__0;
static lean_once_cell_t lp_mathlib_FreeMonoid_uniqueUnits___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeMonoid_uniqueUnits___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_uniqueUnits(lean_object*);
static lean_once_cell_t lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0;
static lean_once_cell_t lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_uniqueAddUnits(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_reverse(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_reverse(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_freeAddMonoidCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_freeAddMonoidCongr(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_FreeMonoid_toList___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_toList(lean_object* v_00_u03b1_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_toList(lean_object* v_00_u03b1_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_ofList(lean_object* v_00_u03b1_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_ofList(lean_object* v_00_u03b1_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___lam__0(lean_object* v_self_10_, lean_object* v___y_11_){
_start:
{
lean_object* v_toFun_12_; lean_object* v___x_13_; 
v_toFun_12_ = lean_ctor_get(v_self_10_, 0);
lean_inc(v_toFun_12_);
lean_dec_ref(v_self_10_);
v___x_13_ = lean_apply_1(v_toFun_12_, v___y_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid___lam__2(lean_object* v___f_14_, lean_object* v___x_15_, lean_object* v___f_16_, lean_object* v_x_17_, lean_object* v_y_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
lean_inc_ref(v___f_14_);
lean_inc_ref_n(v___x_15_, 2);
v___x_19_ = lean_apply_2(v___f_14_, v___x_15_, v_x_17_);
v___x_20_ = lean_apply_2(v___f_14_, v___x_15_, v_y_18_);
v___x_21_ = l_List_appendTR___redArg(v___x_19_, v___x_20_);
v___x_22_ = lean_apply_2(v___f_16_, v___x_15_, v___x_21_);
return v___x_22_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__1(void){
_start:
{
lean_object* v___x_24_; lean_object* v___f_25_; lean_object* v___f_26_; 
v___x_24_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v___f_25_ = ((lean_object*)(lp_mathlib_FreeMonoid_instCancelMonoid___closed__0));
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_instCancelMonoid___lam__2), 5, 3);
lean_closure_set(v___f_26_, 0, v___f_25_);
lean_closure_set(v___f_26_, 1, v___x_24_);
lean_closure_set(v___f_26_, 2, v___f_25_);
return v___f_26_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__2(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_27_ = lean_box(0);
v___x_28_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v___x_29_ = lp_mathlib_FreeMonoid_instCancelMonoid___lam__0(v___x_28_, v___x_27_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__3(void){
_start:
{
lean_object* v___x_30_; lean_object* v___f_31_; lean_object* v___x_32_; 
v___x_30_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__2, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__2_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__2);
v___f_31_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__1, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__1_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__1);
v___x_32_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_32_, 0, lean_box(0));
lean_closure_set(v___x_32_, 1, v___f_31_);
lean_closure_set(v___x_32_, 2, v___x_30_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__4(void){
_start:
{
lean_object* v___x_33_; lean_object* v___f_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_33_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__3, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__3_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__3);
v___f_34_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__1, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__1_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__1);
v___x_35_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__2, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__2_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__2);
v___x_36_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v___f_34_);
lean_ctor_set(v___x_36_, 2, v___x_33_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instCancelMonoid(lean_object* v_00_u03b1_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__4, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__4_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__4);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0(lean_object* v_y_39_, lean_object* v_x_40_){
_start:
{
lean_inc(v_y_39_);
return v_y_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object* v_y_41_, lean_object* v_x_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___lam__0(v_y_41_, v_x_42_);
lean_dec(v_x_42_);
lean_dec(v_y_41_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg___lam__1(lean_object* v___x_44_, lean_object* v___f_45_, lean_object* v_x_46_, lean_object* v_y_47_){
_start:
{
lean_object* v_toFun_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v_toFun_48_ = lean_ctor_get(v___x_44_, 0);
lean_inc(v_toFun_48_);
lean_inc_ref(v___f_45_);
lean_inc_ref(v___x_44_);
v___x_49_ = lean_apply_2(v___f_45_, v___x_44_, v_x_46_);
v___x_50_ = lean_apply_2(v___f_45_, v___x_44_, v_y_47_);
v___x_51_ = l_List_appendTR___redArg(v___x_49_, v___x_50_);
v___x_52_ = lean_apply_1(v_toFun_48_, v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_zero_53_, lean_object* v_n_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = lean_unsigned_to_nat(0u);
v___x_58_ = lean_nat_dec_eq(v_n_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___y_62_; lean_object* v___y_63_; lean_object* v___x_66_; lean_object* v___x_70_; uint8_t v___x_71_; 
v___f_59_ = ((lean_object*)(lp_mathlib_FreeMonoid_instCancelMonoid___closed__0));
v___x_60_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v___x_66_ = lean_unsigned_to_nat(1u);
v___x_70_ = lean_nat_land(v___x_66_, v_n_54_);
v___x_71_ = lean_nat_dec_eq(v___x_70_, v___x_57_);
lean_dec(v___x_70_);
if (v___x_71_ == 0)
{
goto v___jp_67_;
}
else
{
if (v___x_58_ == 0)
{
lean_object* v___x_72_; 
v___x_72_ = lean_nat_shiftr(v_n_54_, v___x_66_);
lean_dec(v_n_54_);
v___y_62_ = v___x_72_;
v___y_63_ = v___y_55_;
goto v___jp_61_;
}
else
{
goto v___jp_67_;
}
}
v___jp_61_:
{
lean_object* v___x_64_; 
lean_inc(v___y_56_);
v___x_64_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg___lam__1(v___x_60_, v___f_59_, v___y_56_, v___y_56_);
v_n_54_ = v___y_62_;
v___y_55_ = v___y_63_;
v___y_56_ = v___x_64_;
goto _start;
}
v___jp_67_:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_nat_shiftr(v_n_54_, v___x_66_);
lean_dec(v_n_54_);
lean_inc(v___y_56_);
v___x_69_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg___lam__1(v___x_60_, v___f_59_, v___y_55_, v___y_56_);
v___y_62_ = v___x_68_;
v___y_63_ = v___x_69_;
goto v___jp_61_;
}
}
else
{
lean_object* v___x_73_; 
lean_dec(v_n_54_);
v___x_73_ = lean_apply_2(v_zero_53_, v___y_55_, v___y_56_);
return v___x_73_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg(lean_object* v_k_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v___f_78_; lean_object* v___x_79_; 
v___f_78_ = ((lean_object*)(lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg___closed__0));
v___x_79_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg(v___f_78_, v_k_75_, v_a_76_, v_a_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg(lean_object* v_k_80_, lean_object* v_a_81_){
_start:
{
lean_object* v___x_82_; lean_object* v_toFun_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_82_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_83_ = lean_ctor_get(v___x_82_, 0);
v___x_84_ = lean_box(0);
lean_inc(v_toFun_83_);
v___x_85_ = lean_apply_1(v_toFun_83_, v___x_84_);
v___x_86_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg(v_k_80_, v___x_85_, v_a_81_);
return v___x_86_;
}
}
static lean_object* _init_lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1(void){
_start:
{
lean_object* v___f_88_; lean_object* v___f_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___f_88_ = ((lean_object*)(lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__0));
v___f_89_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__1, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__1_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__1);
v___x_90_ = lean_obj_once(&lp_mathlib_FreeMonoid_instCancelMonoid___closed__2, &lp_mathlib_FreeMonoid_instCancelMonoid___closed__2_once, _init_lp_mathlib_FreeMonoid_instCancelMonoid___closed__2);
v___x_91_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___f_89_);
lean_ctor_set(v___x_91_, 2, v___f_88_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instAddCancelMonoid(lean_object* v_00_u03b1_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lean_obj_once(&lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1, &lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1_once, _init_lp_mathlib_FreeAddMonoid_instAddCancelMonoid___closed__1);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0___redArg(lean_object* v_k_94_, lean_object* v_m_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg(v_k_94_, v_m_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0(lean_object* v_00_u03b1_97_, lean_object* v_k_98_, lean_object* v_m_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg(v_k_98_, v_m_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0(lean_object* v_00_u03b1_101_, lean_object* v_k_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0___redArg(v_k_102_, v_a_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_105_, lean_object* v_k_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1___redArg(v_k_106_, v_a_107_, v_a_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_110_, lean_object* v_zero_111_, lean_object* v_n_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_Nat_binaryRec___at___00nsmulBinRec_go___at___00nsmulBinRec___at___00nsmulBinRecAuto___at___00FreeAddMonoid_instAddCancelMonoid_spec__0_spec__0_spec__1_spec__2___redArg(v_zero_111_, v_n_112_, v___y_113_, v___y_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instInhabited(lean_object* v_00_u03b1_116_){
_start:
{
lean_object* v___x_117_; lean_object* v_toFun_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_117_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_118_ = lean_ctor_get(v___x_117_, 0);
v___x_119_ = lean_box(0);
lean_inc(v_toFun_118_);
v___x_120_ = lean_apply_1(v_toFun_118_, v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instInhabited(lean_object* v_00_u03b1_121_){
_start:
{
lean_object* v___x_122_; lean_object* v_toFun_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_122_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_123_ = lean_ctor_get(v___x_122_, 0);
v___x_124_ = lean_box(0);
lean_inc(v_toFun_123_);
v___x_125_ = lean_apply_1(v_toFun_123_, v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0(void){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_FreeMonoid_instInhabited(lean_box(0));
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instUniqueOfIsEmpty(lean_object* v_00_u03b1_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_obj_once(&lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0, &lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0_once, _init_lp_mathlib_FreeMonoid_instUniqueOfIsEmpty___closed__0);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0(void){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_FreeAddMonoid_instInhabited(lean_box(0));
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty(lean_object* v_00_u03b1_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lean_obj_once(&lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0, &lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0_once, _init_lp_mathlib_FreeAddMonoid_instUniqueOfIsEmpty___closed__0);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_of___redArg(lean_object* v_x_134_){
_start:
{
lean_object* v___x_135_; lean_object* v_toFun_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_135_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_136_ = lean_ctor_get(v___x_135_, 0);
v___x_137_ = lean_box(0);
v___x_138_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_138_, 0, v_x_134_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
lean_inc(v_toFun_136_);
v___x_139_ = lean_apply_1(v_toFun_136_, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_of(lean_object* v_00_u03b1_140_, lean_object* v_x_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_FreeMonoid_of___redArg(v_x_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_of___redArg(lean_object* v_x_143_){
_start:
{
lean_object* v___x_144_; lean_object* v_toFun_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_144_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_145_ = lean_ctor_get(v___x_144_, 0);
v___x_146_ = lean_box(0);
v___x_147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_147_, 0, v_x_143_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
lean_inc(v_toFun_145_);
v___x_148_ = lean_apply_1(v_toFun_145_, v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_of(lean_object* v_00_u03b1_149_, lean_object* v_x_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_FreeAddMonoid_of___redArg(v_x_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_length___redArg(lean_object* v_a_152_){
_start:
{
lean_object* v___x_153_; lean_object* v_toFun_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_153_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_154_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_toFun_154_);
v___x_155_ = lean_apply_1(v_toFun_154_, v_a_152_);
v___x_156_ = l_List_lengthTR___redArg(v___x_155_);
lean_dec(v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_length(lean_object* v_00_u03b1_157_, lean_object* v_a_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_FreeMonoid_length___redArg(v_a_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_length___redArg(lean_object* v_a_160_){
_start:
{
lean_object* v___x_161_; lean_object* v_toFun_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_161_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_toFun_162_);
v___x_163_ = lean_apply_1(v_toFun_162_, v_a_160_);
v___x_164_ = l_List_lengthTR___redArg(v___x_163_);
lean_dec(v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_length(lean_object* v_00_u03b1_165_, lean_object* v_a_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_FreeAddMonoid_length___redArg(v_a_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_instMembership(lean_object* v_00_u03b1_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lean_box(0);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_instMembership(lean_object* v_00_u03b1_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___redArg(lean_object* v_xs_172_, lean_object* v_one_173_, lean_object* v_of__mul_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_one_173_, v_of__mul_174_, v_xs_172_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___redArg___boxed(lean_object* v_xs_176_, lean_object* v_one_177_, lean_object* v_of__mul_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_FreeMonoid_recOn___redArg(v_xs_176_, v_one_177_, v_of__mul_178_);
lean_dec(v_one_177_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn(lean_object* v_00_u03b1_180_, lean_object* v_motive_181_, lean_object* v_xs_182_, lean_object* v_one_183_, lean_object* v_of__mul_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_one_183_, v_of__mul_184_, v_xs_182_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_recOn___boxed(lean_object* v_00_u03b1_186_, lean_object* v_motive_187_, lean_object* v_xs_188_, lean_object* v_one_189_, lean_object* v_of__mul_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_FreeMonoid_recOn(v_00_u03b1_186_, v_motive_187_, v_xs_188_, v_one_189_, v_of__mul_190_);
lean_dec(v_one_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___redArg(lean_object* v_xs_192_, lean_object* v_one_193_, lean_object* v_of__mul_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_one_193_, v_of__mul_194_, v_xs_192_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___redArg___boxed(lean_object* v_xs_196_, lean_object* v_one_197_, lean_object* v_of__mul_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_FreeAddMonoid_recOn___redArg(v_xs_196_, v_one_197_, v_of__mul_198_);
lean_dec(v_one_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn(lean_object* v_00_u03b1_200_, lean_object* v_motive_201_, lean_object* v_xs_202_, lean_object* v_one_203_, lean_object* v_of__mul_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_one_203_, v_of__mul_204_, v_xs_202_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_recOn___boxed(lean_object* v_00_u03b1_206_, lean_object* v_motive_207_, lean_object* v_xs_208_, lean_object* v_one_209_, lean_object* v_of__mul_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_FreeAddMonoid_recOn(v_00_u03b1_206_, v_motive_207_, v_xs_208_, v_one_209_, v_of__mul_210_);
lean_dec(v_one_209_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___redArg(lean_object* v_xs_212_, lean_object* v_one_213_, lean_object* v_of__mul_214_){
_start:
{
if (lean_obj_tag(v_xs_212_) == 0)
{
lean_dec(v_of__mul_214_);
lean_inc(v_one_213_);
return v_one_213_;
}
else
{
lean_object* v_x_215_; lean_object* v_xs_216_; lean_object* v___x_217_; 
v_x_215_ = lean_ctor_get(v_xs_212_, 0);
lean_inc(v_x_215_);
v_xs_216_ = lean_ctor_get(v_xs_212_, 1);
lean_inc(v_xs_216_);
lean_dec_ref_known(v_xs_212_, 2);
v___x_217_ = lean_apply_2(v_of__mul_214_, v_x_215_, v_xs_216_);
return v___x_217_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___redArg___boxed(lean_object* v_xs_218_, lean_object* v_one_219_, lean_object* v_of__mul_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_FreeMonoid_casesOn___redArg(v_xs_218_, v_one_219_, v_of__mul_220_);
lean_dec(v_one_219_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn(lean_object* v_00_u03b1_222_, lean_object* v_motive_223_, lean_object* v_xs_224_, lean_object* v_one_225_, lean_object* v_of__mul_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_FreeMonoid_casesOn___redArg(v_xs_224_, v_one_225_, v_of__mul_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_casesOn___boxed(lean_object* v_00_u03b1_228_, lean_object* v_motive_229_, lean_object* v_xs_230_, lean_object* v_one_231_, lean_object* v_of__mul_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_FreeMonoid_casesOn(v_00_u03b1_228_, v_motive_229_, v_xs_230_, v_one_231_, v_of__mul_232_);
lean_dec(v_one_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___redArg(lean_object* v_xs_234_, lean_object* v_one_235_, lean_object* v_of__mul_236_){
_start:
{
if (lean_obj_tag(v_xs_234_) == 0)
{
lean_dec(v_of__mul_236_);
lean_inc(v_one_235_);
return v_one_235_;
}
else
{
lean_object* v_x_237_; lean_object* v_xs_238_; lean_object* v___x_239_; 
v_x_237_ = lean_ctor_get(v_xs_234_, 0);
lean_inc(v_x_237_);
v_xs_238_ = lean_ctor_get(v_xs_234_, 1);
lean_inc(v_xs_238_);
lean_dec_ref_known(v_xs_234_, 2);
v___x_239_ = lean_apply_2(v_of__mul_236_, v_x_237_, v_xs_238_);
return v___x_239_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___redArg___boxed(lean_object* v_xs_240_, lean_object* v_one_241_, lean_object* v_of__mul_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_FreeAddMonoid_casesOn___redArg(v_xs_240_, v_one_241_, v_of__mul_242_);
lean_dec(v_one_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn(lean_object* v_00_u03b1_244_, lean_object* v_motive_245_, lean_object* v_xs_246_, lean_object* v_one_247_, lean_object* v_of__mul_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_FreeAddMonoid_casesOn___redArg(v_xs_246_, v_one_247_, v_of__mul_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_casesOn___boxed(lean_object* v_00_u03b1_250_, lean_object* v_motive_251_, lean_object* v_xs_252_, lean_object* v_one_253_, lean_object* v_of__mul_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_FreeAddMonoid_casesOn(v_00_u03b1_250_, v_motive_251_, v_xs_252_, v_one_253_, v_of__mul_254_);
lean_dec(v_one_253_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg___lam__0(lean_object* v_toMul_256_, lean_object* v_x1_257_, lean_object* v_x2_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lean_apply_2(v_toMul_256_, v_x1_257_, v_x2_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg(lean_object* v_inst_260_, lean_object* v_x_261_){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_260_);
v___x_263_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_262_);
if (lean_obj_tag(v_x_261_) == 0)
{
lean_object* v_toOne_264_; 
v_toOne_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc(v_toOne_264_);
lean_dec_ref(v___x_263_);
return v_toOne_264_;
}
else
{
lean_object* v_toMul_265_; lean_object* v_head_266_; lean_object* v_tail_267_; lean_object* v___f_268_; lean_object* v___x_269_; 
v_toMul_265_ = lean_ctor_get(v___x_263_, 1);
lean_inc(v_toMul_265_);
lean_dec_ref(v___x_263_);
v_head_266_ = lean_ctor_get(v_x_261_, 0);
lean_inc(v_head_266_);
v_tail_267_ = lean_ctor_get(v_x_261_, 1);
lean_inc(v_tail_267_);
lean_dec_ref_known(v_x_261_, 2);
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_prodAux___redArg___lam__0), 3, 1);
lean_closure_set(v___f_268_, 0, v_toMul_265_);
v___x_269_ = l_List_foldl___redArg(v___f_268_, v_head_266_, v_tail_267_);
return v___x_269_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___redArg___boxed(lean_object* v_inst_270_, lean_object* v_x_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_FreeMonoid_prodAux___redArg(v_inst_270_, v_x_271_);
lean_dec_ref(v_inst_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux(lean_object* v_M_273_, lean_object* v_inst_274_, lean_object* v_x_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_FreeMonoid_prodAux___redArg(v_inst_274_, v_x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_prodAux___boxed(lean_object* v_M_277_, lean_object* v_inst_278_, lean_object* v_x_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_FreeMonoid_prodAux(v_M_277_, v_inst_278_, v_x_279_);
lean_dec_ref(v_inst_278_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux_match__1___redArg(lean_object* v_x_281_, lean_object* v_h__1_282_, lean_object* v_h__2_283_){
_start:
{
if (lean_obj_tag(v_x_281_) == 0)
{
lean_object* v___x_284_; lean_object* v___x_285_; 
lean_dec(v_h__2_283_);
v___x_284_ = lean_box(0);
v___x_285_ = lean_apply_1(v_h__1_282_, v___x_284_);
return v___x_285_;
}
else
{
lean_object* v_head_286_; lean_object* v_tail_287_; lean_object* v___x_288_; 
lean_dec(v_h__1_282_);
v_head_286_ = lean_ctor_get(v_x_281_, 0);
lean_inc(v_head_286_);
v_tail_287_ = lean_ctor_get(v_x_281_, 1);
lean_inc(v_tail_287_);
lean_dec_ref_known(v_x_281_, 2);
v___x_288_ = lean_apply_2(v_h__2_283_, v_head_286_, v_tail_287_);
return v___x_288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux_match__1(lean_object* v_M_289_, lean_object* v_motive_290_, lean_object* v_x_291_, lean_object* v_h__1_292_, lean_object* v_h__2_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_mathlib_FreeAddMonoid_sumAux_match__1___redArg(v_x_291_, v_h__1_292_, v_h__2_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg___lam__0(lean_object* v_toAdd_295_, lean_object* v_x1_296_, lean_object* v_x2_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lean_apply_2(v_toAdd_295_, v_x1_296_, v_x2_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg(lean_object* v_inst_299_, lean_object* v_x_300_){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_301_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_299_);
v___x_302_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_301_);
if (lean_obj_tag(v_x_300_) == 0)
{
lean_object* v_toZero_303_; 
v_toZero_303_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_toZero_303_);
lean_dec_ref(v___x_302_);
return v_toZero_303_;
}
else
{
lean_object* v_toAdd_304_; lean_object* v_head_305_; lean_object* v_tail_306_; lean_object* v___f_307_; lean_object* v___x_308_; 
v_toAdd_304_ = lean_ctor_get(v___x_302_, 1);
lean_inc(v_toAdd_304_);
lean_dec_ref(v___x_302_);
v_head_305_ = lean_ctor_get(v_x_300_, 0);
lean_inc(v_head_305_);
v_tail_306_ = lean_ctor_get(v_x_300_, 1);
lean_inc(v_tail_306_);
lean_dec_ref_known(v_x_300_, 2);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddMonoid_sumAux___redArg___lam__0), 3, 1);
lean_closure_set(v___f_307_, 0, v_toAdd_304_);
v___x_308_ = l_List_foldl___redArg(v___f_307_, v_head_305_, v_tail_306_);
return v___x_308_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___redArg___boxed(lean_object* v_inst_309_, lean_object* v_x_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_FreeAddMonoid_sumAux___redArg(v_inst_309_, v_x_310_);
lean_dec_ref(v_inst_309_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux(lean_object* v_M_312_, lean_object* v_inst_313_, lean_object* v_x_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_FreeAddMonoid_sumAux___redArg(v_inst_313_, v_x_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_sumAux___boxed(lean_object* v_M_316_, lean_object* v_inst_317_, lean_object* v_x_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_FreeAddMonoid_sumAux(v_M_316_, v_inst_317_, v_x_318_);
lean_dec_ref(v_inst_317_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeMonoid_prodAux_match__1_splitter___redArg(lean_object* v_x_320_, lean_object* v_h__1_321_, lean_object* v_h__2_322_){
_start:
{
if (lean_obj_tag(v_x_320_) == 0)
{
lean_object* v___x_323_; lean_object* v___x_324_; 
lean_dec(v_h__2_322_);
v___x_323_ = lean_box(0);
v___x_324_ = lean_apply_1(v_h__1_321_, v___x_323_);
return v___x_324_;
}
else
{
lean_object* v_head_325_; lean_object* v_tail_326_; lean_object* v___x_327_; 
lean_dec(v_h__1_321_);
v_head_325_ = lean_ctor_get(v_x_320_, 0);
lean_inc(v_head_325_);
v_tail_326_ = lean_ctor_get(v_x_320_, 1);
lean_inc(v_tail_326_);
lean_dec_ref_known(v_x_320_, 2);
v___x_327_ = lean_apply_2(v_h__2_322_, v_head_325_, v_tail_326_);
return v___x_327_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeMonoid_prodAux_match__1_splitter(lean_object* v_M_328_, lean_object* v_motive_329_, lean_object* v_x_330_, lean_object* v_h__1_331_, lean_object* v_h__2_332_){
_start:
{
if (lean_obj_tag(v_x_330_) == 0)
{
lean_object* v___x_333_; lean_object* v___x_334_; 
lean_dec(v_h__2_332_);
v___x_333_ = lean_box(0);
v___x_334_ = lean_apply_1(v_h__1_331_, v___x_333_);
return v___x_334_;
}
else
{
lean_object* v_head_335_; lean_object* v_tail_336_; lean_object* v___x_337_; 
lean_dec(v_h__1_331_);
v_head_335_ = lean_ctor_get(v_x_330_, 0);
lean_inc(v_head_335_);
v_tail_336_ = lean_ctor_get(v_x_330_, 1);
lean_inc(v_tail_336_);
lean_dec_ref_known(v_x_330_, 2);
v___x_337_ = lean_apply_2(v_h__2_332_, v_head_335_, v_tail_336_);
return v___x_337_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeAddMonoid_sumAux_match__1_splitter___redArg(lean_object* v_x_338_, lean_object* v_h__1_339_, lean_object* v_h__2_340_){
_start:
{
if (lean_obj_tag(v_x_338_) == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v_h__2_340_);
v___x_341_ = lean_box(0);
v___x_342_ = lean_apply_1(v_h__1_339_, v___x_341_);
return v___x_342_;
}
else
{
lean_object* v_head_343_; lean_object* v_tail_344_; lean_object* v___x_345_; 
lean_dec(v_h__1_339_);
v_head_343_ = lean_ctor_get(v_x_338_, 0);
lean_inc(v_head_343_);
v_tail_344_ = lean_ctor_get(v_x_338_, 1);
lean_inc(v_tail_344_);
lean_dec_ref_known(v_x_338_, 2);
v___x_345_ = lean_apply_2(v_h__2_340_, v_head_343_, v_tail_344_);
return v___x_345_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_FreeMonoid_Basic_0__FreeAddMonoid_sumAux_match__1_splitter(lean_object* v_M_346_, lean_object* v_motive_347_, lean_object* v_x_348_, lean_object* v_h__1_349_, lean_object* v_h__2_350_){
_start:
{
if (lean_obj_tag(v_x_348_) == 0)
{
lean_object* v___x_351_; lean_object* v___x_352_; 
lean_dec(v_h__2_350_);
v___x_351_ = lean_box(0);
v___x_352_ = lean_apply_1(v_h__1_349_, v___x_351_);
return v___x_352_;
}
else
{
lean_object* v_head_353_; lean_object* v_tail_354_; lean_object* v___x_355_; 
lean_dec(v_h__1_349_);
v_head_353_ = lean_ctor_get(v_x_348_, 0);
lean_inc(v_head_353_);
v_tail_354_ = lean_ctor_get(v_x_348_, 1);
lean_inc(v_tail_354_);
lean_dec_ref_known(v_x_348_, 2);
v___x_355_ = lean_apply_2(v_h__2_350_, v_head_353_, v_tail_354_);
return v___x_355_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__0(lean_object* v_f_356_, lean_object* v_x_357_){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = lp_mathlib_FreeMonoid_of___redArg(v_x_357_);
v___x_359_ = lean_apply_1(v_f_356_, v___x_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__1(lean_object* v_inst_360_, lean_object* v_f_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_363_; lean_object* v_toFun_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_363_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_toFun_364_);
v___x_365_ = lean_apply_1(v_toFun_364_, v___y_362_);
v___x_366_ = lean_box(0);
v___x_367_ = l_List_mapTR_loop___redArg(v_f_361_, v___x_365_, v___x_366_);
v___x_368_ = lp_mathlib_FreeMonoid_prodAux___redArg(v_inst_360_, v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg___lam__1___boxed(lean_object* v_inst_369_, lean_object* v_f_370_, lean_object* v___y_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_FreeMonoid_lift___redArg___lam__1(v_inst_369_, v_f_370_, v___y_371_);
lean_dec_ref(v_inst_369_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift___redArg(lean_object* v_inst_374_){
_start:
{
lean_object* v___f_375_; lean_object* v___f_376_; lean_object* v___x_377_; 
v___f_375_ = ((lean_object*)(lp_mathlib_FreeMonoid_lift___redArg___closed__0));
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_lift___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_376_, 0, v_inst_374_);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v___f_376_);
lean_ctor_set(v___x_377_, 1, v___f_375_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_lift(lean_object* v_00_u03b1_378_, lean_object* v_M_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_mathlib_FreeMonoid_lift___redArg(v_inst_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__0(lean_object* v_f_382_, lean_object* v_x_383_){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_384_ = lp_mathlib_FreeAddMonoid_of___redArg(v_x_383_);
v___x_385_ = lean_apply_1(v_f_382_, v___x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__1(lean_object* v_inst_386_, lean_object* v_f_387_, lean_object* v___y_388_){
_start:
{
lean_object* v___x_389_; lean_object* v_toFun_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_389_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_390_ = lean_ctor_get(v___x_389_, 0);
lean_inc(v_toFun_390_);
v___x_391_ = lean_apply_1(v_toFun_390_, v___y_388_);
v___x_392_ = lean_box(0);
v___x_393_ = l_List_mapTR_loop___redArg(v_f_387_, v___x_391_, v___x_392_);
v___x_394_ = lp_mathlib_FreeAddMonoid_sumAux___redArg(v_inst_386_, v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg___lam__1___boxed(lean_object* v_inst_395_, lean_object* v_f_396_, lean_object* v___y_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_FreeAddMonoid_lift___redArg___lam__1(v_inst_395_, v_f_396_, v___y_397_);
lean_dec_ref(v_inst_395_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift___redArg(lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; lean_object* v___f_402_; lean_object* v___x_403_; 
v___f_401_ = ((lean_object*)(lp_mathlib_FreeAddMonoid_lift___redArg___closed__0));
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_FreeAddMonoid_lift___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_402_, 0, v_inst_400_);
v___x_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_403_, 0, v___f_402_);
lean_ctor_set(v___x_403_, 1, v___f_401_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_lift(lean_object* v_00_u03b1_404_, lean_object* v_M_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_FreeAddMonoid_lift___redArg(v_inst_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg(lean_object* v_f_408_, lean_object* v_as_409_, size_t v_i_410_, size_t v_stop_411_, lean_object* v_b_412_){
_start:
{
uint8_t v___x_413_; 
v___x_413_ = lean_usize_dec_eq(v_i_410_, v_stop_411_);
if (v___x_413_ == 0)
{
size_t v___x_414_; size_t v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_414_ = ((size_t)1ULL);
v___x_415_ = lean_usize_sub(v_i_410_, v___x_414_);
v___x_416_ = lean_array_uget_borrowed(v_as_409_, v___x_415_);
lean_inc(v_f_408_);
lean_inc(v___x_416_);
v___x_417_ = lean_apply_2(v_f_408_, v___x_416_, v_b_412_);
v_i_410_ = v___x_415_;
v_b_412_ = v___x_417_;
goto _start;
}
else
{
lean_dec(v_f_408_);
return v_b_412_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg___boxed(lean_object* v_f_419_, lean_object* v_as_420_, lean_object* v_i_421_, lean_object* v_stop_422_, lean_object* v_b_423_){
_start:
{
size_t v_i_boxed_424_; size_t v_stop_boxed_425_; lean_object* v_res_426_; 
v_i_boxed_424_ = lean_unbox_usize(v_i_421_);
lean_dec(v_i_421_);
v_stop_boxed_425_ = lean_unbox_usize(v_stop_422_);
lean_dec(v_stop_422_);
v_res_426_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg(v_f_419_, v_as_420_, v_i_boxed_424_, v_stop_boxed_425_, v_b_423_);
lean_dec_ref(v_as_420_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0___redArg(lean_object* v_f_427_, lean_object* v_init_428_, lean_object* v_l_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; uint8_t v___x_433_; 
v___x_430_ = lean_array_mk(v_l_429_);
v___x_431_ = lean_array_get_size(v___x_430_);
v___x_432_ = lean_unsigned_to_nat(0u);
v___x_433_ = lean_nat_dec_lt(v___x_432_, v___x_431_);
if (v___x_433_ == 0)
{
lean_dec_ref(v___x_430_);
lean_dec(v_f_427_);
return v_init_428_;
}
else
{
size_t v___x_434_; size_t v___x_435_; lean_object* v___x_436_; 
v___x_434_ = lean_usize_of_nat(v___x_431_);
v___x_435_ = ((size_t)0ULL);
v___x_436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg(v_f_427_, v___x_430_, v___x_434_, v___x_435_, v_init_428_);
lean_dec_ref(v___x_430_);
return v___x_436_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0(lean_object* v_f_437_, lean_object* v_l_438_, lean_object* v_b_439_){
_start:
{
lean_object* v___x_440_; lean_object* v_toFun_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_440_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_441_ = lean_ctor_get(v___x_440_, 0);
lean_inc(v_toFun_441_);
v___x_442_ = lean_apply_1(v_toFun_441_, v_l_438_);
v___x_443_ = lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0___redArg(v_f_437_, v_b_439_, v___x_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction___redArg(lean_object* v_f_444_){
_start:
{
lean_object* v___f_445_; 
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_445_, 0, v_f_444_);
return v___f_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_mkMulAction(lean_object* v_00_u03b1_446_, lean_object* v_00_u03b2_447_, lean_object* v_f_448_){
_start:
{
lean_object* v___f_449_; 
v___f_449_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_449_, 0, v_f_448_);
return v___f_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0(lean_object* v_00_u03b1_450_, lean_object* v_00_u03b2_451_, lean_object* v_f_452_, lean_object* v_init_453_, lean_object* v_l_454_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lp_mathlib_List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0___redArg(v_f_452_, v_init_453_, v_l_454_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0(lean_object* v_00_u03b1_456_, lean_object* v_00_u03b2_457_, lean_object* v_f_458_, lean_object* v_as_459_, size_t v_i_460_, size_t v_stop_461_, lean_object* v_b_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___redArg(v_f_458_, v_as_459_, v_i_460_, v_stop_461_, v_b_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0___boxed(lean_object* v_00_u03b1_464_, lean_object* v_00_u03b2_465_, lean_object* v_f_466_, lean_object* v_as_467_, lean_object* v_i_468_, lean_object* v_stop_469_, lean_object* v_b_470_){
_start:
{
size_t v_i_boxed_471_; size_t v_stop_boxed_472_; lean_object* v_res_473_; 
v_i_boxed_471_ = lean_unbox_usize(v_i_468_);
lean_dec(v_i_468_);
v_stop_boxed_472_ = lean_unbox_usize(v_stop_469_);
lean_dec(v_stop_469_);
v_res_473_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00FreeMonoid_mkMulAction_spec__0_spec__0(v_00_u03b1_464_, v_00_u03b2_465_, v_f_466_, v_as_467_, v_i_boxed_471_, v_stop_boxed_472_, v_b_470_);
lean_dec_ref(v_as_467_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_mkAddAction___redArg(lean_object* v_f_474_){
_start:
{
lean_object* v___f_475_; 
v___f_475_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_475_, 0, v_f_474_);
return v___f_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_mkAddAction(lean_object* v_00_u03b1_476_, lean_object* v_00_u03b2_477_, lean_object* v_f_478_){
_start:
{
lean_object* v___f_479_; 
v___f_479_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_mkMulAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_479_, 0, v_f_478_);
return v___f_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0___redArg(lean_object* v_f_480_, lean_object* v_a_481_, lean_object* v_a_482_){
_start:
{
if (lean_obj_tag(v_a_481_) == 0)
{
lean_object* v___x_483_; 
lean_dec(v_f_480_);
v___x_483_ = l_List_reverse___redArg(v_a_482_);
return v___x_483_;
}
else
{
lean_object* v_head_484_; lean_object* v_tail_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_494_; 
v_head_484_ = lean_ctor_get(v_a_481_, 0);
v_tail_485_ = lean_ctor_get(v_a_481_, 1);
v_isSharedCheck_494_ = !lean_is_exclusive(v_a_481_);
if (v_isSharedCheck_494_ == 0)
{
v___x_487_ = v_a_481_;
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_tail_485_);
lean_inc(v_head_484_);
lean_dec(v_a_481_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_489_; lean_object* v___x_491_; 
lean_inc(v_f_480_);
v___x_489_ = lean_apply_1(v_f_480_, v_head_484_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 1, v_a_482_);
lean_ctor_set(v___x_487_, 0, v___x_489_);
v___x_491_ = v___x_487_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v_a_482_);
v___x_491_ = v_reuseFailAlloc_493_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
v_a_481_ = v_tail_485_;
v_a_482_ = v___x_491_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map___redArg___lam__0(lean_object* v_f_495_, lean_object* v_l_496_){
_start:
{
lean_object* v___x_497_; lean_object* v_toFun_498_; lean_object* v_toFun_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_497_ = lean_obj_once(&lp_mathlib_FreeMonoid_toList___closed__0, &lp_mathlib_FreeMonoid_toList___closed__0_once, _init_lp_mathlib_FreeMonoid_toList___closed__0);
v_toFun_498_ = lean_ctor_get(v___x_497_, 0);
v_toFun_499_ = lean_ctor_get(v___x_497_, 0);
lean_inc(v_toFun_498_);
v___x_500_ = lean_apply_1(v_toFun_498_, v_l_496_);
v___x_501_ = lean_box(0);
v___x_502_ = lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0___redArg(v_f_495_, v___x_500_, v___x_501_);
lean_inc(v_toFun_499_);
v___x_503_ = lean_apply_1(v_toFun_499_, v___x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map___redArg(lean_object* v_f_504_){
_start:
{
lean_object* v___f_505_; 
v___f_505_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_505_, 0, v_f_504_);
return v___f_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_map(lean_object* v_00_u03b1_506_, lean_object* v_00_u03b2_507_, lean_object* v_f_508_){
_start:
{
lean_object* v___f_509_; 
v___f_509_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_509_, 0, v_f_508_);
return v___f_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0(lean_object* v_00_u03b1_510_, lean_object* v_00_u03b2_511_, lean_object* v_f_512_, lean_object* v_a_513_, lean_object* v_a_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_mathlib_List_mapTR_loop___at___00FreeMonoid_map_spec__0___redArg(v_f_512_, v_a_513_, v_a_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_map___redArg(lean_object* v_f_516_){
_start:
{
lean_object* v___f_517_; 
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_517_, 0, v_f_516_);
return v___f_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_map(lean_object* v_00_u03b1_518_, lean_object* v_00_u03b2_519_, lean_object* v_f_520_){
_start:
{
lean_object* v___f_521_; 
v___f_521_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_521_, 0, v_f_520_);
return v___f_521_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_uniqueUnits___closed__0(void){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_FreeMonoid_instCancelMonoid(lean_box(0));
return v___x_522_;
}
}
static lean_object* _init_lp_mathlib_FreeMonoid_uniqueUnits___closed__1(void){
_start:
{
lean_object* v___x_523_; lean_object* v___x_524_; 
v___x_523_ = lean_obj_once(&lp_mathlib_FreeMonoid_uniqueUnits___closed__0, &lp_mathlib_FreeMonoid_uniqueUnits___closed__0_once, _init_lp_mathlib_FreeMonoid_uniqueUnits___closed__0);
v___x_524_ = lp_mathlib_Units_instInhabited___redArg(v___x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_uniqueUnits(lean_object* v_00_u03b1_525_){
_start:
{
lean_object* v___x_526_; 
v___x_526_ = lean_obj_once(&lp_mathlib_FreeMonoid_uniqueUnits___closed__1, &lp_mathlib_FreeMonoid_uniqueUnits___closed__1_once, _init_lp_mathlib_FreeMonoid_uniqueUnits___closed__1);
return v___x_526_;
}
}
static lean_object* _init_lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0(void){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = lp_mathlib_FreeAddMonoid_instAddCancelMonoid(lean_box(0));
return v___x_527_;
}
}
static lean_object* _init_lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1(void){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = lean_obj_once(&lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0, &lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0_once, _init_lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__0);
v___x_529_ = lp_mathlib_AddUnits_instInhabited___redArg(v___x_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_uniqueAddUnits(lean_object* v_00_u03b1_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lean_obj_once(&lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1, &lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1_once, _init_lp_mathlib_FreeAddMonoid_uniqueAddUnits___closed__1);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_reverse___redArg(lean_object* v_as_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = l_List_reverse___redArg(v_as_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_reverse(lean_object* v_00_u03b1_534_, lean_object* v_as_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = l_List_reverse___redArg(v_as_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_reverse___redArg(lean_object* v_as_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = l_List_reverse___redArg(v_as_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_reverse(lean_object* v_00_u03b1_539_, lean_object* v_as_540_){
_start:
{
lean_object* v___x_541_; 
v___x_541_ = l_List_reverse___redArg(v_as_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__0(lean_object* v_e_542_, lean_object* v___y_543_){
_start:
{
lean_object* v_toFun_544_; lean_object* v___x_545_; 
v_toFun_544_ = lean_ctor_get(v_e_542_, 0);
lean_inc(v_toFun_544_);
lean_dec_ref(v_e_542_);
v___x_545_ = lean_apply_1(v_toFun_544_, v___y_543_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1(lean_object* v___f_546_, lean_object* v___y_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_mathlib_FreeMonoid_map___redArg___lam__0(v___f_546_, v___y_547_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__2(lean_object* v___x_549_, lean_object* v___y_550_){
_start:
{
lean_object* v_toFun_551_; lean_object* v___x_552_; 
v_toFun_551_ = lean_ctor_get(v___x_549_, 0);
lean_inc(v_toFun_551_);
lean_dec_ref(v___x_549_);
v___x_552_ = lean_apply_1(v_toFun_551_, v___y_550_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr___redArg(lean_object* v_e_553_){
_start:
{
lean_object* v___f_554_; lean_object* v___f_555_; lean_object* v___x_556_; lean_object* v___f_557_; lean_object* v___f_558_; lean_object* v___x_559_; 
lean_inc_ref(v_e_553_);
v___f_554_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_554_, 0, v_e_553_);
v___f_555_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_555_, 0, v___f_554_);
v___x_556_ = lp_mathlib_Equiv_symm___redArg(v_e_553_);
v___f_557_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_557_, 0, v___x_556_);
v___f_558_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_558_, 0, v___f_557_);
v___x_559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_559_, 0, v___f_555_);
lean_ctor_set(v___x_559_, 1, v___f_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeMonoid_freeMonoidCongr(lean_object* v_00_u03b1_560_, lean_object* v_00_u03b2_561_, lean_object* v_e_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lp_mathlib_FreeMonoid_freeMonoidCongr___redArg(v_e_562_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_freeAddMonoidCongr___redArg(lean_object* v_e_564_){
_start:
{
lean_object* v___f_565_; lean_object* v___f_566_; lean_object* v___x_567_; lean_object* v___f_568_; lean_object* v___f_569_; lean_object* v___x_570_; 
lean_inc_ref(v_e_564_);
v___f_565_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_565_, 0, v_e_564_);
v___f_566_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_566_, 0, v___f_565_);
v___x_567_ = lp_mathlib_Equiv_symm___redArg(v_e_564_);
v___f_568_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_568_, 0, v___x_567_);
v___f_569_ = lean_alloc_closure((void*)(lp_mathlib_FreeMonoid_freeMonoidCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_569_, 0, v___f_568_);
v___x_570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_570_, 0, v___f_566_);
lean_ctor_set(v___x_570_, 1, v___f_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAddMonoid_freeAddMonoidCongr(lean_object* v_00_u03b1_571_, lean_object* v_00_u03b2_572_, lean_object* v_e_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_mathlib_FreeAddMonoid_freeAddMonoidCongr___redArg(v_e_573_);
return v___x_574_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
