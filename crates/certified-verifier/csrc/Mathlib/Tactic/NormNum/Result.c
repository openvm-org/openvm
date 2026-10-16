// Lean compiler output
// Module: Mathlib.Tactic.NormNum.Result
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.GroupWithZero.Invertible public import Mathlib.Algebra.Ring.Nat public import Mathlib.Data.Int.Cast.Basic public meta import Mathlib.Data.Sigma.Basic
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
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_int_neg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_rawNatLit_x3f(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_instInhabitedQuoted(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "not an AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "not a semiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not a ring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__2_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__5_value),LEAN_SCALAR_PTR_LITERAL(192, 66, 133, 102, 95, 170, 134, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mkRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__0_value),LEAN_SCALAR_PTR_LITERAL(220, 174, 118, 125, 113, 89, 9, 133)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit(lean_object*);
static const lean_string_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "natAbs"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__5_value),LEAN_SCALAR_PTR_LITERAL(255, 186, 174, 182, 213, 167, 94, 168)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__8_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Mathlib.Tactic.NormNum.Result"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Meta.NormNum.rawIntLitNatAbs"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "not a raw integer literal"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "natAbs_neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__15_value),LEAN_SCALAR_PTR_LITERAL(20, 139, 134, 104, 245, 108, 105, 217)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instNatCastInt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__22_value),LEAN_SCALAR_PTR_LITERAL(116, 224, 75, 57, 255, 108, 159, 197)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "natAbs_natCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__25_value),LEAN_SCALAR_PTR_LITERAL(149, 210, 88, 108, 140, 3, 25, 202)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Rat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__2_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__5_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__6_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__7_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "AddZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(171, 135, 49, 0, 6, 244, 57, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__10_value),LEAN_SCALAR_PTR_LITERAL(87, 27, 84, 210, 142, 102, 48, 129)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toAddZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__12_value),LEAN_SCALAR_PTR_LITERAL(157, 204, 59, 233, 207, 78, 141, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__13_value),LEAN_SCALAR_PTR_LITERAL(64, 236, 134, 119, 35, 182, 73, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__15_value),LEAN_SCALAR_PTR_LITERAL(110, 12, 45, 85, 216, 81, 49, 169)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 217, 102, 131, 1, 241, 19, 50)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toAddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__18_value),LEAN_SCALAR_PTR_LITERAL(231, 178, 143, 16, 208, 220, 52, 201)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "cast_zero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__23_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__24_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__26_value),LEAN_SCALAR_PTR_LITERAL(52, 219, 71, 246, 148, 114, 208, 126)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_one"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NormNum"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instAtLeastTwo"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__32_value),LEAN_SCALAR_PTR_LITERAL(18, 129, 239, 212, 53, 125, 191, 140)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "instOfNatAtLeastTwo"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__35_value),LEAN_SCALAR_PTR_LITERAL(223, 182, 28, 70, 145, 92, 58, 230)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toNatCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__37_value),LEAN_SCALAR_PTR_LITERAL(83, 227, 187, 63, 172, 112, 247, 90)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__39_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__41_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instOfNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__46_value),LEAN_SCALAR_PTR_LITERAL(217, 182, 143, 149, 136, 99, 16, 5)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__46_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 253, 199, 38, 151, 242, 146)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__52_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__55_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__56_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_rawCast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_rawCast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_rawCast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_rawCast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_rawCast___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_rawCast(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_rawCast___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_rawCast(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isBool_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isBool_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNat_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNat_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNat_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNat_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNNRat_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNNRat_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNNRat_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNNRat_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___aux__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "instAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(129, 65, 157, 144, 0, 78, 170, 16)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsInt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "to_isNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__3_value),LEAN_SCALAR_PTR_LITERAL(161, 211, 145, 221, 239, 118, 69, 153)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instAddMonoidWithOne'"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 200, 200, 23, 119, 176, 249, 135)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "DivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__3_value),LEAN_SCALAR_PTR_LITERAL(241, 35, 131, 210, 203, 216, 146, 177)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "IsNNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__3_value),LEAN_SCALAR_PTR_LITERAL(127, 20, 209, 31, 69, 39, 204, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toDivisionSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__2_value),LEAN_SCALAR_PTR_LITERAL(66, 176, 51, 228, 18, 108, 54, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "to_isNNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__5_value),LEAN_SCALAR_PTR_LITERAL(157, 69, 51, 38, 43, 3, 204, 56)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__7_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "to_isInt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(150, 34, 5, 103, 119, 62, 99, 26)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isFalse ("};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isTrue ("};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "isNat "};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isNegNat "};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isNNRat "};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "isNegNNRat "};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "den_nz"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 12, 45, 45, 118, 187, 101, 38)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 69, 72, 86, 50, 41, 73, 149)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "IsNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__9_value),LEAN_SCALAR_PTR_LITERAL(49, 26, 56, 221, 172, 175, 154, 247)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___auto__1;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__5_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 17, 209, 89, 30, 202, 18)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___auto__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "to_isRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(215, 166, 136, 182, 114, 45, 147, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__3_value),LEAN_SCALAR_PTR_LITERAL(236, 38, 194, 105, 137, 30, 136, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(9, 6, 62, 67, 54, 203, 3, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "eq_false"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__3_value),LEAN_SCALAR_PTR_LITERAL(242, 127, 91, 199, 130, 171, 29, 27)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__6_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_true"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__9_value),LEAN_SCALAR_PTR_LITERAL(50, 213, 255, 45, 151, 209, 83, 175)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rawCast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "to_raw_eq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(34, 248, 165, 88, 27, 2, 30, 241)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12_value),LEAN_SCALAR_PTR_LITERAL(72, 208, 251, 102, 233, 243, 211, 50)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(195, 233, 72, 51, 191, 3, 58, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__17_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12_value),LEAN_SCALAR_PTR_LITERAL(169, 80, 136, 140, 138, 237, 112, 64)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(157, 141, 199, 51, 4, 195, 149, 251)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12_value),LEAN_SCALAR_PTR_LITERAL(218, 237, 153, 176, 238, 25, 53, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__13_value),LEAN_SCALAR_PTR_LITERAL(189, 154, 84, 247, 95, 101, 64, 140)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawIntEq(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__6 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Meta.NormNum.Result.ofRawNat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "not a raw nat cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "of_raw"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value),LEAN_SCALAR_PTR_LITERAL(3, 193, 0, 151, 166, 192, 33, 62)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Meta.NormNum.Result.ofRawInt"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "not a raw int cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value),LEAN_SCALAR_PTR_LITERAL(226, 118, 187, 242, 94, 83, 130, 191)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNNRat_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "Mathlib.Meta.NormNum.Result.ofRawNNRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "not a raw nnrat cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value),LEAN_SCALAR_PTR_LITERAL(20, 158, 163, 9, 7, 155, 218, 40)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Meta.NormNum.Result.ofRawRat"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "not a raw rat cast"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__3_value),LEAN_SCALAR_PTR_LITERAL(116, 186, 94, 204, 8, 149, 21, 152)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "to_eq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 155, 29, 215, 133, 193, 236, 170)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "AddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__2_value),LEAN_SCALAR_PTR_LITERAL(126, 216, 146, 120, 99, 62, 20, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__3_value),LEAN_SCALAR_PTR_LITERAL(172, 33, 204, 185, 213, 137, 110, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "NonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "toAddCommMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__5_value),LEAN_SCALAR_PTR_LITERAL(46, 119, 91, 198, 213, 11, 55, 139)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__6_value),LEAN_SCALAR_PTR_LITERAL(2, 121, 193, 151, 116, 56, 170, 8)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toNonAssocSemiring"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__8_value),LEAN_SCALAR_PTR_LITERAL(146, 92, 66, 67, 127, 202, 60, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__10_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__11_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toNeg"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__13_value),LEAN_SCALAR_PTR_LITERAL(156, 44, 233, 53, 1, 106, 24, 217)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__14_value),LEAN_SCALAR_PTR_LITERAL(124, 136, 108, 160, 134, 153, 101, 8)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toNegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__16_value),LEAN_SCALAR_PTR_LITERAL(135, 233, 160, 34, 207, 245, 132, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__17_value),LEAN_SCALAR_PTR_LITERAL(107, 179, 145, 12, 37, 42, 18, 108)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toSubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__19_value),LEAN_SCALAR_PTR_LITERAL(203, 24, 17, 79, 61, 156, 198, 150)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__20_value),LEAN_SCALAR_PTR_LITERAL(94, 234, 159, 237, 9, 124, 201, 94)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "SubtractionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toSubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__22_value),LEAN_SCALAR_PTR_LITERAL(100, 8, 183, 201, 110, 57, 85, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__23_value),LEAN_SCALAR_PTR_LITERAL(203, 26, 135, 240, 118, 74, 112, 111)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toDivisionAddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__25_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__26_value),LEAN_SCALAR_PTR_LITERAL(65, 138, 55, 164, 85, 246, 87, 209)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__28_value),LEAN_SCALAR_PTR_LITERAL(121, 151, 225, 139, 113, 68, 25, 156)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "neg_to_eq"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__30_value),LEAN_SCALAR_PTR_LITERAL(108, 220, 243, 131, 196, 108, 207, 74)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__32_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__33_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__35_value),LEAN_SCALAR_PTR_LITERAL(34, 70, 113, 198, 157, 211, 131, 18)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toDiv"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__37_value),LEAN_SCALAR_PTR_LITERAL(231, 106, 236, 89, 112, 21, 122, 113)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__38_value),LEAN_SCALAR_PTR_LITERAL(95, 209, 62, 72, 37, 30, 170, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__40_value),LEAN_SCALAR_PTR_LITERAL(55, 132, 4, 209, 65, 207, 153, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__41_value),LEAN_SCALAR_PTR_LITERAL(172, 54, 25, 155, 165, 99, 150, 23)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 208, 64, 71, 63, 26, 215, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__43_value),LEAN_SCALAR_PTR_LITERAL(164, 129, 71, 97, 30, 189, 214, 64)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 141, 87, 66, 190, 132, 81, 100)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__1_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__41_value),LEAN_SCALAR_PTR_LITERAL(157, 154, 239, 235, 210, 195, 14, 77)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__30_value),LEAN_SCALAR_PTR_LITERAL(234, 180, 6, 13, 185, 1, 121, 255)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rec"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__3_value),LEAN_SCALAR_PTR_LITERAL(86, 17, 7, 2, 233, 148, 36, 75)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__9_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__11_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "symm"};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__17_value),LEAN_SCALAR_PTR_LITERAL(220, 149, 144, 59, 77, 93, 25, 217)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 144, 12, 127, 73, 247, 143, 14)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 140, 236, 194, 147, 62, 208, 210)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 242, 99, 215, 32, 214, 250, 222)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__29_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__30_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__31_value),LEAN_SCALAR_PTR_LITERAL(233, 114, 34, 138, 32, 245, 157, 89)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__4_value),LEAN_SCALAR_PTR_LITERAL(231, 161, 84, 175, 195, 12, 78, 146)}};
static const lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne_x27___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1_);
v___x_3_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne_x27(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne_x27___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v_toAddMonoidWithOne_9_; 
v___x_8_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_7_);
v_toAddMonoidWithOne_9_ = lean_ctor_get(v___x_8_, 1);
lean_inc_ref(v_toAddMonoidWithOne_9_);
lean_dec_ref(v___x_8_);
return v_toAddMonoidWithOne_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Mathlib_Meta_NormNum_instAddMonoidWithOne___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0(lean_object* v_msgData_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
lean_object* v___x_19_; lean_object* v_env_20_; lean_object* v___x_21_; lean_object* v_mctx_22_; lean_object* v_lctx_23_; lean_object* v_options_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_19_ = lean_st_ref_get(v___y_17_);
v_env_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc_ref(v_env_20_);
lean_dec(v___x_19_);
v___x_21_ = lean_st_ref_get(v___y_15_);
v_mctx_22_ = lean_ctor_get(v___x_21_, 0);
lean_inc_ref(v_mctx_22_);
lean_dec(v___x_21_);
v_lctx_23_ = lean_ctor_get(v___y_14_, 2);
v_options_24_ = lean_ctor_get(v___y_16_, 2);
lean_inc_ref(v_options_24_);
lean_inc_ref(v_lctx_23_);
v___x_25_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_25_, 0, v_env_20_);
lean_ctor_set(v___x_25_, 1, v_mctx_22_);
lean_ctor_set(v___x_25_, 2, v_lctx_23_);
lean_ctor_set(v___x_25_, 3, v_options_24_);
v___x_26_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
lean_ctor_set(v___x_26_, 1, v_msgData_13_);
v___x_27_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0___boxed(lean_object* v_msgData_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0(v_msgData_28_, v___y_29_, v___y_30_, v___y_31_, v___y_32_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(lean_object* v_msg_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
lean_object* v_ref_41_; lean_object* v___x_42_; lean_object* v_a_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_51_; 
v_ref_41_ = lean_ctor_get(v___y_38_, 5);
v___x_42_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0_spec__0(v_msg_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
v_a_43_ = lean_ctor_get(v___x_42_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_42_);
if (v_isSharedCheck_51_ == 0)
{
v___x_45_ = v___x_42_;
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_a_43_);
lean_dec(v___x_42_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_47_; lean_object* v___x_49_; 
lean_inc(v_ref_41_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v_ref_41_);
lean_ctor_set(v___x_47_, 1, v_a_43_);
if (v_isShared_46_ == 0)
{
lean_ctor_set_tag(v___x_45_, 1);
lean_ctor_set(v___x_45_, 0, v___x_47_);
v___x_49_ = v___x_45_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v___x_47_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg___boxed(lean_object* v_msg_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v_msg_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
return v_res_58_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__2));
v___x_64_ = l_Lean_stringToMessageData(v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne(lean_object* v_u_65_, lean_object* v_00_u03b1_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = l_Lean_Meta_saveState___redArg(v_a_68_, v_a_70_);
if (lean_obj_tag(v___x_72_) == 0)
{
lean_object* v_a_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_a_73_ = lean_ctor_get(v___x_72_, 0);
lean_inc(v_a_73_);
lean_dec_ref_known(v___x_72_, 1);
v___x_74_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__1));
v___x_75_ = lean_box(0);
v___x_76_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_76_, 0, v_u_65_);
lean_ctor_set(v___x_76_, 1, v___x_75_);
v___x_77_ = l_Lean_Expr_const___override(v___x_74_, v___x_76_);
v___x_78_ = l_Lean_Expr_app___override(v___x_77_, v_00_u03b1_66_);
v___x_79_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_78_, v_a_67_, v_a_68_, v_a_69_, v_a_70_);
if (lean_obj_tag(v___x_79_) == 0)
{
lean_dec(v_a_73_);
return v___x_79_;
}
else
{
lean_object* v_a_80_; uint8_t v___y_82_; uint8_t v___x_94_; 
v_a_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc(v_a_80_);
v___x_94_ = l_Lean_Exception_isInterrupt(v_a_80_);
if (v___x_94_ == 0)
{
uint8_t v___x_95_; 
v___x_95_ = l_Lean_Exception_isRuntime(v_a_80_);
v___y_82_ = v___x_95_;
goto v___jp_81_;
}
else
{
lean_dec(v_a_80_);
v___y_82_ = v___x_94_;
goto v___jp_81_;
}
v___jp_81_:
{
if (v___y_82_ == 0)
{
lean_object* v___x_83_; 
lean_dec_ref_known(v___x_79_, 1);
v___x_83_ = l_Lean_Meta_SavedState_restore___redArg(v_a_73_, v_a_68_, v_a_70_);
lean_dec(v_a_73_);
if (lean_obj_tag(v___x_83_) == 0)
{
lean_object* v___x_84_; lean_object* v___x_85_; 
lean_dec_ref_known(v___x_83_, 1);
v___x_84_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___closed__3);
v___x_85_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v___x_84_, v_a_67_, v_a_68_, v_a_69_, v_a_70_);
return v___x_85_;
}
else
{
lean_object* v_a_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_93_; 
v_a_86_ = lean_ctor_get(v___x_83_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v___x_83_);
if (v_isSharedCheck_93_ == 0)
{
v___x_88_ = v___x_83_;
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_a_86_);
lean_dec(v___x_83_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_91_; 
if (v_isShared_89_ == 0)
{
v___x_91_ = v___x_88_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v_a_86_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
}
else
{
lean_dec(v_a_73_);
return v___x_79_;
}
}
}
}
else
{
lean_object* v_a_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_103_; 
lean_dec_ref(v_00_u03b1_66_);
lean_dec(v_u_65_);
v_a_96_ = lean_ctor_get(v___x_72_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_103_ == 0)
{
v___x_98_ = v___x_72_;
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_a_96_);
lean_dec(v___x_72_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___x_101_; 
if (v_isShared_99_ == 0)
{
v___x_101_ = v___x_98_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_a_96_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne___boxed(lean_object* v_u_104_, lean_object* v_00_u03b1_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Mathlib_Meta_NormNum_inferAddMonoidWithOne(v_u_104_, v_00_u03b1_105_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
lean_dec(v_a_109_);
lean_dec_ref(v_a_108_);
lean_dec(v_a_107_);
lean_dec_ref(v_a_106_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0(lean_object* v_00_u03b1_112_, lean_object* v_msg_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v_msg_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___boxed(lean_object* v_00_u03b1_120_, lean_object* v_msg_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0(v_00_u03b1_120_, v_msg_121_, v___y_122_, v___y_123_, v___y_124_, v___y_125_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
lean_dec(v___y_123_);
lean_dec_ref(v___y_122_);
return v_res_127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__2));
v___x_133_ = l_Lean_stringToMessageData(v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring(lean_object* v_u_134_, lean_object* v_00_u03b1_135_, lean_object* v_a_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = l_Lean_Meta_saveState___redArg(v_a_137_, v_a_139_);
if (lean_obj_tag(v___x_141_) == 0)
{
lean_object* v_a_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_a_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc(v_a_142_);
lean_dec_ref_known(v___x_141_, 1);
v___x_143_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__1));
v___x_144_ = lean_box(0);
v___x_145_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_145_, 0, v_u_134_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
v___x_146_ = l_Lean_Expr_const___override(v___x_143_, v___x_145_);
v___x_147_ = l_Lean_Expr_app___override(v___x_146_, v_00_u03b1_135_);
v___x_148_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_147_, v_a_136_, v_a_137_, v_a_138_, v_a_139_);
if (lean_obj_tag(v___x_148_) == 0)
{
lean_dec(v_a_142_);
return v___x_148_;
}
else
{
lean_object* v_a_149_; uint8_t v___y_151_; uint8_t v___x_163_; 
v_a_149_ = lean_ctor_get(v___x_148_, 0);
lean_inc(v_a_149_);
v___x_163_ = l_Lean_Exception_isInterrupt(v_a_149_);
if (v___x_163_ == 0)
{
uint8_t v___x_164_; 
v___x_164_ = l_Lean_Exception_isRuntime(v_a_149_);
v___y_151_ = v___x_164_;
goto v___jp_150_;
}
else
{
lean_dec(v_a_149_);
v___y_151_ = v___x_163_;
goto v___jp_150_;
}
v___jp_150_:
{
if (v___y_151_ == 0)
{
lean_object* v___x_152_; 
lean_dec_ref_known(v___x_148_, 1);
v___x_152_ = l_Lean_Meta_SavedState_restore___redArg(v_a_142_, v_a_137_, v_a_139_);
lean_dec(v_a_142_);
if (lean_obj_tag(v___x_152_) == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
lean_dec_ref_known(v___x_152_, 1);
v___x_153_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___closed__3);
v___x_154_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v___x_153_, v_a_136_, v_a_137_, v_a_138_, v_a_139_);
return v___x_154_;
}
else
{
lean_object* v_a_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
v_a_155_ = lean_ctor_get(v___x_152_, 0);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_152_);
if (v_isSharedCheck_162_ == 0)
{
v___x_157_ = v___x_152_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_a_155_);
lean_dec(v___x_152_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_a_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
else
{
lean_dec(v_a_142_);
return v___x_148_;
}
}
}
}
else
{
lean_object* v_a_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_172_; 
lean_dec_ref(v_00_u03b1_135_);
lean_dec(v_u_134_);
v_a_165_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_172_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_172_ == 0)
{
v___x_167_ = v___x_141_;
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_a_165_);
lean_dec(v___x_141_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_172_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v___x_170_; 
if (v_isShared_168_ == 0)
{
v___x_170_ = v___x_167_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v_a_165_);
v___x_170_ = v_reuseFailAlloc_171_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
return v___x_170_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferSemiring___boxed(lean_object* v_u_173_, lean_object* v_00_u03b1_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Mathlib_Meta_NormNum_inferSemiring(v_u_173_, v_00_u03b1_174_, v_a_175_, v_a_176_, v_a_177_, v_a_178_);
lean_dec(v_a_178_);
lean_dec_ref(v_a_177_);
lean_dec(v_a_176_);
lean_dec_ref(v_a_175_);
return v_res_180_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_185_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__2));
v___x_186_ = l_Lean_stringToMessageData(v___x_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing(lean_object* v_u_187_, lean_object* v_00_u03b1_188_, lean_object* v_a_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = l_Lean_Meta_saveState___redArg(v_a_190_, v_a_192_);
if (lean_obj_tag(v___x_194_) == 0)
{
lean_object* v_a_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
v_a_195_ = lean_ctor_get(v___x_194_, 0);
lean_inc(v_a_195_);
lean_dec_ref_known(v___x_194_, 1);
v___x_196_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__1));
v___x_197_ = lean_box(0);
v___x_198_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_198_, 0, v_u_187_);
lean_ctor_set(v___x_198_, 1, v___x_197_);
v___x_199_ = l_Lean_Expr_const___override(v___x_196_, v___x_198_);
v___x_200_ = l_Lean_Expr_app___override(v___x_199_, v_00_u03b1_188_);
v___x_201_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_200_, v_a_189_, v_a_190_, v_a_191_, v_a_192_);
if (lean_obj_tag(v___x_201_) == 0)
{
lean_dec(v_a_195_);
return v___x_201_;
}
else
{
lean_object* v_a_202_; uint8_t v___y_204_; uint8_t v___x_216_; 
v_a_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_a_202_);
v___x_216_ = l_Lean_Exception_isInterrupt(v_a_202_);
if (v___x_216_ == 0)
{
uint8_t v___x_217_; 
v___x_217_ = l_Lean_Exception_isRuntime(v_a_202_);
v___y_204_ = v___x_217_;
goto v___jp_203_;
}
else
{
lean_dec(v_a_202_);
v___y_204_ = v___x_216_;
goto v___jp_203_;
}
v___jp_203_:
{
if (v___y_204_ == 0)
{
lean_object* v___x_205_; 
lean_dec_ref_known(v___x_201_, 1);
v___x_205_ = l_Lean_Meta_SavedState_restore___redArg(v_a_195_, v_a_190_, v_a_192_);
lean_dec(v_a_195_);
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v___x_206_; lean_object* v___x_207_; 
lean_dec_ref_known(v___x_205_, 1);
v___x_206_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_inferRing___closed__3);
v___x_207_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v___x_206_, v_a_189_, v_a_190_, v_a_191_, v_a_192_);
return v___x_207_;
}
else
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_215_; 
v_a_208_ = lean_ctor_get(v___x_205_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_215_ == 0)
{
v___x_210_ = v___x_205_;
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_205_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_213_; 
if (v_isShared_211_ == 0)
{
v___x_213_ = v___x_210_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v_a_208_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
}
else
{
lean_dec(v_a_195_);
return v___x_201_;
}
}
}
}
else
{
lean_object* v_a_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_225_; 
lean_dec_ref(v_00_u03b1_188_);
lean_dec(v_u_187_);
v_a_218_ = lean_ctor_get(v___x_194_, 0);
v_isSharedCheck_225_ = !lean_is_exclusive(v___x_194_);
if (v_isSharedCheck_225_ == 0)
{
v___x_220_ = v___x_194_;
v_isShared_221_ = v_isSharedCheck_225_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_a_218_);
lean_dec(v___x_194_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_225_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v___x_223_; 
if (v_isShared_221_ == 0)
{
v___x_223_ = v___x_220_;
goto v_reusejp_222_;
}
else
{
lean_object* v_reuseFailAlloc_224_; 
v_reuseFailAlloc_224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_224_, 0, v_a_218_);
v___x_223_ = v_reuseFailAlloc_224_;
goto v_reusejp_222_;
}
v_reusejp_222_:
{
return v___x_223_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_inferRing___boxed(lean_object* v_u_226_, lean_object* v_00_u03b1_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Mathlib_Meta_NormNum_inferRing(v_u_226_, v_00_u03b1_227_, v_a_228_, v_a_229_, v_a_230_, v_a_231_);
lean_dec(v_a_231_);
lean_dec_ref(v_a_230_);
lean_dec(v_a_229_);
lean_dec_ref(v_a_228_);
return v_res_233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_234_ = lean_unsigned_to_nat(0u);
v___x_235_ = lean_nat_to_int(v___x_234_);
return v___x_235_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4(void){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_241_ = lean_box(0);
v___x_242_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3));
v___x_243_ = l_Lean_Expr_const___override(v___x_242_, v___x_241_);
return v___x_243_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7(void){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_248_ = lean_box(0);
v___x_249_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6));
v___x_250_ = l_Lean_Expr_const___override(v___x_249_, v___x_248_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(lean_object* v_n_251_){
_start:
{
lean_object* v___x_252_; lean_object* v_lit_253_; lean_object* v___x_254_; uint8_t v___x_255_; 
v___x_252_ = lean_nat_abs(v_n_251_);
v_lit_253_ = l_Lean_mkRawNatLit(v___x_252_);
v___x_254_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0);
v___x_255_ = lean_int_dec_le(v___x_254_, v_n_251_);
if (v___x_255_ == 0)
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_257_ = l_Lean_Expr_app___override(v___x_256_, v_lit_253_);
return v___x_257_;
}
else
{
lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_258_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7);
v___x_259_ = l_Lean_Expr_app___override(v___x_258_, v_lit_253_);
return v___x_259_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___boxed(lean_object* v_n_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_n_260_);
lean_dec(v_n_260_);
return v_res_261_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_265_ = lean_box(0);
v___x_266_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__1));
v___x_267_ = l_Lean_Expr_const___override(v___x_266_, v___x_265_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit(lean_object* v_q_268_){
_start:
{
lean_object* v_num_269_; lean_object* v_den_270_; lean_object* v_nlit_271_; lean_object* v_dlit_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v_num_269_ = lean_ctor_get(v_q_268_, 0);
lean_inc(v_num_269_);
v_den_270_ = lean_ctor_get(v_q_268_, 1);
lean_inc(v_den_270_);
lean_dec_ref(v_q_268_);
v_nlit_271_ = lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit(v_num_269_);
lean_dec(v_num_269_);
v_dlit_272_ = l_Lean_mkRawNatLit(v_den_270_);
v___x_273_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawRatLit___closed__2);
v___x_274_ = l_Lean_Expr_app___override(v___x_273_, v_nlit_271_);
v___x_275_ = l_Lean_Expr_app___override(v___x_274_, v_dlit_272_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0(lean_object* v___x_277_, lean_object* v_msg_278_){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_279_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_280_ = l_Lean_Name_mkStr1(v___x_279_);
v___x_281_ = lean_box(0);
v___x_282_ = l_Lean_Expr_const___override(v___x_280_, v___x_281_);
v___x_283_ = lp_Qq_Qq_instInhabitedQuoted(v___x_282_);
lean_dec_ref(v___x_282_);
v___x_284_ = lp_Qq_Qq_instInhabitedQuoted(v___x_277_);
v___x_285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_285_, 0, v___x_283_);
lean_ctor_set(v___x_285_, 1, v___x_284_);
v___x_286_ = lean_panic_fn_borrowed(v___x_285_, v_msg_278_);
lean_dec_ref_known(v___x_285_, 2);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___boxed(lean_object* v___x_287_, lean_object* v_msg_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0(v___x_287_, v_msg_288_);
lean_dec_ref(v___x_287_);
return v_res_289_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_293_ = lean_box(0);
v___x_294_ = l_Lean_Level_succ___override(v___x_293_);
return v___x_294_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3(void){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_295_ = lean_box(0);
v___x_296_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__2);
v___x_297_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v___x_295_);
return v___x_297_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_298_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3);
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1));
v___x_300_ = l_Lean_Expr_const___override(v___x_299_, v___x_298_);
return v___x_300_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7(void){
_start:
{
lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_305_ = lean_box(0);
v___x_306_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__6));
v___x_307_ = l_Lean_Expr_const___override(v___x_306_, v___x_305_);
return v___x_307_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10(void){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_311_ = lean_box(0);
v___x_312_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__9));
v___x_313_ = l_Lean_Expr_const___override(v___x_312_, v___x_311_);
return v___x_313_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__13));
v___x_318_ = lean_unsigned_to_nat(4u);
v___x_319_ = lean_unsigned_to_nat(101u);
v___x_320_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__12));
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11));
v___x_322_ = l_mkPanicMessageWithDecl(v___x_321_, v___x_320_, v___x_319_, v___x_318_, v___x_317_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17(void){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_327_ = lean_box(0);
v___x_328_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__16));
v___x_329_ = l_Lean_Expr_const___override(v___x_328_, v___x_327_);
return v___x_329_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21(void){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_336_ = lean_box(0);
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__20));
v___x_338_ = l_Lean_Expr_const___override(v___x_337_, v___x_336_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_342_ = lean_box(0);
v___x_343_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__23));
v___x_344_ = l_Lean_Expr_const___override(v___x_343_, v___x_342_);
return v___x_344_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = lean_box(0);
v___x_350_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__26));
v___x_351_ = l_Lean_Expr_const___override(v___x_350_, v___x_349_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs(lean_object* v_n_352_){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; uint8_t v___x_355_; 
v___x_353_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__6));
v___x_354_ = lean_unsigned_to_nat(1u);
v___x_355_ = l_Lean_Expr_isAppOfArity(v_n_352_, v___x_353_, v___x_354_);
if (v___x_355_ == 0)
{
lean_object* v___x_356_; uint8_t v___x_357_; 
v___x_356_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__3));
v___x_357_ = l_Lean_Expr_isAppOfArity(v_n_352_, v___x_356_, v___x_354_);
if (v___x_357_ == 0)
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_358_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_359_ = l_Lean_Name_mkStr1(v___x_358_);
v___x_360_ = lean_box(0);
v___x_361_ = l_Lean_Expr_const___override(v___x_359_, v___x_360_);
v___x_362_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4);
v___x_363_ = l_Lean_Expr_app___override(v___x_362_, v___x_361_);
v___x_364_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__7);
v___x_365_ = l_Lean_Expr_app___override(v___x_364_, v_n_352_);
v___x_366_ = l_Lean_Expr_app___override(v___x_363_, v___x_365_);
v___x_367_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10);
v___x_368_ = l_Lean_Expr_app___override(v___x_366_, v___x_367_);
v___x_369_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__14);
v___x_370_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0(v___x_368_, v___x_369_);
lean_dec_ref(v___x_368_);
return v___x_370_;
}
else
{
lean_object* v_m_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v_this_383_; lean_object* v___x_384_; 
v_m_371_ = l_Lean_Expr_appArg_x21(v_n_352_);
lean_dec_ref(v_n_352_);
v___x_372_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_373_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__17);
v___x_374_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__18));
v___x_375_ = l_Lean_Name_mkStr2(v___x_372_, v___x_374_);
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__19));
v___x_377_ = l_Lean_Expr_const___override(v___x_375_, v___x_376_);
v___x_378_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21);
v___x_379_ = l_Lean_Expr_app___override(v___x_377_, v___x_378_);
v___x_380_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__24);
v___x_381_ = l_Lean_Expr_app___override(v___x_379_, v___x_380_);
lean_inc_ref(v_m_371_);
v___x_382_ = l_Lean_Expr_app___override(v___x_381_, v_m_371_);
v_this_383_ = l_Lean_Expr_app___override(v___x_373_, v___x_382_);
v___x_384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_384_, 0, v_m_371_);
lean_ctor_set(v___x_384_, 1, v_this_383_);
return v___x_384_;
}
}
else
{
lean_object* v_m_385_; lean_object* v___x_386_; lean_object* v_this_387_; lean_object* v___x_388_; 
v_m_385_ = l_Lean_Expr_appArg_x21(v_n_352_);
lean_dec_ref(v_n_352_);
v___x_386_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__27);
lean_inc_ref(v_m_385_);
v_this_387_ = l_Lean_Expr_app___override(v___x_386_, v_m_385_);
v___x_388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_388_, 0, v_m_385_);
lean_ctor_set(v___x_388_, 1, v_this_387_);
return v___x_388_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__4));
v___x_399_ = l_Lean_Expr_lit___override(v___x_398_);
return v___x_399_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22(void){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_427_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__21));
v___x_428_ = l_Lean_Expr_lit___override(v___x_427_);
return v___x_428_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34(void){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; 
v___x_448_ = lean_box(0);
v___x_449_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__33));
v___x_450_ = l_Lean_Expr_const___override(v___x_449_, v___x_448_);
return v___x_450_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42(void){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__41));
v___x_464_ = l_Lean_stringToMessageData(v___x_463_);
return v___x_464_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_465_ = lean_box(0);
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__1));
v___x_467_ = l_Lean_Expr_const___override(v___x_466_, v___x_465_);
return v___x_467_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44(void){
_start:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_468_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__19));
v___x_469_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3));
v___x_470_ = l_Lean_Expr_const___override(v___x_469_, v___x_468_);
return v___x_470_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45(void){
_start:
{
lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_471_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43);
v___x_472_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44);
v___x_473_ = l_Lean_Expr_app___override(v___x_472_, v___x_471_);
return v___x_473_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48(void){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_478_ = lean_box(0);
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__47));
v___x_480_ = l_Lean_Expr_const___override(v___x_479_, v___x_478_);
return v___x_480_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49(void){
_start:
{
lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; 
v___x_481_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3);
v___x_482_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40));
v___x_483_ = l_Lean_Expr_const___override(v___x_482_, v___x_481_);
return v___x_483_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50(void){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_484_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__43);
v___x_485_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v___x_484_);
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_487_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21);
v___x_488_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v___x_487_);
return v___x_489_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53(void){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_492_ = lean_box(0);
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__52));
v___x_494_ = l_Lean_Expr_const___override(v___x_493_, v___x_492_);
return v___x_494_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54(void){
_start:
{
lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_495_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__21);
v___x_496_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49);
v___x_497_ = l_Lean_Expr_app___override(v___x_496_, v___x_495_);
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57(void){
_start:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_501_ = lean_box(0);
v___x_502_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__56));
v___x_503_ = l_Lean_Expr_const___override(v___x_502_, v___x_501_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(lean_object* v_u_504_, lean_object* v_00_u03b1_505_, lean_object* v___s_u03b1_506_, lean_object* v_lit_507_, lean_object* v_a_508_, lean_object* v_a_509_, lean_object* v_a_510_, lean_object* v_a_511_){
_start:
{
lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; 
v___x_513_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_514_ = l_Lean_Name_mkStr1(v___x_513_);
v___x_515_ = l_Lean_Expr_isConstOf(v_00_u03b1_505_, v___x_514_);
if (v___x_515_ == 0)
{
lean_object* v___x_516_; uint8_t v___x_517_; 
lean_dec(v___x_514_);
v___x_516_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__20));
v___x_517_ = l_Lean_Expr_isConstOf(v_00_u03b1_505_, v___x_516_);
if (v___x_517_ == 0)
{
lean_object* v___x_518_; uint8_t v___x_519_; 
v___x_518_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__1));
v___x_519_ = l_Lean_Expr_isConstOf(v_00_u03b1_505_, v___x_518_);
if (v___x_519_ == 0)
{
lean_object* v___x_520_; 
lean_inc_ref(v_lit_507_);
v___x_520_ = l_Lean_Expr_rawNatLit_x3f(v_lit_507_);
if (lean_obj_tag(v___x_520_) == 1)
{
lean_object* v_val_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_623_; 
v_val_521_ = lean_ctor_get(v___x_520_, 0);
v_isSharedCheck_623_ = !lean_is_exclusive(v___x_520_);
if (v_isSharedCheck_623_ == 0)
{
v___x_523_ = v___x_520_;
v_isShared_524_ = v_isSharedCheck_623_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_val_521_);
lean_dec(v___x_520_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_623_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v_zero_525_; uint8_t v_isZero_526_; 
v_zero_525_ = lean_unsigned_to_nat(0u);
v_isZero_526_ = lean_nat_dec_eq(v_val_521_, v_zero_525_);
if (v_isZero_526_ == 1)
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_562_; 
lean_dec(v_val_521_);
lean_dec_ref(v_lit_507_);
v___x_527_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3));
v___x_528_ = lean_box(0);
v___x_529_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_529_, 0, v_u_504_);
lean_ctor_set(v___x_529_, 1, v___x_528_);
lean_inc_ref_n(v___x_529_, 6);
v___x_530_ = l_Lean_Expr_const___override(v___x_527_, v___x_529_);
lean_inc_ref_n(v_00_u03b1_505_, 6);
v___x_531_ = l_Lean_Expr_app___override(v___x_530_, v_00_u03b1_505_);
v___x_532_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__5);
v___x_533_ = l_Lean_Expr_app___override(v___x_531_, v___x_532_);
v___x_534_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__8));
v___x_535_ = l_Lean_Expr_const___override(v___x_534_, v___x_529_);
v___x_536_ = l_Lean_Expr_app___override(v___x_535_, v_00_u03b1_505_);
v___x_537_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__11));
v___x_538_ = l_Lean_Expr_const___override(v___x_537_, v___x_529_);
v___x_539_ = l_Lean_Expr_app___override(v___x_538_, v_00_u03b1_505_);
v___x_540_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__14));
v___x_541_ = l_Lean_Expr_const___override(v___x_540_, v___x_529_);
v___x_542_ = l_Lean_Expr_app___override(v___x_541_, v_00_u03b1_505_);
v___x_543_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__17));
v___x_544_ = l_Lean_Expr_const___override(v___x_543_, v___x_529_);
v___x_545_ = l_Lean_Expr_app___override(v___x_544_, v_00_u03b1_505_);
v___x_546_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__19));
v___x_547_ = l_Lean_Expr_const___override(v___x_546_, v___x_529_);
v___x_548_ = l_Lean_Expr_app___override(v___x_547_, v_00_u03b1_505_);
lean_inc_ref(v___s_u03b1_506_);
v___x_549_ = l_Lean_Expr_app___override(v___x_548_, v___s_u03b1_506_);
v___x_550_ = l_Lean_Expr_app___override(v___x_545_, v___x_549_);
v___x_551_ = l_Lean_Expr_app___override(v___x_542_, v___x_550_);
v___x_552_ = l_Lean_Expr_app___override(v___x_539_, v___x_551_);
v___x_553_ = l_Lean_Expr_app___override(v___x_536_, v___x_552_);
v___x_554_ = l_Lean_Expr_app___override(v___x_533_, v___x_553_);
v___x_555_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__20));
v___x_556_ = l_Lean_Name_mkStr2(v___x_513_, v___x_555_);
v___x_557_ = l_Lean_Expr_const___override(v___x_556_, v___x_529_);
v___x_558_ = l_Lean_Expr_app___override(v___x_557_, v_00_u03b1_505_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v___s_u03b1_506_);
v___x_560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_560_, 0, v___x_554_);
lean_ctor_set(v___x_560_, 1, v___x_559_);
if (v_isShared_524_ == 0)
{
lean_ctor_set_tag(v___x_523_, 0);
lean_ctor_set(v___x_523_, 0, v___x_560_);
v___x_562_ = v___x_523_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_560_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
else
{
lean_object* v_one_564_; lean_object* v_n_565_; uint8_t v_isZero_566_; 
v_one_564_ = lean_unsigned_to_nat(1u);
v_n_565_ = lean_nat_sub(v_val_521_, v_one_564_);
lean_dec(v_val_521_);
v_isZero_566_ = lean_nat_dec_eq(v_n_565_, v_zero_525_);
if (v_isZero_566_ == 1)
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_590_; 
lean_dec(v_n_565_);
lean_dec_ref(v_lit_507_);
v___x_567_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3));
v___x_568_ = lean_box(0);
v___x_569_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_569_, 0, v_u_504_);
lean_ctor_set(v___x_569_, 1, v___x_568_);
lean_inc_ref_n(v___x_569_, 3);
v___x_570_ = l_Lean_Expr_const___override(v___x_567_, v___x_569_);
lean_inc_ref_n(v_00_u03b1_505_, 3);
v___x_571_ = l_Lean_Expr_app___override(v___x_570_, v_00_u03b1_505_);
v___x_572_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22);
v___x_573_ = l_Lean_Expr_app___override(v___x_571_, v___x_572_);
v___x_574_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__25));
v___x_575_ = l_Lean_Expr_const___override(v___x_574_, v___x_569_);
v___x_576_ = l_Lean_Expr_app___override(v___x_575_, v_00_u03b1_505_);
v___x_577_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__27));
v___x_578_ = l_Lean_Expr_const___override(v___x_577_, v___x_569_);
v___x_579_ = l_Lean_Expr_app___override(v___x_578_, v_00_u03b1_505_);
lean_inc_ref(v___s_u03b1_506_);
v___x_580_ = l_Lean_Expr_app___override(v___x_579_, v___s_u03b1_506_);
v___x_581_ = l_Lean_Expr_app___override(v___x_576_, v___x_580_);
v___x_582_ = l_Lean_Expr_app___override(v___x_573_, v___x_581_);
v___x_583_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__28));
v___x_584_ = l_Lean_Name_mkStr2(v___x_513_, v___x_583_);
v___x_585_ = l_Lean_Expr_const___override(v___x_584_, v___x_569_);
v___x_586_ = l_Lean_Expr_app___override(v___x_585_, v_00_u03b1_505_);
v___x_587_ = l_Lean_Expr_app___override(v___x_586_, v___s_u03b1_506_);
v___x_588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_582_);
lean_ctor_set(v___x_588_, 1, v___x_587_);
if (v_isShared_524_ == 0)
{
lean_ctor_set_tag(v___x_523_, 0);
lean_ctor_set(v___x_523_, 0, v___x_588_);
v___x_590_ = v___x_523_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v___x_588_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
else
{
lean_object* v_n_592_; lean_object* v_k_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v_a_x27_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_621_; 
v_n_592_ = lean_nat_sub(v_n_565_, v_one_564_);
lean_dec(v_n_565_);
v_k_593_ = l_Lean_mkRawNatLit(v_n_592_);
v___x_594_ = lean_box(0);
v___x_595_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__3));
v___x_596_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__34);
v___x_597_ = l_Lean_Expr_app___override(v___x_596_, v_k_593_);
lean_inc(v_u_504_);
v___x_598_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_598_, 0, v_u_504_);
lean_ctor_set(v___x_598_, 1, v___x_594_);
lean_inc_ref_n(v___x_598_, 2);
v___x_599_ = l_Lean_Expr_const___override(v___x_595_, v___x_598_);
lean_inc_ref_n(v_00_u03b1_505_, 3);
v___x_600_ = l_Lean_Expr_app___override(v___x_599_, v_00_u03b1_505_);
lean_inc_ref(v_lit_507_);
v___x_601_ = l_Lean_Expr_app___override(v___x_600_, v_lit_507_);
v___x_602_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__36));
v___x_603_ = l_Lean_Expr_const___override(v___x_602_, v___x_598_);
v___x_604_ = l_Lean_Expr_app___override(v___x_603_, v_00_u03b1_505_);
v___x_605_ = l_Lean_Expr_app___override(v___x_604_, v_lit_507_);
v___x_606_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__38));
v___x_607_ = l_Lean_Expr_const___override(v___x_606_, v___x_598_);
v___x_608_ = l_Lean_Expr_app___override(v___x_607_, v_00_u03b1_505_);
v___x_609_ = l_Lean_Expr_app___override(v___x_608_, v___s_u03b1_506_);
v___x_610_ = l_Lean_Expr_app___override(v___x_605_, v___x_609_);
v___x_611_ = l_Lean_Expr_app___override(v___x_610_, v___x_597_);
v_a_x27_612_ = l_Lean_Expr_app___override(v___x_601_, v___x_611_);
v___x_613_ = l_Lean_Level_succ___override(v_u_504_);
v___x_614_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_614_, 0, v___x_613_);
lean_ctor_set(v___x_614_, 1, v___x_594_);
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__40));
v___x_616_ = l_Lean_Expr_const___override(v___x_615_, v___x_614_);
v___x_617_ = l_Lean_Expr_app___override(v___x_616_, v_00_u03b1_505_);
lean_inc_ref(v_a_x27_612_);
v___x_618_ = l_Lean_Expr_app___override(v___x_617_, v_a_x27_612_);
v___x_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_619_, 0, v_a_x27_612_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
if (v_isShared_524_ == 0)
{
lean_ctor_set_tag(v___x_523_, 0);
lean_ctor_set(v___x_523_, 0, v___x_619_);
v___x_621_ = v___x_523_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v___x_619_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
}
}
else
{
lean_object* v___x_624_; lean_object* v___x_625_; 
lean_dec(v___x_520_);
lean_dec_ref(v_lit_507_);
lean_dec_ref(v___s_u03b1_506_);
lean_dec_ref(v_00_u03b1_505_);
lean_dec(v_u_504_);
v___x_624_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__42);
v___x_625_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_NormNum_inferAddMonoidWithOne_spec__0___redArg(v___x_624_, v_a_508_, v_a_509_, v_a_510_, v_a_511_);
return v___x_625_;
}
}
else
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v_a_x27_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
lean_dec_ref(v___s_u03b1_506_);
lean_dec_ref(v_00_u03b1_505_);
lean_dec(v_u_504_);
v___x_626_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__45);
lean_inc_ref(v_lit_507_);
v___x_627_ = l_Lean_Expr_app___override(v___x_626_, v_lit_507_);
v___x_628_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__48);
v___x_629_ = l_Lean_Expr_app___override(v___x_628_, v_lit_507_);
v_a_x27_630_ = l_Lean_Expr_app___override(v___x_627_, v___x_629_);
v___x_631_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__50);
lean_inc_ref(v_a_x27_630_);
v___x_632_ = l_Lean_Expr_app___override(v___x_631_, v_a_x27_630_);
v___x_633_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_633_, 0, v_a_x27_630_);
lean_ctor_set(v___x_633_, 1, v___x_632_);
v___x_634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_634_, 0, v___x_633_);
return v___x_634_;
}
}
else
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v_a_x27_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; 
lean_dec_ref(v___s_u03b1_506_);
lean_dec_ref(v_00_u03b1_505_);
lean_dec(v_u_504_);
v___x_635_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__51);
lean_inc_ref(v_lit_507_);
v___x_636_ = l_Lean_Expr_app___override(v___x_635_, v_lit_507_);
v___x_637_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__53);
v___x_638_ = l_Lean_Expr_app___override(v___x_637_, v_lit_507_);
v_a_x27_639_ = l_Lean_Expr_app___override(v___x_636_, v___x_638_);
v___x_640_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__54);
lean_inc_ref(v_a_x27_639_);
v___x_641_ = l_Lean_Expr_app___override(v___x_640_, v_a_x27_639_);
v___x_642_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_642_, 0, v_a_x27_639_);
lean_ctor_set(v___x_642_, 1, v___x_641_);
v___x_643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_643_, 0, v___x_642_);
return v___x_643_;
}
}
else
{
lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v_a_x27_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
lean_dec_ref(v___s_u03b1_506_);
lean_dec_ref(v_00_u03b1_505_);
lean_dec(v_u_504_);
v___x_644_ = lean_box(0);
v___x_645_ = l_Lean_Expr_const___override(v___x_514_, v___x_644_);
v___x_646_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__44);
lean_inc_ref(v___x_645_);
v___x_647_ = l_Lean_Expr_app___override(v___x_646_, v___x_645_);
lean_inc_ref(v_lit_507_);
v___x_648_ = l_Lean_Expr_app___override(v___x_647_, v_lit_507_);
v___x_649_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__57);
v___x_650_ = l_Lean_Expr_app___override(v___x_649_, v_lit_507_);
v_a_x27_651_ = l_Lean_Expr_app___override(v___x_648_, v___x_650_);
v___x_652_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__49);
v___x_653_ = l_Lean_Expr_app___override(v___x_652_, v___x_645_);
lean_inc_ref(v_a_x27_651_);
v___x_654_ = l_Lean_Expr_app___override(v___x_653_, v_a_x27_651_);
v___x_655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_655_, 0, v_a_x27_651_);
lean_ctor_set(v___x_655_, 1, v___x_654_);
v___x_656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_656_, 0, v___x_655_);
return v___x_656_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___boxed(lean_object* v_u_657_, lean_object* v_00_u03b1_658_, lean_object* v___s_u03b1_659_, lean_object* v_lit_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_657_, v_00_u03b1_658_, v___s_u03b1_659_, v_lit_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_);
lean_dec(v_a_664_);
lean_dec_ref(v_a_663_);
lean_dec(v_a_662_);
lean_dec_ref(v_a_661_);
return v_res_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_rawCast___redArg(lean_object* v_inst_667_, lean_object* v_n_668_){
_start:
{
lean_object* v_toNatCast_669_; lean_object* v___x_670_; 
v_toNatCast_669_ = lean_ctor_get(v_inst_667_, 0);
lean_inc(v_toNatCast_669_);
lean_dec_ref(v_inst_667_);
v___x_670_ = lean_apply_1(v_toNatCast_669_, v_n_668_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_rawCast(lean_object* v_00_u03b1_671_, lean_object* v_inst_672_, lean_object* v_n_673_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_mathlib_Nat_rawCast___redArg(v_inst_672_, v_n_673_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_rawCast___redArg(lean_object* v_inst_675_, lean_object* v_n_676_){
_start:
{
lean_object* v___x_677_; lean_object* v_toIntCast_678_; lean_object* v___x_679_; 
v___x_677_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_675_);
v_toIntCast_678_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_toIntCast_678_);
lean_dec_ref(v___x_677_);
v___x_679_ = lean_apply_1(v_toIntCast_678_, v_n_676_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_rawCast(lean_object* v_00_u03b1_680_, lean_object* v_inst_681_, lean_object* v_n_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib_Int_rawCast___redArg(v_inst_681_, v_n_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_rawCast___redArg(lean_object* v_inst_684_, lean_object* v_n_685_, lean_object* v_d_686_){
_start:
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v_toDiv_689_; lean_object* v_toSemiring_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v_toNatCast_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_687_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_684_);
v___x_688_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_687_);
v_toDiv_689_ = lean_ctor_get(v___x_688_, 2);
lean_inc(v_toDiv_689_);
lean_dec_ref(v___x_688_);
v_toSemiring_690_ = lean_ctor_get(v_inst_684_, 0);
lean_inc_ref(v_toSemiring_690_);
lean_dec_ref(v_inst_684_);
v___x_691_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_toSemiring_690_);
v___x_692_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_691_);
v_toNatCast_693_ = lean_ctor_get(v___x_692_, 0);
lean_inc_n(v_toNatCast_693_, 2);
lean_dec_ref(v___x_692_);
v___x_694_ = lean_apply_1(v_toNatCast_693_, v_n_685_);
v___x_695_ = lean_apply_1(v_toNatCast_693_, v_d_686_);
v___x_696_ = lean_apply_2(v_toDiv_689_, v___x_694_, v___x_695_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_rawCast(lean_object* v_00_u03b1_697_, lean_object* v_inst_698_, lean_object* v_n_699_, lean_object* v_d_700_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_mathlib_NNRat_rawCast___redArg(v_inst_698_, v_n_699_, v_d_700_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_rawCast___redArg(lean_object* v_inst_702_, lean_object* v_n_703_, lean_object* v_d_704_){
_start:
{
lean_object* v___x_705_; lean_object* v_toDiv_706_; lean_object* v_toRing_707_; lean_object* v___x_708_; lean_object* v_toAddMonoidWithOne_709_; lean_object* v_toIntCast_710_; lean_object* v_toNatCast_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; 
v___x_705_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_702_);
v_toDiv_706_ = lean_ctor_get(v___x_705_, 2);
lean_inc(v_toDiv_706_);
lean_dec_ref(v___x_705_);
v_toRing_707_ = lean_ctor_get(v_inst_702_, 0);
lean_inc_ref(v_toRing_707_);
lean_dec_ref(v_inst_702_);
v___x_708_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_707_);
v_toAddMonoidWithOne_709_ = lean_ctor_get(v___x_708_, 1);
lean_inc_ref(v_toAddMonoidWithOne_709_);
v_toIntCast_710_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_toIntCast_710_);
lean_dec_ref(v___x_708_);
v_toNatCast_711_ = lean_ctor_get(v_toAddMonoidWithOne_709_, 0);
lean_inc(v_toNatCast_711_);
lean_dec_ref(v_toAddMonoidWithOne_709_);
v___x_712_ = lean_apply_1(v_toIntCast_710_, v_n_703_);
v___x_713_ = lean_apply_1(v_toNatCast_711_, v_d_704_);
v___x_714_ = lean_apply_2(v_toDiv_706_, v___x_712_, v___x_713_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_rawCast(lean_object* v_00_u03b1_715_, lean_object* v_inst_716_, lean_object* v_n_717_, lean_object* v_d_718_){
_start:
{
lean_object* v___x_719_; 
v___x_719_ = lp_mathlib_Rat_rawCast___redArg(v_inst_716_, v_n_717_, v_d_718_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorIdx(lean_object* v_x_720_){
_start:
{
switch(lean_obj_tag(v_x_720_))
{
case 0:
{
lean_object* v___x_721_; 
v___x_721_ = lean_unsigned_to_nat(0u);
return v___x_721_;
}
case 1:
{
lean_object* v___x_722_; 
v___x_722_ = lean_unsigned_to_nat(1u);
return v___x_722_;
}
case 2:
{
lean_object* v___x_723_; 
v___x_723_ = lean_unsigned_to_nat(2u);
return v___x_723_;
}
case 3:
{
lean_object* v___x_724_; 
v___x_724_ = lean_unsigned_to_nat(3u);
return v___x_724_;
}
default: 
{
lean_object* v___x_725_; 
v___x_725_ = lean_unsigned_to_nat(4u);
return v___x_725_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorIdx___boxed(lean_object* v_x_726_){
_start:
{
lean_object* v_res_727_; 
v_res_727_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorIdx(v_x_726_);
lean_dec_ref(v_x_726_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(lean_object* v_t_728_, lean_object* v_k_729_){
_start:
{
switch(lean_obj_tag(v_t_728_))
{
case 0:
{
uint8_t v_val_730_; lean_object* v_proof_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v_val_730_ = lean_ctor_get_uint8(v_t_728_, sizeof(void*)*1);
v_proof_731_ = lean_ctor_get(v_t_728_, 0);
lean_inc_ref(v_proof_731_);
lean_dec_ref_known(v_t_728_, 1);
v___x_732_ = lean_box(v_val_730_);
v___x_733_ = lean_apply_2(v_k_729_, v___x_732_, v_proof_731_);
return v___x_733_;
}
case 3:
{
lean_object* v_inst_734_; lean_object* v_q_735_; lean_object* v_n_736_; lean_object* v_d_737_; lean_object* v_proof_738_; lean_object* v___x_739_; 
v_inst_734_ = lean_ctor_get(v_t_728_, 0);
lean_inc_ref(v_inst_734_);
v_q_735_ = lean_ctor_get(v_t_728_, 1);
lean_inc_ref(v_q_735_);
v_n_736_ = lean_ctor_get(v_t_728_, 2);
lean_inc_ref(v_n_736_);
v_d_737_ = lean_ctor_get(v_t_728_, 3);
lean_inc_ref(v_d_737_);
v_proof_738_ = lean_ctor_get(v_t_728_, 4);
lean_inc_ref(v_proof_738_);
lean_dec_ref_known(v_t_728_, 5);
v___x_739_ = lean_apply_5(v_k_729_, v_inst_734_, v_q_735_, v_n_736_, v_d_737_, v_proof_738_);
return v___x_739_;
}
case 4:
{
lean_object* v_inst_740_; lean_object* v_q_741_; lean_object* v_n_742_; lean_object* v_d_743_; lean_object* v_proof_744_; lean_object* v___x_745_; 
v_inst_740_ = lean_ctor_get(v_t_728_, 0);
lean_inc_ref(v_inst_740_);
v_q_741_ = lean_ctor_get(v_t_728_, 1);
lean_inc_ref(v_q_741_);
v_n_742_ = lean_ctor_get(v_t_728_, 2);
lean_inc_ref(v_n_742_);
v_d_743_ = lean_ctor_get(v_t_728_, 3);
lean_inc_ref(v_d_743_);
v_proof_744_ = lean_ctor_get(v_t_728_, 4);
lean_inc_ref(v_proof_744_);
lean_dec_ref_known(v_t_728_, 5);
v___x_745_ = lean_apply_5(v_k_729_, v_inst_740_, v_q_741_, v_n_742_, v_d_743_, v_proof_744_);
return v___x_745_;
}
default: 
{
lean_object* v_inst_746_; lean_object* v_lit_747_; lean_object* v_proof_748_; lean_object* v___x_749_; 
v_inst_746_ = lean_ctor_get(v_t_728_, 0);
lean_inc_ref(v_inst_746_);
v_lit_747_ = lean_ctor_get(v_t_728_, 1);
lean_inc_ref(v_lit_747_);
v_proof_748_ = lean_ctor_get(v_t_728_, 2);
lean_inc_ref(v_proof_748_);
lean_dec_ref(v_t_728_);
v___x_749_ = lean_apply_3(v_k_729_, v_inst_746_, v_lit_747_, v_proof_748_);
return v___x_749_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim(lean_object* v_motive_750_, lean_object* v_ctorIdx_751_, lean_object* v_t_752_, lean_object* v_h_753_, lean_object* v_k_754_){
_start:
{
lean_object* v___x_755_; 
v___x_755_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_752_, v_k_754_);
return v___x_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___boxed(lean_object* v_motive_756_, lean_object* v_ctorIdx_757_, lean_object* v_t_758_, lean_object* v_h_759_, lean_object* v_k_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim(v_motive_756_, v_ctorIdx_757_, v_t_758_, v_h_759_, v_k_760_);
lean_dec(v_ctorIdx_757_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isBool_elim___redArg(lean_object* v_t_762_, lean_object* v_isBool_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_762_, v_isBool_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isBool_elim(lean_object* v_motive_765_, lean_object* v_t_766_, lean_object* v_h_767_, lean_object* v_isBool_768_){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_766_, v_isBool_768_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNat_elim___redArg(lean_object* v_t_770_, lean_object* v_isNat_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_770_, v_isNat_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNat_elim(lean_object* v_motive_773_, lean_object* v_t_774_, lean_object* v_h_775_, lean_object* v_isNat_776_){
_start:
{
lean_object* v___x_777_; 
v___x_777_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_774_, v_isNat_776_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNat_elim___redArg(lean_object* v_t_778_, lean_object* v_isNegNat_779_){
_start:
{
lean_object* v___x_780_; 
v___x_780_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_778_, v_isNegNat_779_);
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNat_elim(lean_object* v_motive_781_, lean_object* v_t_782_, lean_object* v_h_783_, lean_object* v_isNegNat_784_){
_start:
{
lean_object* v___x_785_; 
v___x_785_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_782_, v_isNegNat_784_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNNRat_elim___redArg(lean_object* v_t_786_, lean_object* v_isNNRat_787_){
_start:
{
lean_object* v___x_788_; 
v___x_788_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_786_, v_isNNRat_787_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNNRat_elim(lean_object* v_motive_789_, lean_object* v_t_790_, lean_object* v_h_791_, lean_object* v_isNNRat_792_){
_start:
{
lean_object* v___x_793_; 
v___x_793_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_790_, v_isNNRat_792_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNNRat_elim___redArg(lean_object* v_t_794_, lean_object* v_isNegNNRat_795_){
_start:
{
lean_object* v___x_796_; 
v___x_796_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_794_, v_isNegNNRat_795_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_x27_isNegNNRat_elim(lean_object* v_motive_797_, lean_object* v_t_798_, lean_object* v_h_799_, lean_object* v_isNegNNRat_800_){
_start:
{
lean_object* v___x_801_; 
v___x_801_ = lp_mathlib_Mathlib_Meta_NormNum_Result_x27_ctorElim___redArg(v_t_798_, v_isNegNNRat_800_);
return v___x_801_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0(void){
_start:
{
lean_object* v___x_802_; uint8_t v___x_803_; lean_object* v___x_804_; 
v___x_802_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__10);
v___x_803_ = 0;
v___x_804_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_804_, 0, v___x_802_);
lean_ctor_set_uint8(v___x_804_, sizeof(void*)*1, v___x_803_);
return v___x_804_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default(void){
_start:
{
lean_object* v___x_805_; 
v___x_805_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default___closed__0);
return v___x_805_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27(void){
_start:
{
lean_object* v___x_806_; 
v___x_806_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default;
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___aux__1(lean_object* v_u_807_, lean_object* v_00_u03b1_808_, lean_object* v_x_809_){
_start:
{
lean_object* v___x_810_; 
v___x_810_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default;
return v___x_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___aux__1___boxed(lean_object* v_u_811_, lean_object* v_00_u03b1_812_, lean_object* v_x_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___aux__1(v_u_811_, v_00_u03b1_812_, v_x_813_);
lean_dec_ref(v_x_813_);
lean_dec_ref(v_00_u03b1_812_);
lean_dec(v_u_811_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult(lean_object* v_u_815_, lean_object* v_00_u03b1_816_, lean_object* v_x_817_){
_start:
{
lean_object* v___x_818_; 
v___x_818_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default;
return v___x_818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult___boxed(lean_object* v_u_819_, lean_object* v_00_u03b1_820_, lean_object* v_x_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult(v_u_819_, v_00_u03b1_820_, v_x_821_);
lean_dec_ref(v_x_821_);
lean_dec_ref(v_00_u03b1_820_);
lean_dec(v_u_819_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue___redArg(lean_object* v_proof_823_){
_start:
{
uint8_t v___x_824_; lean_object* v___x_825_; 
v___x_824_ = 1;
v___x_825_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_825_, 0, v_proof_823_);
lean_ctor_set_uint8(v___x_825_, sizeof(void*)*1, v___x_824_);
return v___x_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue(lean_object* v_x_826_, lean_object* v_proof_827_){
_start:
{
uint8_t v___x_828_; lean_object* v___x_829_; 
v___x_828_ = 1;
v___x_829_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_829_, 0, v_proof_827_);
lean_ctor_set_uint8(v___x_829_, sizeof(void*)*1, v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue___boxed(lean_object* v_x_830_, lean_object* v_proof_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isTrue(v_x_830_, v_proof_831_);
lean_dec_ref(v_x_830_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse___redArg(lean_object* v_proof_833_){
_start:
{
uint8_t v___x_834_; lean_object* v___x_835_; 
v___x_834_ = 0;
v___x_835_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_835_, 0, v_proof_833_);
lean_ctor_set_uint8(v___x_835_, sizeof(void*)*1, v___x_834_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse(lean_object* v_x_836_, lean_object* v_proof_837_){
_start:
{
uint8_t v___x_838_; lean_object* v___x_839_; 
v___x_838_ = 0;
v___x_839_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_839_, 0, v_proof_837_);
lean_ctor_set_uint8(v___x_839_, sizeof(void*)*1, v___x_838_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse___boxed(lean_object* v_x_840_, lean_object* v_proof_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isFalse(v_x_840_, v_proof_841_);
lean_dec_ref(v_x_840_);
return v_res_842_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12(void){
_start:
{
lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_869_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__10));
v___x_870_ = l_Lean_mkAtom(v___x_869_);
return v___x_870_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13(void){
_start:
{
lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v___x_871_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__12);
v___x_872_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_873_ = lean_array_push(v___x_872_, v___x_871_);
return v___x_873_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14(void){
_start:
{
lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; 
v___x_874_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__13);
v___x_875_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__11));
v___x_876_ = lean_box(2);
v___x_877_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_877_, 0, v___x_876_);
lean_ctor_set(v___x_877_, 1, v___x_875_);
lean_ctor_set(v___x_877_, 2, v___x_874_);
return v___x_877_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15(void){
_start:
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_878_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__14);
v___x_879_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_880_ = lean_array_push(v___x_879_, v___x_878_);
return v___x_880_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16(void){
_start:
{
lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_881_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__15);
v___x_882_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__9));
v___x_883_ = lean_box(2);
v___x_884_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_884_, 0, v___x_883_);
lean_ctor_set(v___x_884_, 1, v___x_882_);
lean_ctor_set(v___x_884_, 2, v___x_881_);
return v___x_884_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17(void){
_start:
{
lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; 
v___x_885_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__16);
v___x_886_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_887_ = lean_array_push(v___x_886_, v___x_885_);
return v___x_887_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18(void){
_start:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; 
v___x_888_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__17);
v___x_889_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7));
v___x_890_ = lean_box(2);
v___x_891_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_891_, 0, v___x_890_);
lean_ctor_set(v___x_891_, 1, v___x_889_);
lean_ctor_set(v___x_891_, 2, v___x_888_);
return v___x_891_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19(void){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_892_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__18);
v___x_893_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_894_ = lean_array_push(v___x_893_, v___x_892_);
return v___x_894_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20(void){
_start:
{
lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
v___x_895_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__19);
v___x_896_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4));
v___x_897_ = lean_box(2);
v___x_898_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_898_, 0, v___x_897_);
lean_ctor_set(v___x_898_, 1, v___x_896_);
lean_ctor_set(v___x_898_, 2, v___x_895_);
return v___x_898_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1(void){
_start:
{
lean_object* v___x_899_; 
v___x_899_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___redArg(lean_object* v_inst_900_, lean_object* v_lit_901_, lean_object* v_proof_902_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_903_, 0, v_inst_900_);
lean_ctor_set(v___x_903_, 1, v_lit_901_);
lean_ctor_set(v___x_903_, 2, v_proof_902_);
return v___x_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat(lean_object* v_u_904_, lean_object* v_00_u03b1_905_, lean_object* v_x_906_, lean_object* v_inst_907_, lean_object* v_lit_908_, lean_object* v_proof_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_910_, 0, v_inst_907_);
lean_ctor_set(v___x_910_, 1, v_lit_908_);
lean_ctor_set(v___x_910_, 2, v_proof_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___boxed(lean_object* v_u_911_, lean_object* v_00_u03b1_912_, lean_object* v_x_913_, lean_object* v_inst_914_, lean_object* v_lit_915_, lean_object* v_proof_916_){
_start:
{
lean_object* v_res_917_; 
v_res_917_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isNat(v_u_911_, v_00_u03b1_912_, v_x_913_, v_inst_914_, v_lit_915_, v_proof_916_);
lean_dec_ref(v_x_913_);
lean_dec_ref(v_00_u03b1_912_);
lean_dec(v_u_911_);
return v_res_917_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___auto__1(void){
_start:
{
lean_object* v___x_918_; 
v___x_918_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___redArg(lean_object* v_inst_919_, lean_object* v_lit_920_, lean_object* v_proof_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_922_, 0, v_inst_919_);
lean_ctor_set(v___x_922_, 1, v_lit_920_);
lean_ctor_set(v___x_922_, 2, v_proof_921_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat(lean_object* v_u_923_, lean_object* v_00_u03b1_924_, lean_object* v_x_925_, lean_object* v_inst_926_, lean_object* v_lit_927_, lean_object* v_proof_928_){
_start:
{
lean_object* v___x_929_; 
v___x_929_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_929_, 0, v_inst_926_);
lean_ctor_set(v___x_929_, 1, v_lit_927_);
lean_ctor_set(v___x_929_, 2, v_proof_928_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___boxed(lean_object* v_u_930_, lean_object* v_00_u03b1_931_, lean_object* v_x_932_, lean_object* v_inst_933_, lean_object* v_lit_934_, lean_object* v_proof_935_){
_start:
{
lean_object* v_res_936_; 
v_res_936_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat(v_u_930_, v_00_u03b1_931_, v_x_932_, v_inst_933_, v_lit_934_, v_proof_935_);
lean_dec_ref(v_x_932_);
lean_dec_ref(v_00_u03b1_931_);
lean_dec(v_u_930_);
return v_res_936_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___auto__1(void){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___redArg(lean_object* v_inst_938_, lean_object* v_q_939_, lean_object* v_n_940_, lean_object* v_d_941_, lean_object* v_proof_942_){
_start:
{
lean_object* v___x_943_; 
v___x_943_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_943_, 0, v_inst_938_);
lean_ctor_set(v___x_943_, 1, v_q_939_);
lean_ctor_set(v___x_943_, 2, v_n_940_);
lean_ctor_set(v___x_943_, 3, v_d_941_);
lean_ctor_set(v___x_943_, 4, v_proof_942_);
return v___x_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat(lean_object* v_u_944_, lean_object* v_00_u03b1_945_, lean_object* v_x_946_, lean_object* v_inst_947_, lean_object* v_q_948_, lean_object* v_n_949_, lean_object* v_d_950_, lean_object* v_proof_951_){
_start:
{
lean_object* v___x_952_; 
v___x_952_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_952_, 0, v_inst_947_);
lean_ctor_set(v___x_952_, 1, v_q_948_);
lean_ctor_set(v___x_952_, 2, v_n_949_);
lean_ctor_set(v___x_952_, 3, v_d_950_);
lean_ctor_set(v___x_952_, 4, v_proof_951_);
return v___x_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___boxed(lean_object* v_u_953_, lean_object* v_00_u03b1_954_, lean_object* v_x_955_, lean_object* v_inst_956_, lean_object* v_q_957_, lean_object* v_n_958_, lean_object* v_d_959_, lean_object* v_proof_960_){
_start:
{
lean_object* v_res_961_; 
v_res_961_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat(v_u_953_, v_00_u03b1_954_, v_x_955_, v_inst_956_, v_q_957_, v_n_958_, v_d_959_, v_proof_960_);
lean_dec_ref(v_x_955_);
lean_dec_ref(v_00_u03b1_954_);
lean_dec(v_u_953_);
return v_res_961_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___auto__1(void){
_start:
{
lean_object* v___x_962_; 
v___x_962_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___redArg(lean_object* v_inst_963_, lean_object* v_q_964_, lean_object* v_n_965_, lean_object* v_d_966_, lean_object* v_proof_967_){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v___x_968_, 0, v_inst_963_);
lean_ctor_set(v___x_968_, 1, v_q_964_);
lean_ctor_set(v___x_968_, 2, v_n_965_);
lean_ctor_set(v___x_968_, 3, v_d_966_);
lean_ctor_set(v___x_968_, 4, v_proof_967_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat(lean_object* v_u_969_, lean_object* v_00_u03b1_970_, lean_object* v_x_971_, lean_object* v_inst_972_, lean_object* v_q_973_, lean_object* v_n_974_, lean_object* v_d_975_, lean_object* v_proof_976_){
_start:
{
lean_object* v___x_977_; 
v___x_977_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v___x_977_, 0, v_inst_972_);
lean_ctor_set(v___x_977_, 1, v_q_973_);
lean_ctor_set(v___x_977_, 2, v_n_974_);
lean_ctor_set(v___x_977_, 3, v_d_975_);
lean_ctor_set(v___x_977_, 4, v_proof_976_);
return v___x_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___boxed(lean_object* v_u_978_, lean_object* v_00_u03b1_979_, lean_object* v_x_980_, lean_object* v_inst_981_, lean_object* v_q_982_, lean_object* v_n_983_, lean_object* v_d_984_, lean_object* v_proof_985_){
_start:
{
lean_object* v_res_986_; 
v_res_986_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat(v_u_978_, v_00_u03b1_979_, v_x_980_, v_inst_981_, v_q_982_, v_n_983_, v_d_984_, v_proof_985_);
lean_dec_ref(v_x_980_);
lean_dec_ref(v_00_u03b1_979_);
lean_dec(v_u_978_);
return v_res_986_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___auto__1(void){
_start:
{
lean_object* v___x_987_; 
v___x_987_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(lean_object* v_u_1002_, lean_object* v_00_u03b1_1003_, lean_object* v_x_1004_, lean_object* v_inst_1005_, lean_object* v_z_1006_, lean_object* v_n_1007_, lean_object* v_proof_1008_){
_start:
{
lean_object* v_lit_1009_; lean_object* v___x_1010_; uint8_t v___x_1011_; 
v_lit_1009_ = l_Lean_Expr_appArg_x21(v_z_1006_);
v___x_1010_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0);
v___x_1011_ = lean_int_dec_le(v___x_1010_, v_n_1007_);
if (v___x_1011_ == 0)
{
lean_object* v___x_1012_; 
lean_dec_ref(v_x_1004_);
lean_dec_ref(v_00_u03b1_1003_);
lean_dec(v_u_1002_);
v___x_1012_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1012_, 0, v_inst_1005_);
lean_ctor_set(v___x_1012_, 1, v_lit_1009_);
lean_ctor_set(v___x_1012_, 2, v_proof_1008_);
return v___x_1012_;
}
else
{
lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1013_ = lean_box(0);
v___x_1014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1014_, 0, v_u_1002_);
lean_ctor_set(v___x_1014_, 1, v___x_1013_);
v___x_1015_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__1));
lean_inc_ref(v___x_1014_);
v___x_1016_ = l_Lean_Expr_const___override(v___x_1015_, v___x_1014_);
lean_inc_ref(v_00_u03b1_1003_);
v___x_1017_ = l_Lean_Expr_app___override(v___x_1016_, v_00_u03b1_1003_);
lean_inc_ref(v_inst_1005_);
v___x_1018_ = l_Lean_Expr_app___override(v___x_1017_, v_inst_1005_);
v___x_1019_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___closed__4));
v___x_1020_ = l_Lean_Expr_const___override(v___x_1019_, v___x_1014_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_00_u03b1_1003_);
v___x_1022_ = l_Lean_Expr_app___override(v___x_1021_, v_inst_1005_);
v___x_1023_ = l_Lean_Expr_app___override(v___x_1022_, v_x_1004_);
lean_inc_ref(v_lit_1009_);
v___x_1024_ = l_Lean_Expr_app___override(v___x_1023_, v_lit_1009_);
v___x_1025_ = l_Lean_Expr_app___override(v___x_1024_, v_proof_1008_);
v___x_1026_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1026_, 0, v___x_1018_);
lean_ctor_set(v___x_1026_, 1, v_lit_1009_);
lean_ctor_set(v___x_1026_, 2, v___x_1025_);
return v___x_1026_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___boxed(lean_object* v_u_1027_, lean_object* v_00_u03b1_1028_, lean_object* v_x_1029_, lean_object* v_inst_1030_, lean_object* v_z_1031_, lean_object* v_n_1032_, lean_object* v_proof_1033_){
_start:
{
lean_object* v_res_1034_; 
v_res_1034_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_1027_, v_00_u03b1_1028_, v_x_1029_, v_inst_1030_, v_z_1031_, v_n_1032_, v_proof_1033_);
lean_dec(v_n_1032_);
lean_dec_ref(v_z_1031_);
return v_res_1034_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___auto__1(void){
_start:
{
lean_object* v___x_1035_; 
v___x_1035_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27(lean_object* v_u_1054_, lean_object* v_00_u03b1_1055_, lean_object* v_x_1056_, lean_object* v_inst_1057_, lean_object* v_q_1058_, lean_object* v_n_1059_, lean_object* v_d_1060_, lean_object* v_proof_1061_){
_start:
{
lean_object* v_den_1062_; lean_object* v___x_1063_; uint8_t v___x_1064_; 
v_den_1062_ = lean_ctor_get(v_q_1058_, 1);
v___x_1063_ = lean_unsigned_to_nat(1u);
v___x_1064_ = lean_nat_dec_eq(v_den_1062_, v___x_1063_);
if (v___x_1064_ == 0)
{
lean_object* v___x_1065_; 
lean_dec_ref(v_x_1056_);
lean_dec_ref(v_00_u03b1_1055_);
lean_dec(v_u_1054_);
v___x_1065_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_1065_, 0, v_inst_1057_);
lean_ctor_set(v___x_1065_, 1, v_q_1058_);
lean_ctor_set(v___x_1065_, 2, v_n_1059_);
lean_ctor_set(v___x_1065_, 3, v_d_1060_);
lean_ctor_set(v___x_1065_, 4, v_proof_1061_);
return v___x_1065_;
}
else
{
lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1089_; 
lean_dec_ref(v_d_1060_);
v_isSharedCheck_1089_ = !lean_is_exclusive(v_q_1058_);
if (v_isSharedCheck_1089_ == 0)
{
lean_object* v_unused_1090_; lean_object* v_unused_1091_; 
v_unused_1090_ = lean_ctor_get(v_q_1058_, 1);
lean_dec(v_unused_1090_);
v_unused_1091_ = lean_ctor_get(v_q_1058_, 0);
lean_dec(v_unused_1091_);
v___x_1067_ = v_q_1058_;
v_isShared_1068_ = v_isSharedCheck_1089_;
goto v_resetjp_1066_;
}
else
{
lean_dec(v_q_1058_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1089_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v___x_1069_; lean_object* v___x_1071_; 
v___x_1069_ = lean_box(0);
if (v_isShared_1068_ == 0)
{
lean_ctor_set_tag(v___x_1067_, 1);
lean_ctor_set(v___x_1067_, 1, v___x_1069_);
lean_ctor_set(v___x_1067_, 0, v_u_1054_);
v___x_1071_ = v___x_1067_;
goto v_reusejp_1070_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v_u_1054_);
lean_ctor_set(v_reuseFailAlloc_1088_, 1, v___x_1069_);
v___x_1071_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1070_;
}
v_reusejp_1070_:
{
lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1072_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__1));
lean_inc_ref_n(v___x_1071_, 2);
v___x_1073_ = l_Lean_Expr_const___override(v___x_1072_, v___x_1071_);
lean_inc_ref_n(v_00_u03b1_1055_, 2);
v___x_1074_ = l_Lean_Expr_app___override(v___x_1073_, v_00_u03b1_1055_);
v___x_1075_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4));
v___x_1076_ = l_Lean_Expr_const___override(v___x_1075_, v___x_1071_);
v___x_1077_ = l_Lean_Expr_app___override(v___x_1076_, v_00_u03b1_1055_);
v___x_1078_ = l_Lean_Expr_app___override(v___x_1077_, v_inst_1057_);
lean_inc_ref(v___x_1078_);
v___x_1079_ = l_Lean_Expr_app___override(v___x_1074_, v___x_1078_);
v___x_1080_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__6));
v___x_1081_ = l_Lean_Expr_const___override(v___x_1080_, v___x_1071_);
v___x_1082_ = l_Lean_Expr_app___override(v___x_1081_, v_00_u03b1_1055_);
v___x_1083_ = l_Lean_Expr_app___override(v___x_1082_, v___x_1078_);
v___x_1084_ = l_Lean_Expr_app___override(v___x_1083_, v_x_1056_);
lean_inc_ref(v_n_1059_);
v___x_1085_ = l_Lean_Expr_app___override(v___x_1084_, v_n_1059_);
v___x_1086_ = l_Lean_Expr_app___override(v___x_1085_, v_proof_1061_);
v___x_1087_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1087_, 0, v___x_1079_);
lean_ctor_set(v___x_1087_, 1, v_n_1059_);
lean_ctor_set(v___x_1087_, 2, v___x_1086_);
return v___x_1087_;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___auto__1(void){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(lean_object* v_a_1093_){
_start:
{
lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1094_ = lean_nat_to_int(v_a_1093_);
v___x_1095_ = l_Rat_ofInt(v___x_1094_);
return v___x_1095_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0(void){
_start:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; 
v___x_1096_ = lean_unsigned_to_nat(0u);
v___x_1097_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1096_);
return v___x_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_isRat(lean_object* v_u_1122_, lean_object* v_00_u03b1_1123_, lean_object* v_x_1124_, lean_object* v_inst_1125_, lean_object* v_q_1126_, lean_object* v_n_1127_, lean_object* v_d_1128_, lean_object* v_proof_1129_){
_start:
{
lean_object* v_num_1130_; lean_object* v_den_1131_; lean_object* v___x_1132_; uint8_t v___x_1133_; 
v_num_1130_ = lean_ctor_get(v_q_1126_, 0);
v_den_1131_ = lean_ctor_get(v_q_1126_, 1);
v___x_1132_ = lean_unsigned_to_nat(1u);
v___x_1133_ = lean_nat_dec_eq(v_den_1131_, v___x_1132_);
if (v___x_1133_ == 0)
{
lean_object* v_lit_1134_; lean_object* v___x_1135_; uint8_t v___x_1136_; 
v_lit_1134_ = l_Lean_Expr_appArg_x21(v_n_1127_);
lean_dec_ref(v_n_1127_);
v___x_1135_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0);
lean_inc_ref(v_q_1126_);
v___x_1136_ = l_Rat_instDecidableLe(v___x_1135_, v_q_1126_);
if (v___x_1136_ == 0)
{
lean_object* v___x_1137_; 
lean_dec_ref(v_x_1124_);
lean_dec_ref(v_00_u03b1_1123_);
lean_dec(v_u_1122_);
v___x_1137_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v___x_1137_, 0, v_inst_1125_);
lean_ctor_set(v___x_1137_, 1, v_q_1126_);
lean_ctor_set(v___x_1137_, 2, v_lit_1134_);
lean_ctor_set(v___x_1137_, 3, v_d_1128_);
lean_ctor_set(v___x_1137_, 4, v_proof_1129_);
return v___x_1137_;
}
else
{
lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; 
v___x_1138_ = lean_box(0);
v___x_1139_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1139_, 0, v_u_1122_);
lean_ctor_set(v___x_1139_, 1, v___x_1138_);
v___x_1140_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__3));
lean_inc_ref_n(v___x_1139_, 2);
v___x_1141_ = l_Lean_Expr_const___override(v___x_1140_, v___x_1139_);
lean_inc_ref_n(v_00_u03b1_1123_, 2);
v___x_1142_ = l_Lean_Expr_app___override(v___x_1141_, v_00_u03b1_1123_);
lean_inc_ref(v_inst_1125_);
v___x_1143_ = l_Lean_Expr_app___override(v___x_1142_, v_inst_1125_);
v___x_1144_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__6));
v___x_1145_ = l_Lean_Expr_const___override(v___x_1144_, v___x_1139_);
v___x_1146_ = l_Lean_Expr_app___override(v___x_1145_, v_00_u03b1_1123_);
v___x_1147_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
v___x_1148_ = l_Lean_Expr_const___override(v___x_1147_, v___x_1139_);
v___x_1149_ = l_Lean_Expr_app___override(v___x_1148_, v_00_u03b1_1123_);
v___x_1150_ = l_Lean_Expr_app___override(v___x_1149_, v_inst_1125_);
v___x_1151_ = l_Lean_Expr_app___override(v___x_1146_, v___x_1150_);
v___x_1152_ = l_Lean_Expr_app___override(v___x_1151_, v_x_1124_);
lean_inc_ref(v_lit_1134_);
v___x_1153_ = l_Lean_Expr_app___override(v___x_1152_, v_lit_1134_);
lean_inc_ref(v_d_1128_);
v___x_1154_ = l_Lean_Expr_app___override(v___x_1153_, v_d_1128_);
v___x_1155_ = l_Lean_Expr_app___override(v___x_1154_, v_proof_1129_);
v___x_1156_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_1156_, 0, v___x_1143_);
lean_ctor_set(v___x_1156_, 1, v_q_1126_);
lean_ctor_set(v___x_1156_, 2, v_lit_1134_);
lean_ctor_set(v___x_1156_, 3, v_d_1128_);
lean_ctor_set(v___x_1156_, 4, v___x_1155_);
return v___x_1156_;
}
}
else
{
lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1176_; 
lean_inc(v_num_1130_);
lean_dec_ref(v_d_1128_);
v_isSharedCheck_1176_ = !lean_is_exclusive(v_q_1126_);
if (v_isSharedCheck_1176_ == 0)
{
lean_object* v_unused_1177_; lean_object* v_unused_1178_; 
v_unused_1177_ = lean_ctor_get(v_q_1126_, 1);
lean_dec(v_unused_1177_);
v_unused_1178_ = lean_ctor_get(v_q_1126_, 0);
lean_dec(v_unused_1178_);
v___x_1158_ = v_q_1126_;
v_isShared_1159_ = v_isSharedCheck_1176_;
goto v_resetjp_1157_;
}
else
{
lean_dec(v_q_1126_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1176_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1160_; lean_object* v___x_1162_; 
v___x_1160_ = lean_box(0);
lean_inc(v_u_1122_);
if (v_isShared_1159_ == 0)
{
lean_ctor_set_tag(v___x_1158_, 1);
lean_ctor_set(v___x_1158_, 1, v___x_1160_);
lean_ctor_set(v___x_1158_, 0, v_u_1122_);
v___x_1162_ = v___x_1158_;
goto v_reusejp_1161_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v_u_1122_);
lean_ctor_set(v_reuseFailAlloc_1175_, 1, v___x_1160_);
v___x_1162_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1161_;
}
v_reusejp_1161_:
{
lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v___x_1163_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
lean_inc_ref(v___x_1162_);
v___x_1164_ = l_Lean_Expr_const___override(v___x_1163_, v___x_1162_);
lean_inc_ref_n(v_00_u03b1_1123_, 2);
v___x_1165_ = l_Lean_Expr_app___override(v___x_1164_, v_00_u03b1_1123_);
v___x_1166_ = l_Lean_Expr_app___override(v___x_1165_, v_inst_1125_);
v___x_1167_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__10));
v___x_1168_ = l_Lean_Expr_const___override(v___x_1167_, v___x_1162_);
v___x_1169_ = l_Lean_Expr_app___override(v___x_1168_, v_00_u03b1_1123_);
lean_inc_ref(v___x_1166_);
v___x_1170_ = l_Lean_Expr_app___override(v___x_1169_, v___x_1166_);
lean_inc_ref(v_x_1124_);
v___x_1171_ = l_Lean_Expr_app___override(v___x_1170_, v_x_1124_);
lean_inc_ref(v_n_1127_);
v___x_1172_ = l_Lean_Expr_app___override(v___x_1171_, v_n_1127_);
v___x_1173_ = l_Lean_Expr_app___override(v___x_1172_, v_proof_1129_);
v___x_1174_ = lp_mathlib_Mathlib_Meta_NormNum_Result_isInt(v_u_1122_, v_00_u03b1_1123_, v_x_1124_, v___x_1166_, v_n_1127_, v_num_1130_, v___x_1173_);
lean_dec(v_num_1130_);
lean_dec_ref(v_n_1127_);
return v___x_1174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0_spec__0(lean_object* v_a_1179_){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = lean_nat_to_int(v_a_1179_);
return v___x_1180_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1182_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__0));
v___x_1183_ = l_Lean_stringToMessageData(v___x_1182_);
return v___x_1183_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1185_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__2));
v___x_1186_ = l_Lean_stringToMessageData(v___x_1185_);
return v___x_1186_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; 
v___x_1188_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__4));
v___x_1189_ = l_Lean_stringToMessageData(v___x_1188_);
return v___x_1189_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7(void){
_start:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1191_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__6));
v___x_1192_ = l_Lean_stringToMessageData(v___x_1191_);
return v___x_1192_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9(void){
_start:
{
lean_object* v___x_1194_; lean_object* v___x_1195_; 
v___x_1194_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__8));
v___x_1195_ = l_Lean_stringToMessageData(v___x_1194_);
return v___x_1195_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11(void){
_start:
{
lean_object* v___x_1197_; lean_object* v___x_1198_; 
v___x_1197_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__10));
v___x_1198_ = l_Lean_stringToMessageData(v___x_1197_);
return v___x_1198_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13(void){
_start:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; 
v___x_1200_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__12));
v___x_1201_ = l_Lean_stringToMessageData(v___x_1200_);
return v___x_1201_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16(void){
_start:
{
lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1204_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__15));
v___x_1205_ = l_Lean_stringToMessageData(v___x_1204_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0(lean_object* v_x_1206_){
_start:
{
switch(lean_obj_tag(v_x_1206_))
{
case 0:
{
uint8_t v_val_1207_; 
v_val_1207_ = lean_ctor_get_uint8(v_x_1206_, sizeof(void*)*1);
if (v_val_1207_ == 0)
{
lean_object* v_proof_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; 
v_proof_1208_ = lean_ctor_get(v_x_1206_, 0);
lean_inc_ref(v_proof_1208_);
lean_dec_ref_known(v_x_1206_, 1);
v___x_1209_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__1);
v___x_1210_ = l_Lean_MessageData_ofExpr(v_proof_1208_);
v___x_1211_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1209_);
lean_ctor_set(v___x_1211_, 1, v___x_1210_);
v___x_1212_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1213_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1213_, 0, v___x_1211_);
lean_ctor_set(v___x_1213_, 1, v___x_1212_);
return v___x_1213_;
}
else
{
lean_object* v_proof_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; 
v_proof_1214_ = lean_ctor_get(v_x_1206_, 0);
lean_inc_ref(v_proof_1214_);
lean_dec_ref_known(v_x_1206_, 1);
v___x_1215_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__5);
v___x_1216_ = l_Lean_MessageData_ofExpr(v_proof_1214_);
v___x_1217_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1217_, 0, v___x_1215_);
lean_ctor_set(v___x_1217_, 1, v___x_1216_);
v___x_1218_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1217_);
lean_ctor_set(v___x_1219_, 1, v___x_1218_);
return v___x_1219_;
}
}
case 1:
{
lean_object* v_lit_1220_; lean_object* v_proof_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; 
v_lit_1220_ = lean_ctor_get(v_x_1206_, 1);
lean_inc_ref(v_lit_1220_);
v_proof_1221_ = lean_ctor_get(v_x_1206_, 2);
lean_inc_ref(v_proof_1221_);
lean_dec_ref_known(v_x_1206_, 3);
v___x_1222_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__7);
v___x_1223_ = l_Lean_MessageData_ofExpr(v_lit_1220_);
v___x_1224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1224_, 0, v___x_1222_);
lean_ctor_set(v___x_1224_, 1, v___x_1223_);
v___x_1225_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9);
v___x_1226_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1226_, 0, v___x_1224_);
lean_ctor_set(v___x_1226_, 1, v___x_1225_);
v___x_1227_ = l_Lean_MessageData_ofExpr(v_proof_1221_);
v___x_1228_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1228_, 0, v___x_1226_);
lean_ctor_set(v___x_1228_, 1, v___x_1227_);
v___x_1229_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1230_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1230_, 0, v___x_1228_);
lean_ctor_set(v___x_1230_, 1, v___x_1229_);
return v___x_1230_;
}
case 2:
{
lean_object* v_lit_1231_; lean_object* v_proof_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; 
v_lit_1231_ = lean_ctor_get(v_x_1206_, 1);
lean_inc_ref(v_lit_1231_);
v_proof_1232_ = lean_ctor_get(v_x_1206_, 2);
lean_inc_ref(v_proof_1232_);
lean_dec_ref_known(v_x_1206_, 3);
v___x_1233_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__11);
v___x_1234_ = l_Lean_MessageData_ofExpr(v_lit_1231_);
v___x_1235_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1235_, 0, v___x_1233_);
lean_ctor_set(v___x_1235_, 1, v___x_1234_);
v___x_1236_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9);
v___x_1237_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1237_, 0, v___x_1235_);
lean_ctor_set(v___x_1237_, 1, v___x_1236_);
v___x_1238_ = l_Lean_MessageData_ofExpr(v_proof_1232_);
v___x_1239_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1239_, 0, v___x_1237_);
lean_ctor_set(v___x_1239_, 1, v___x_1238_);
v___x_1240_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1241_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1241_, 0, v___x_1239_);
lean_ctor_set(v___x_1241_, 1, v___x_1240_);
return v___x_1241_;
}
case 3:
{
lean_object* v_q_1242_; lean_object* v_proof_1243_; lean_object* v_num_1244_; lean_object* v_den_1245_; lean_object* v___x_1247_; uint8_t v_isShared_1248_; uint8_t v_isSharedCheck_1271_; 
v_q_1242_ = lean_ctor_get(v_x_1206_, 1);
lean_inc_ref(v_q_1242_);
v_proof_1243_ = lean_ctor_get(v_x_1206_, 4);
lean_inc_ref(v_proof_1243_);
lean_dec_ref_known(v_x_1206_, 5);
v_num_1244_ = lean_ctor_get(v_q_1242_, 0);
v_den_1245_ = lean_ctor_get(v_q_1242_, 1);
v_isSharedCheck_1271_ = !lean_is_exclusive(v_q_1242_);
if (v_isSharedCheck_1271_ == 0)
{
v___x_1247_ = v_q_1242_;
v_isShared_1248_ = v_isSharedCheck_1271_;
goto v_resetjp_1246_;
}
else
{
lean_inc(v_den_1245_);
lean_inc(v_num_1244_);
lean_dec(v_q_1242_);
v___x_1247_ = lean_box(0);
v_isShared_1248_ = v_isSharedCheck_1271_;
goto v_resetjp_1246_;
}
v_resetjp_1246_:
{
lean_object* v___x_1249_; lean_object* v___y_1251_; lean_object* v___x_1263_; uint8_t v___x_1264_; 
v___x_1249_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__13);
v___x_1263_ = lean_unsigned_to_nat(1u);
v___x_1264_ = lean_nat_dec_eq(v_den_1245_, v___x_1263_);
if (v___x_1264_ == 0)
{
lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; 
v___x_1265_ = l_Int_repr(v_num_1244_);
lean_dec(v_num_1244_);
v___x_1266_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__14));
v___x_1267_ = lean_string_append(v___x_1265_, v___x_1266_);
v___x_1268_ = l_Nat_reprFast(v_den_1245_);
v___x_1269_ = lean_string_append(v___x_1267_, v___x_1268_);
lean_dec_ref(v___x_1268_);
v___y_1251_ = v___x_1269_;
goto v___jp_1250_;
}
else
{
lean_object* v___x_1270_; 
lean_dec(v_den_1245_);
v___x_1270_ = l_Int_repr(v_num_1244_);
lean_dec(v_num_1244_);
v___y_1251_ = v___x_1270_;
goto v___jp_1250_;
}
v___jp_1250_:
{
lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1255_; 
v___x_1252_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1252_, 0, v___y_1251_);
v___x_1253_ = l_Lean_MessageData_ofFormat(v___x_1252_);
if (v_isShared_1248_ == 0)
{
lean_ctor_set_tag(v___x_1247_, 7);
lean_ctor_set(v___x_1247_, 1, v___x_1253_);
lean_ctor_set(v___x_1247_, 0, v___x_1249_);
v___x_1255_ = v___x_1247_;
goto v_reusejp_1254_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1262_, 1, v___x_1253_);
v___x_1255_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1254_;
}
v_reusejp_1254_:
{
lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; 
v___x_1256_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9);
v___x_1257_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1257_, 0, v___x_1255_);
lean_ctor_set(v___x_1257_, 1, v___x_1256_);
v___x_1258_ = l_Lean_MessageData_ofExpr(v_proof_1243_);
v___x_1259_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1259_, 0, v___x_1257_);
lean_ctor_set(v___x_1259_, 1, v___x_1258_);
v___x_1260_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1261_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1261_, 0, v___x_1259_);
lean_ctor_set(v___x_1261_, 1, v___x_1260_);
return v___x_1261_;
}
}
}
}
default: 
{
lean_object* v_q_1272_; lean_object* v_proof_1273_; lean_object* v_num_1274_; lean_object* v_den_1275_; lean_object* v___x_1277_; uint8_t v_isShared_1278_; uint8_t v_isSharedCheck_1301_; 
v_q_1272_ = lean_ctor_get(v_x_1206_, 1);
lean_inc_ref(v_q_1272_);
v_proof_1273_ = lean_ctor_get(v_x_1206_, 4);
lean_inc_ref(v_proof_1273_);
lean_dec_ref_known(v_x_1206_, 5);
v_num_1274_ = lean_ctor_get(v_q_1272_, 0);
v_den_1275_ = lean_ctor_get(v_q_1272_, 1);
v_isSharedCheck_1301_ = !lean_is_exclusive(v_q_1272_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1277_ = v_q_1272_;
v_isShared_1278_ = v_isSharedCheck_1301_;
goto v_resetjp_1276_;
}
else
{
lean_inc(v_den_1275_);
lean_inc(v_num_1274_);
lean_dec(v_q_1272_);
v___x_1277_ = lean_box(0);
v_isShared_1278_ = v_isSharedCheck_1301_;
goto v_resetjp_1276_;
}
v_resetjp_1276_:
{
lean_object* v___x_1279_; lean_object* v___y_1281_; lean_object* v___x_1293_; uint8_t v___x_1294_; 
v___x_1279_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__16);
v___x_1293_ = lean_unsigned_to_nat(1u);
v___x_1294_ = lean_nat_dec_eq(v_den_1275_, v___x_1293_);
if (v___x_1294_ == 0)
{
lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; 
v___x_1295_ = l_Int_repr(v_num_1274_);
lean_dec(v_num_1274_);
v___x_1296_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__14));
v___x_1297_ = lean_string_append(v___x_1295_, v___x_1296_);
v___x_1298_ = l_Nat_reprFast(v_den_1275_);
v___x_1299_ = lean_string_append(v___x_1297_, v___x_1298_);
lean_dec_ref(v___x_1298_);
v___y_1281_ = v___x_1299_;
goto v___jp_1280_;
}
else
{
lean_object* v___x_1300_; 
lean_dec(v_den_1275_);
v___x_1300_ = l_Int_repr(v_num_1274_);
lean_dec(v_num_1274_);
v___y_1281_ = v___x_1300_;
goto v___jp_1280_;
}
v___jp_1280_:
{
lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1285_; 
v___x_1282_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1282_, 0, v___y_1281_);
v___x_1283_ = l_Lean_MessageData_ofFormat(v___x_1282_);
if (v_isShared_1278_ == 0)
{
lean_ctor_set_tag(v___x_1277_, 7);
lean_ctor_set(v___x_1277_, 1, v___x_1283_);
lean_ctor_set(v___x_1277_, 0, v___x_1279_);
v___x_1285_ = v___x_1277_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1292_; 
v_reuseFailAlloc_1292_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1292_, 0, v___x_1279_);
lean_ctor_set(v_reuseFailAlloc_1292_, 1, v___x_1283_);
v___x_1285_ = v_reuseFailAlloc_1292_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; 
v___x_1286_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__9);
v___x_1287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1285_);
lean_ctor_set(v___x_1287_, 1, v___x_1286_);
v___x_1288_ = l_Lean_MessageData_ofExpr(v_proof_1273_);
v___x_1289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1289_, 0, v___x_1287_);
lean_ctor_set(v___x_1289_, 1, v___x_1288_);
v___x_1290_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___lam__0___closed__3);
v___x_1291_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1291_, 0, v___x_1289_);
lean_ctor_set(v___x_1291_, 1, v___x_1290_);
return v___x_1291_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult(lean_object* v_u_1303_, lean_object* v_00_u03b1_1304_, lean_object* v_x_1305_){
_start:
{
lean_object* v___f_1306_; 
v___f_1306_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___closed__0));
return v___f_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult___boxed(lean_object* v_u_1307_, lean_object* v_00_u03b1_1308_, lean_object* v_x_1309_){
_start:
{
lean_object* v_res_1310_; 
v_res_1310_ = lp_mathlib_Mathlib_Meta_NormNum_instToMessageDataResult(v_u_1307_, v_00_u03b1_1308_, v_x_1309_);
lean_dec_ref(v_x_1309_);
lean_dec_ref(v_00_u03b1_1308_);
lean_dec(v_u_1307_);
return v_res_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg(lean_object* v_x_1311_){
_start:
{
switch(lean_obj_tag(v_x_1311_))
{
case 0:
{
lean_object* v___x_1312_; 
v___x_1312_ = lean_box(0);
return v___x_1312_;
}
case 1:
{
lean_object* v_lit_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; 
v_lit_1313_ = lean_ctor_get(v_x_1311_, 1);
v___x_1314_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1313_);
v___x_1315_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1314_);
v___x_1316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1315_);
return v___x_1316_;
}
case 2:
{
lean_object* v_lit_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; 
v_lit_1317_ = lean_ctor_get(v_x_1311_, 1);
v___x_1318_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1317_);
v___x_1319_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1318_);
v___x_1320_ = l_Rat_neg(v___x_1319_);
v___x_1321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
return v___x_1321_;
}
default: 
{
lean_object* v_q_1322_; lean_object* v___x_1323_; 
v_q_1322_ = lean_ctor_get(v_x_1311_, 1);
lean_inc_ref(v_q_1322_);
v___x_1323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1323_, 0, v_q_1322_);
return v___x_1323_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg___boxed(lean_object* v_x_1324_){
_start:
{
lean_object* v_res_1325_; 
v_res_1325_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg(v_x_1324_);
lean_dec_ref(v_x_1324_);
return v_res_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat(lean_object* v_u_1326_, lean_object* v_00_u03b1_1327_, lean_object* v_e_1328_, lean_object* v_x_1329_){
_start:
{
lean_object* v___x_1330_; 
v___x_1330_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___redArg(v_x_1329_);
return v___x_1330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat___boxed(lean_object* v_u_1331_, lean_object* v_00_u03b1_1332_, lean_object* v_e_1333_, lean_object* v_x_1334_){
_start:
{
lean_object* v_res_1335_; 
v_res_1335_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRat(v_u_1331_, v_00_u03b1_1332_, v_e_1333_, v_x_1334_);
lean_dec_ref(v_x_1334_);
lean_dec_ref(v_e_1333_);
lean_dec_ref(v_00_u03b1_1332_);
lean_dec(v_u_1331_);
return v_res_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ(lean_object* v_u_1349_, lean_object* v_00_u03b1_1350_, lean_object* v_e_1351_, lean_object* v_x_1352_){
_start:
{
switch(lean_obj_tag(v_x_1352_))
{
case 0:
{
lean_object* v___x_1353_; 
lean_dec_ref_known(v_x_1352_, 1);
lean_dec_ref(v_e_1351_);
lean_dec_ref(v_00_u03b1_1350_);
lean_dec(v_u_1349_);
v___x_1353_ = lean_box(0);
return v___x_1353_;
}
case 1:
{
lean_object* v_lit_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; 
lean_dec_ref(v_e_1351_);
lean_dec_ref(v_00_u03b1_1350_);
lean_dec(v_u_1349_);
v_lit_1354_ = lean_ctor_get(v_x_1352_, 1);
lean_inc_ref(v_lit_1354_);
lean_dec_ref_known(v_x_1352_, 3);
v___x_1355_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1354_);
lean_dec_ref(v_lit_1354_);
v___x_1356_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1355_);
v___x_1357_ = lean_box(0);
v___x_1358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1358_, 0, v___x_1356_);
lean_ctor_set(v___x_1358_, 1, v___x_1357_);
v___x_1359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1359_, 0, v___x_1358_);
return v___x_1359_;
}
case 2:
{
lean_object* v_lit_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; 
lean_dec_ref(v_e_1351_);
lean_dec_ref(v_00_u03b1_1350_);
lean_dec(v_u_1349_);
v_lit_1360_ = lean_ctor_get(v_x_1352_, 1);
lean_inc_ref(v_lit_1360_);
lean_dec_ref_known(v_x_1352_, 3);
v___x_1361_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1360_);
lean_dec_ref(v_lit_1360_);
v___x_1362_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1361_);
v___x_1363_ = l_Rat_neg(v___x_1362_);
v___x_1364_ = lean_box(0);
v___x_1365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1365_, 0, v___x_1363_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1366_, 0, v___x_1365_);
return v___x_1366_;
}
case 3:
{
lean_object* v_inst_1367_; lean_object* v_q_1368_; lean_object* v_n_1369_; lean_object* v_d_1370_; lean_object* v_proof_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; 
v_inst_1367_ = lean_ctor_get(v_x_1352_, 0);
lean_inc_ref(v_inst_1367_);
v_q_1368_ = lean_ctor_get(v_x_1352_, 1);
lean_inc_ref(v_q_1368_);
v_n_1369_ = lean_ctor_get(v_x_1352_, 2);
lean_inc_ref(v_n_1369_);
v_d_1370_ = lean_ctor_get(v_x_1352_, 3);
lean_inc_ref(v_d_1370_);
v_proof_1371_ = lean_ctor_get(v_x_1352_, 4);
lean_inc_ref(v_proof_1371_);
lean_dec_ref_known(v_x_1352_, 5);
v___x_1372_ = lean_box(0);
v___x_1373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1373_, 0, v_u_1349_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
v___x_1374_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__1));
v___x_1375_ = l_Lean_Expr_const___override(v___x_1374_, v___x_1373_);
v___x_1376_ = l_Lean_Expr_app___override(v___x_1375_, v_00_u03b1_1350_);
v___x_1377_ = l_Lean_Expr_app___override(v___x_1376_, v_inst_1367_);
v___x_1378_ = l_Lean_Expr_app___override(v___x_1377_, v_e_1351_);
v___x_1379_ = l_Lean_Expr_app___override(v___x_1378_, v_n_1369_);
v___x_1380_ = l_Lean_Expr_app___override(v___x_1379_, v_d_1370_);
v___x_1381_ = l_Lean_Expr_app___override(v___x_1380_, v_proof_1371_);
v___x_1382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1382_, 0, v___x_1381_);
v___x_1383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1383_, 0, v_q_1368_);
lean_ctor_set(v___x_1383_, 1, v___x_1382_);
v___x_1384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1384_, 0, v___x_1383_);
return v___x_1384_;
}
default: 
{
lean_object* v_inst_1385_; lean_object* v_q_1386_; lean_object* v_n_1387_; lean_object* v_d_1388_; lean_object* v_proof_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; 
v_inst_1385_ = lean_ctor_get(v_x_1352_, 0);
lean_inc_ref(v_inst_1385_);
v_q_1386_ = lean_ctor_get(v_x_1352_, 1);
lean_inc_ref(v_q_1386_);
v_n_1387_ = lean_ctor_get(v_x_1352_, 2);
lean_inc_ref(v_n_1387_);
v_d_1388_ = lean_ctor_get(v_x_1352_, 3);
lean_inc_ref(v_d_1388_);
v_proof_1389_ = lean_ctor_get(v_x_1352_, 4);
lean_inc_ref(v_proof_1389_);
lean_dec_ref_known(v_x_1352_, 5);
v___x_1390_ = lean_box(0);
v___x_1391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1391_, 0, v_u_1349_);
lean_ctor_set(v___x_1391_, 1, v___x_1390_);
v___x_1392_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRatNZ___closed__2));
v___x_1393_ = l_Lean_Expr_const___override(v___x_1392_, v___x_1391_);
v___x_1394_ = l_Lean_Expr_app___override(v___x_1393_, v_00_u03b1_1350_);
v___x_1395_ = l_Lean_Expr_app___override(v___x_1394_, v_inst_1385_);
v___x_1396_ = l_Lean_Expr_app___override(v___x_1395_, v_e_1351_);
v___x_1397_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1398_ = l_Lean_Expr_app___override(v___x_1397_, v_n_1387_);
v___x_1399_ = l_Lean_Expr_app___override(v___x_1396_, v___x_1398_);
v___x_1400_ = l_Lean_Expr_app___override(v___x_1399_, v_d_1388_);
v___x_1401_ = l_Lean_Expr_app___override(v___x_1400_, v_proof_1389_);
v___x_1402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1402_, 0, v___x_1401_);
v___x_1403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1403_, 0, v_q_1386_);
lean_ctor_set(v___x_1403_, 1, v___x_1402_);
v___x_1404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1404_, 0, v___x_1403_);
return v___x_1404_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3(void){
_start:
{
lean_object* v___x_1412_; lean_object* v___x_1413_; 
v___x_1412_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__2));
v___x_1413_ = l_Lean_mkAtom(v___x_1412_);
return v___x_1413_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4(void){
_start:
{
lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; 
v___x_1414_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__3);
v___x_1415_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_1416_ = lean_array_push(v___x_1415_, v___x_1414_);
return v___x_1416_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5(void){
_start:
{
lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; 
v___x_1417_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__20);
v___x_1418_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__4);
v___x_1419_ = lean_array_push(v___x_1418_, v___x_1417_);
return v___x_1419_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6(void){
_start:
{
lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; 
v___x_1420_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__5);
v___x_1421_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__1));
v___x_1422_ = lean_box(2);
v___x_1423_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1422_);
lean_ctor_set(v___x_1423_, 1, v___x_1421_);
lean_ctor_set(v___x_1423_, 2, v___x_1420_);
return v___x_1423_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7(void){
_start:
{
lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; 
v___x_1424_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__6);
v___x_1425_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_1426_ = lean_array_push(v___x_1425_, v___x_1424_);
return v___x_1426_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8(void){
_start:
{
lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; 
v___x_1427_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__7);
v___x_1428_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__9));
v___x_1429_ = lean_box(2);
v___x_1430_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1429_);
lean_ctor_set(v___x_1430_, 1, v___x_1428_);
lean_ctor_set(v___x_1430_, 2, v___x_1427_);
return v___x_1430_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9(void){
_start:
{
lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; 
v___x_1431_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__8);
v___x_1432_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_1433_ = lean_array_push(v___x_1432_, v___x_1431_);
return v___x_1433_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10(void){
_start:
{
lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; 
v___x_1434_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__9);
v___x_1435_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__7));
v___x_1436_ = lean_box(2);
v___x_1437_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1437_, 0, v___x_1436_);
lean_ctor_set(v___x_1437_, 1, v___x_1435_);
lean_ctor_set(v___x_1437_, 2, v___x_1434_);
return v___x_1437_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11(void){
_start:
{
lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; 
v___x_1438_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__10);
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__5));
v___x_1440_ = lean_array_push(v___x_1439_, v___x_1438_);
return v___x_1440_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12(void){
_start:
{
lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; 
v___x_1441_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__11);
v___x_1442_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1___closed__4));
v___x_1443_ = lean_box(2);
v___x_1444_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1444_, 0, v___x_1443_);
lean_ctor_set(v___x_1444_, 1, v___x_1442_);
lean_ctor_set(v___x_1444_, 2, v___x_1441_);
return v___x_1444_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1(void){
_start:
{
lean_object* v___x_1445_; 
v___x_1445_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12);
return v___x_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toInt(lean_object* v_u_1453_, lean_object* v_00_u03b1_1454_, lean_object* v_e_1455_, lean_object* v___i_1456_, lean_object* v_x_1457_){
_start:
{
switch(lean_obj_tag(v_x_1457_))
{
case 1:
{
lean_object* v_lit_1458_; lean_object* v_proof_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
v_lit_1458_ = lean_ctor_get(v_x_1457_, 1);
lean_inc_ref_n(v_lit_1458_, 2);
v_proof_1459_ = lean_ctor_get(v_x_1457_, 2);
lean_inc_ref(v_proof_1459_);
lean_dec_ref_known(v_x_1457_, 3);
v___x_1460_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1458_);
v___x_1461_ = lean_nat_to_int(v___x_1460_);
v___x_1462_ = lean_box(0);
v___x_1463_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7);
v___x_1464_ = l_Lean_Expr_app___override(v___x_1463_, v_lit_1458_);
v___x_1465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1465_, 0, v_u_1453_);
lean_ctor_set(v___x_1465_, 1, v___x_1462_);
v___x_1466_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___closed__1));
v___x_1467_ = l_Lean_Expr_const___override(v___x_1466_, v___x_1465_);
v___x_1468_ = l_Lean_Expr_app___override(v___x_1467_, v_00_u03b1_1454_);
v___x_1469_ = l_Lean_Expr_app___override(v___x_1468_, v___i_1456_);
v___x_1470_ = l_Lean_Expr_app___override(v___x_1469_, v_e_1455_);
v___x_1471_ = l_Lean_Expr_app___override(v___x_1470_, v_lit_1458_);
v___x_1472_ = l_Lean_Expr_app___override(v___x_1471_, v_proof_1459_);
v___x_1473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1473_, 0, v___x_1464_);
lean_ctor_set(v___x_1473_, 1, v___x_1472_);
v___x_1474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1474_, 0, v___x_1461_);
lean_ctor_set(v___x_1474_, 1, v___x_1473_);
v___x_1475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1475_, 0, v___x_1474_);
return v___x_1475_;
}
case 2:
{
lean_object* v_lit_1476_; lean_object* v_proof_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; 
lean_dec_ref(v___i_1456_);
lean_dec_ref(v_e_1455_);
lean_dec_ref(v_00_u03b1_1454_);
lean_dec(v_u_1453_);
v_lit_1476_ = lean_ctor_get(v_x_1457_, 1);
lean_inc_ref(v_lit_1476_);
v_proof_1477_ = lean_ctor_get(v_x_1457_, 2);
lean_inc_ref(v_proof_1477_);
lean_dec_ref_known(v_x_1457_, 3);
v___x_1478_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1476_);
v___x_1479_ = lean_nat_to_int(v___x_1478_);
v___x_1480_ = lean_int_neg(v___x_1479_);
lean_dec(v___x_1479_);
v___x_1481_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1482_ = l_Lean_Expr_app___override(v___x_1481_, v_lit_1476_);
v___x_1483_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1483_, 0, v___x_1482_);
lean_ctor_set(v___x_1483_, 1, v_proof_1477_);
v___x_1484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1484_, 0, v___x_1480_);
lean_ctor_set(v___x_1484_, 1, v___x_1483_);
v___x_1485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1484_);
return v___x_1485_;
}
default: 
{
lean_object* v___x_1486_; 
lean_dec_ref(v_x_1457_);
lean_dec_ref(v___i_1456_);
lean_dec_ref(v_e_1455_);
lean_dec_ref(v_00_u03b1_1454_);
lean_dec(v_u_1453_);
v___x_1486_ = lean_box(0);
return v___x_1486_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___auto__1(void){
_start:
{
lean_object* v___x_1487_; 
v___x_1487_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12);
return v___x_1487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27(lean_object* v_u_1494_, lean_object* v_00_u03b1_1495_, lean_object* v_e_1496_, lean_object* v___i_1497_, lean_object* v_x_1498_){
_start:
{
switch(lean_obj_tag(v_x_1498_))
{
case 1:
{
lean_object* v_lit_1499_; lean_object* v_proof_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; 
v_lit_1499_ = lean_ctor_get(v_x_1498_, 1);
lean_inc_ref_n(v_lit_1499_, 2);
v_proof_1500_ = lean_ctor_get(v_x_1498_, 2);
lean_inc_ref(v_proof_1500_);
lean_dec_ref_known(v_x_1498_, 3);
v___x_1501_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1499_);
v___x_1502_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1501_);
v___x_1503_ = lean_box(0);
v___x_1504_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22);
v___x_1505_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1505_, 0, v_u_1494_);
lean_ctor_set(v___x_1505_, 1, v___x_1503_);
v___x_1506_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4));
lean_inc_ref(v___x_1505_);
v___x_1507_ = l_Lean_Expr_const___override(v___x_1506_, v___x_1505_);
lean_inc_ref(v_00_u03b1_1495_);
v___x_1508_ = l_Lean_Expr_app___override(v___x_1507_, v_00_u03b1_1495_);
v___x_1509_ = l_Lean_Expr_app___override(v___x_1508_, v___i_1497_);
v___x_1510_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0));
v___x_1511_ = l_Lean_Expr_const___override(v___x_1510_, v___x_1505_);
v___x_1512_ = l_Lean_Expr_app___override(v___x_1511_, v_00_u03b1_1495_);
v___x_1513_ = l_Lean_Expr_app___override(v___x_1512_, v___x_1509_);
v___x_1514_ = l_Lean_Expr_app___override(v___x_1513_, v_e_1496_);
v___x_1515_ = l_Lean_Expr_app___override(v___x_1514_, v_lit_1499_);
v___x_1516_ = l_Lean_Expr_app___override(v___x_1515_, v_proof_1500_);
v___x_1517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1517_, 0, v___x_1504_);
lean_ctor_set(v___x_1517_, 1, v___x_1516_);
v___x_1518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1518_, 0, v_lit_1499_);
lean_ctor_set(v___x_1518_, 1, v___x_1517_);
v___x_1519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1519_, 0, v___x_1502_);
lean_ctor_set(v___x_1519_, 1, v___x_1518_);
v___x_1520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1520_, 0, v___x_1519_);
return v___x_1520_;
}
case 3:
{
lean_object* v_q_1521_; lean_object* v_n_1522_; lean_object* v_d_1523_; lean_object* v_proof_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; 
lean_dec_ref(v___i_1497_);
lean_dec_ref(v_e_1496_);
lean_dec_ref(v_00_u03b1_1495_);
lean_dec(v_u_1494_);
v_q_1521_ = lean_ctor_get(v_x_1498_, 1);
lean_inc_ref(v_q_1521_);
v_n_1522_ = lean_ctor_get(v_x_1498_, 2);
lean_inc_ref(v_n_1522_);
v_d_1523_ = lean_ctor_get(v_x_1498_, 3);
lean_inc_ref(v_d_1523_);
v_proof_1524_ = lean_ctor_get(v_x_1498_, 4);
lean_inc_ref(v_proof_1524_);
lean_dec_ref_known(v_x_1498_, 5);
v___x_1525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1525_, 0, v_d_1523_);
lean_ctor_set(v___x_1525_, 1, v_proof_1524_);
v___x_1526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1526_, 0, v_n_1522_);
lean_ctor_set(v___x_1526_, 1, v___x_1525_);
v___x_1527_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1527_, 0, v_q_1521_);
lean_ctor_set(v___x_1527_, 1, v___x_1526_);
v___x_1528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1528_, 0, v___x_1527_);
return v___x_1528_;
}
default: 
{
lean_object* v___x_1529_; 
lean_dec_ref(v_x_1498_);
lean_dec_ref(v___i_1497_);
lean_dec_ref(v_e_1496_);
lean_dec_ref(v_00_u03b1_1495_);
lean_dec(v_u_1494_);
v___x_1529_ = lean_box(0);
return v___x_1529_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___auto__1(void){
_start:
{
lean_object* v___x_1530_; 
v___x_1530_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12, &lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1___closed__12);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27(lean_object* v_u_1547_, lean_object* v_00_u03b1_1548_, lean_object* v_e_1549_, lean_object* v___i_1550_, lean_object* v_x_1551_){
_start:
{
switch(lean_obj_tag(v_x_1551_))
{
case 0:
{
lean_object* v___x_1552_; 
lean_dec_ref_known(v_x_1551_, 1);
lean_dec_ref(v___i_1550_);
lean_dec_ref(v_e_1549_);
lean_dec_ref(v_00_u03b1_1548_);
lean_dec(v_u_1547_);
v___x_1552_ = lean_box(0);
return v___x_1552_;
}
case 1:
{
lean_object* v_lit_1553_; lean_object* v_proof_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; 
v_lit_1553_ = lean_ctor_get(v_x_1551_, 1);
lean_inc_ref_n(v_lit_1553_, 3);
v_proof_1554_ = lean_ctor_get(v_x_1551_, 2);
lean_inc_ref(v_proof_1554_);
lean_dec_ref_known(v_x_1551_, 3);
v___x_1555_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1553_);
v___x_1556_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1555_);
v___x_1557_ = lean_box(0);
v___x_1558_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7);
v___x_1559_ = l_Lean_Expr_app___override(v___x_1558_, v_lit_1553_);
v___x_1560_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22);
v___x_1561_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1561_, 0, v_u_1547_);
lean_ctor_set(v___x_1561_, 1, v___x_1557_);
v___x_1562_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
lean_inc_ref_n(v___x_1561_, 3);
v___x_1563_ = l_Lean_Expr_const___override(v___x_1562_, v___x_1561_);
lean_inc_ref_n(v_00_u03b1_1548_, 3);
v___x_1564_ = l_Lean_Expr_app___override(v___x_1563_, v_00_u03b1_1548_);
v___x_1565_ = l_Lean_Expr_app___override(v___x_1564_, v___i_1550_);
v___x_1566_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1));
v___x_1567_ = l_Lean_Expr_const___override(v___x_1566_, v___x_1561_);
v___x_1568_ = l_Lean_Expr_app___override(v___x_1567_, v_00_u03b1_1548_);
lean_inc_ref(v___x_1565_);
v___x_1569_ = l_Lean_Expr_app___override(v___x_1568_, v___x_1565_);
lean_inc_ref(v_e_1549_);
v___x_1570_ = l_Lean_Expr_app___override(v___x_1569_, v_e_1549_);
v___x_1571_ = l_Lean_Expr_app___override(v___x_1570_, v_lit_1553_);
v___x_1572_ = l_Lean_Expr_app___override(v___x_1571_, v___x_1560_);
v___x_1573_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___closed__0));
v___x_1574_ = l_Lean_Expr_const___override(v___x_1573_, v___x_1561_);
v___x_1575_ = l_Lean_Expr_app___override(v___x_1574_, v_00_u03b1_1548_);
v___x_1576_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2));
v___x_1577_ = l_Lean_Expr_const___override(v___x_1576_, v___x_1561_);
v___x_1578_ = l_Lean_Expr_app___override(v___x_1577_, v_00_u03b1_1548_);
v___x_1579_ = l_Lean_Expr_app___override(v___x_1578_, v___x_1565_);
v___x_1580_ = l_Lean_Expr_app___override(v___x_1575_, v___x_1579_);
v___x_1581_ = l_Lean_Expr_app___override(v___x_1580_, v_e_1549_);
v___x_1582_ = l_Lean_Expr_app___override(v___x_1581_, v_lit_1553_);
v___x_1583_ = l_Lean_Expr_app___override(v___x_1582_, v_proof_1554_);
v___x_1584_ = l_Lean_Expr_app___override(v___x_1572_, v___x_1583_);
v___x_1585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1585_, 0, v___x_1560_);
lean_ctor_set(v___x_1585_, 1, v___x_1584_);
v___x_1586_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1586_, 0, v___x_1559_);
lean_ctor_set(v___x_1586_, 1, v___x_1585_);
v___x_1587_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1587_, 0, v___x_1556_);
lean_ctor_set(v___x_1587_, 1, v___x_1586_);
v___x_1588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1588_, 0, v___x_1587_);
return v___x_1588_;
}
case 2:
{
lean_object* v_lit_1589_; lean_object* v_proof_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
v_lit_1589_ = lean_ctor_get(v_x_1551_, 1);
lean_inc_ref(v_lit_1589_);
v_proof_1590_ = lean_ctor_get(v_x_1551_, 2);
lean_inc_ref(v_proof_1590_);
lean_dec_ref_known(v_x_1551_, 3);
v___x_1591_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1589_);
v___x_1592_ = lp_mathlib_Nat_cast___at___00Mathlib_Meta_NormNum_Result_isRat_spec__0(v___x_1591_);
v___x_1593_ = l_Rat_neg(v___x_1592_);
v___x_1594_ = lean_box(0);
v___x_1595_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1596_ = l_Lean_Expr_app___override(v___x_1595_, v_lit_1589_);
v___x_1597_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22, &lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkOfNat___closed__22);
v___x_1598_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1598_, 0, v_u_1547_);
lean_ctor_set(v___x_1598_, 1, v___x_1594_);
v___x_1599_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
lean_inc_ref(v___x_1598_);
v___x_1600_ = l_Lean_Expr_const___override(v___x_1599_, v___x_1598_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1601_ = l_Lean_Expr_app___override(v___x_1600_, v_00_u03b1_1548_);
v___x_1602_ = l_Lean_Expr_app___override(v___x_1601_, v___i_1550_);
v___x_1603_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__3));
v___x_1604_ = l_Lean_Expr_const___override(v___x_1603_, v___x_1598_);
v___x_1605_ = l_Lean_Expr_app___override(v___x_1604_, v_00_u03b1_1548_);
v___x_1606_ = l_Lean_Expr_app___override(v___x_1605_, v___x_1602_);
v___x_1607_ = l_Lean_Expr_app___override(v___x_1606_, v_e_1549_);
lean_inc_ref(v___x_1596_);
v___x_1608_ = l_Lean_Expr_app___override(v___x_1607_, v___x_1596_);
v___x_1609_ = l_Lean_Expr_app___override(v___x_1608_, v_proof_1590_);
v___x_1610_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1610_, 0, v___x_1597_);
lean_ctor_set(v___x_1610_, 1, v___x_1609_);
v___x_1611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1611_, 0, v___x_1596_);
lean_ctor_set(v___x_1611_, 1, v___x_1610_);
v___x_1612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1593_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
v___x_1613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1613_, 0, v___x_1612_);
return v___x_1613_;
}
case 3:
{
lean_object* v_q_1614_; lean_object* v_n_1615_; lean_object* v_d_1616_; lean_object* v_proof_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; 
v_q_1614_ = lean_ctor_get(v_x_1551_, 1);
lean_inc_ref(v_q_1614_);
v_n_1615_ = lean_ctor_get(v_x_1551_, 2);
lean_inc_ref_n(v_n_1615_, 2);
v_d_1616_ = lean_ctor_get(v_x_1551_, 3);
lean_inc_ref_n(v_d_1616_, 2);
v_proof_1617_ = lean_ctor_get(v_x_1551_, 4);
lean_inc_ref(v_proof_1617_);
lean_dec_ref_known(v_x_1551_, 5);
v___x_1618_ = lean_box(0);
v___x_1619_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__7);
v___x_1620_ = l_Lean_Expr_app___override(v___x_1619_, v_n_1615_);
v___x_1621_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1621_, 0, v_u_1547_);
lean_ctor_set(v___x_1621_, 1, v___x_1618_);
v___x_1622_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
lean_inc_ref(v___x_1621_);
v___x_1623_ = l_Lean_Expr_const___override(v___x_1622_, v___x_1621_);
lean_inc_ref(v_00_u03b1_1548_);
v___x_1624_ = l_Lean_Expr_app___override(v___x_1623_, v_00_u03b1_1548_);
v___x_1625_ = l_Lean_Expr_app___override(v___x_1624_, v___i_1550_);
v___x_1626_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__1));
v___x_1627_ = l_Lean_Expr_const___override(v___x_1626_, v___x_1621_);
v___x_1628_ = l_Lean_Expr_app___override(v___x_1627_, v_00_u03b1_1548_);
v___x_1629_ = l_Lean_Expr_app___override(v___x_1628_, v___x_1625_);
v___x_1630_ = l_Lean_Expr_app___override(v___x_1629_, v_e_1549_);
v___x_1631_ = l_Lean_Expr_app___override(v___x_1630_, v_n_1615_);
v___x_1632_ = l_Lean_Expr_app___override(v___x_1631_, v_d_1616_);
v___x_1633_ = l_Lean_Expr_app___override(v___x_1632_, v_proof_1617_);
v___x_1634_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1634_, 0, v_d_1616_);
lean_ctor_set(v___x_1634_, 1, v___x_1633_);
v___x_1635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1635_, 0, v___x_1620_);
lean_ctor_set(v___x_1635_, 1, v___x_1634_);
v___x_1636_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1636_, 0, v_q_1614_);
lean_ctor_set(v___x_1636_, 1, v___x_1635_);
v___x_1637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1637_, 0, v___x_1636_);
return v___x_1637_;
}
default: 
{
lean_object* v_q_1638_; lean_object* v_n_1639_; lean_object* v_d_1640_; lean_object* v_proof_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; 
lean_dec_ref(v___i_1550_);
lean_dec_ref(v_e_1549_);
lean_dec_ref(v_00_u03b1_1548_);
lean_dec(v_u_1547_);
v_q_1638_ = lean_ctor_get(v_x_1551_, 1);
lean_inc_ref(v_q_1638_);
v_n_1639_ = lean_ctor_get(v_x_1551_, 2);
lean_inc_ref(v_n_1639_);
v_d_1640_ = lean_ctor_get(v_x_1551_, 3);
lean_inc_ref(v_d_1640_);
v_proof_1641_ = lean_ctor_get(v_x_1551_, 4);
lean_inc_ref(v_proof_1641_);
lean_dec_ref_known(v_x_1551_, 5);
v___x_1642_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1643_ = l_Lean_Expr_app___override(v___x_1642_, v_n_1639_);
v___x_1644_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1644_, 0, v_d_1640_);
lean_ctor_set(v___x_1644_, 1, v_proof_1641_);
v___x_1645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1645_, 0, v___x_1643_);
lean_ctor_set(v___x_1645_, 1, v___x_1644_);
v___x_1646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1646_, 0, v_q_1638_);
lean_ctor_set(v___x_1646_, 1, v___x_1645_);
v___x_1647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1647_, 0, v___x_1646_);
return v___x_1647_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2(void){
_start:
{
lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; 
v___x_1651_ = lean_box(0);
v___x_1652_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__1));
v___x_1653_ = l_Lean_Expr_const___override(v___x_1652_, v___x_1651_);
return v___x_1653_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5(void){
_start:
{
lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; 
v___x_1657_ = lean_box(0);
v___x_1658_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__4));
v___x_1659_ = l_Lean_Expr_const___override(v___x_1658_, v___x_1657_);
return v___x_1659_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8(void){
_start:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1663_ = lean_box(0);
v___x_1664_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__7));
v___x_1665_ = l_Lean_Expr_const___override(v___x_1664_, v___x_1663_);
return v___x_1665_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11(void){
_start:
{
lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; 
v___x_1669_ = lean_box(0);
v___x_1670_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__10));
v___x_1671_ = l_Lean_Expr_const___override(v___x_1670_, v___x_1669_);
return v___x_1671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq(lean_object* v_u_1708_, lean_object* v_00_u03b1_1709_, lean_object* v_e_1710_, lean_object* v_x_1711_){
_start:
{
switch(lean_obj_tag(v_x_1711_))
{
case 0:
{
uint8_t v_val_1712_; 
lean_dec_ref(v_00_u03b1_1709_);
lean_dec(v_u_1708_);
v_val_1712_ = lean_ctor_get_uint8(v_x_1711_, sizeof(void*)*1);
if (v_val_1712_ == 0)
{
lean_object* v_proof_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v_proof_1713_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref(v_proof_1713_);
lean_dec_ref_known(v_x_1711_, 1);
v___x_1714_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__2);
v___x_1715_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__5);
v___x_1716_ = l_Lean_Expr_app___override(v___x_1715_, v_e_1710_);
v___x_1717_ = l_Lean_Expr_app___override(v___x_1716_, v_proof_1713_);
v___x_1718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1718_, 0, v___x_1714_);
lean_ctor_set(v___x_1718_, 1, v___x_1717_);
return v___x_1718_;
}
else
{
lean_object* v_proof_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; 
v_proof_1719_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref(v_proof_1719_);
lean_dec_ref_known(v_x_1711_, 1);
v___x_1720_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__8);
v___x_1721_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11, &lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__11);
v___x_1722_ = l_Lean_Expr_app___override(v___x_1721_, v_e_1710_);
v___x_1723_ = l_Lean_Expr_app___override(v___x_1722_, v_proof_1719_);
v___x_1724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1724_, 0, v___x_1720_);
lean_ctor_set(v___x_1724_, 1, v___x_1723_);
return v___x_1724_;
}
}
case 1:
{
lean_object* v_inst_1725_; lean_object* v_lit_1726_; lean_object* v_proof_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; 
v_inst_1725_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref_n(v_inst_1725_, 2);
v_lit_1726_ = lean_ctor_get(v_x_1711_, 1);
lean_inc_ref_n(v_lit_1726_, 2);
v_proof_1727_ = lean_ctor_get(v_x_1711_, 2);
lean_inc_ref(v_proof_1727_);
lean_dec_ref_known(v_x_1711_, 3);
v___x_1728_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_1729_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12));
v___x_1730_ = l_Lean_Name_mkStr2(v___x_1728_, v___x_1729_);
v___x_1731_ = lean_box(0);
v___x_1732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1732_, 0, v_u_1708_);
lean_ctor_set(v___x_1732_, 1, v___x_1731_);
lean_inc_ref(v___x_1732_);
v___x_1733_ = l_Lean_Expr_const___override(v___x_1730_, v___x_1732_);
lean_inc_ref(v_00_u03b1_1709_);
v___x_1734_ = l_Lean_Expr_app___override(v___x_1733_, v_00_u03b1_1709_);
v___x_1735_ = l_Lean_Expr_app___override(v___x_1734_, v_inst_1725_);
v___x_1736_ = l_Lean_Expr_app___override(v___x_1735_, v_lit_1726_);
v___x_1737_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14));
v___x_1738_ = l_Lean_Expr_const___override(v___x_1737_, v___x_1732_);
v___x_1739_ = l_Lean_Expr_app___override(v___x_1738_, v_00_u03b1_1709_);
v___x_1740_ = l_Lean_Expr_app___override(v___x_1739_, v_e_1710_);
v___x_1741_ = l_Lean_Expr_app___override(v___x_1740_, v_lit_1726_);
v___x_1742_ = l_Lean_Expr_app___override(v___x_1741_, v_inst_1725_);
v___x_1743_ = l_Lean_Expr_app___override(v___x_1742_, v_proof_1727_);
v___x_1744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1744_, 0, v___x_1736_);
lean_ctor_set(v___x_1744_, 1, v___x_1743_);
return v___x_1744_;
}
case 2:
{
lean_object* v_inst_1745_; lean_object* v_lit_1746_; lean_object* v_proof_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; 
v_inst_1745_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref_n(v_inst_1745_, 2);
v_lit_1746_ = lean_ctor_get(v_x_1711_, 1);
lean_inc_ref(v_lit_1746_);
v_proof_1747_ = lean_ctor_get(v_x_1711_, 2);
lean_inc_ref(v_proof_1747_);
lean_dec_ref_known(v_x_1711_, 3);
v___x_1748_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15));
v___x_1749_ = lean_box(0);
v___x_1750_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1750_, 0, v_u_1708_);
lean_ctor_set(v___x_1750_, 1, v___x_1749_);
lean_inc_ref(v___x_1750_);
v___x_1751_ = l_Lean_Expr_const___override(v___x_1748_, v___x_1750_);
lean_inc_ref(v_00_u03b1_1709_);
v___x_1752_ = l_Lean_Expr_app___override(v___x_1751_, v_00_u03b1_1709_);
v___x_1753_ = l_Lean_Expr_app___override(v___x_1752_, v_inst_1745_);
v___x_1754_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1755_ = l_Lean_Expr_app___override(v___x_1754_, v_lit_1746_);
lean_inc_ref(v___x_1755_);
v___x_1756_ = l_Lean_Expr_app___override(v___x_1753_, v___x_1755_);
v___x_1757_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16));
v___x_1758_ = l_Lean_Expr_const___override(v___x_1757_, v___x_1750_);
v___x_1759_ = l_Lean_Expr_app___override(v___x_1758_, v_00_u03b1_1709_);
v___x_1760_ = l_Lean_Expr_app___override(v___x_1759_, v_e_1710_);
v___x_1761_ = l_Lean_Expr_app___override(v___x_1760_, v___x_1755_);
v___x_1762_ = l_Lean_Expr_app___override(v___x_1761_, v_inst_1745_);
v___x_1763_ = l_Lean_Expr_app___override(v___x_1762_, v_proof_1747_);
v___x_1764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1764_, 0, v___x_1756_);
lean_ctor_set(v___x_1764_, 1, v___x_1763_);
return v___x_1764_;
}
case 3:
{
lean_object* v_inst_1765_; lean_object* v_n_1766_; lean_object* v_d_1767_; lean_object* v_proof_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; 
v_inst_1765_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref_n(v_inst_1765_, 2);
v_n_1766_ = lean_ctor_get(v_x_1711_, 2);
lean_inc_ref_n(v_n_1766_, 2);
v_d_1767_ = lean_ctor_get(v_x_1711_, 3);
lean_inc_ref_n(v_d_1767_, 2);
v_proof_1768_ = lean_ctor_get(v_x_1711_, 4);
lean_inc_ref(v_proof_1768_);
lean_dec_ref_known(v_x_1711_, 5);
v___x_1769_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__18));
v___x_1770_ = lean_box(0);
v___x_1771_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1771_, 0, v_u_1708_);
lean_ctor_set(v___x_1771_, 1, v___x_1770_);
lean_inc_ref(v___x_1771_);
v___x_1772_ = l_Lean_Expr_const___override(v___x_1769_, v___x_1771_);
lean_inc_ref(v_00_u03b1_1709_);
v___x_1773_ = l_Lean_Expr_app___override(v___x_1772_, v_00_u03b1_1709_);
v___x_1774_ = l_Lean_Expr_app___override(v___x_1773_, v_inst_1765_);
v___x_1775_ = l_Lean_Expr_app___override(v___x_1774_, v_n_1766_);
v___x_1776_ = l_Lean_Expr_app___override(v___x_1775_, v_d_1767_);
v___x_1777_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__19));
v___x_1778_ = l_Lean_Expr_const___override(v___x_1777_, v___x_1771_);
v___x_1779_ = l_Lean_Expr_app___override(v___x_1778_, v_00_u03b1_1709_);
v___x_1780_ = l_Lean_Expr_app___override(v___x_1779_, v_n_1766_);
v___x_1781_ = l_Lean_Expr_app___override(v___x_1780_, v_d_1767_);
v___x_1782_ = l_Lean_Expr_app___override(v___x_1781_, v_inst_1765_);
v___x_1783_ = l_Lean_Expr_app___override(v___x_1782_, v_e_1710_);
v___x_1784_ = l_Lean_Expr_app___override(v___x_1783_, v_proof_1768_);
v___x_1785_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1785_, 0, v___x_1776_);
lean_ctor_set(v___x_1785_, 1, v___x_1784_);
return v___x_1785_;
}
default: 
{
lean_object* v_inst_1786_; lean_object* v_n_1787_; lean_object* v_d_1788_; lean_object* v_proof_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; 
v_inst_1786_ = lean_ctor_get(v_x_1711_, 0);
lean_inc_ref_n(v_inst_1786_, 2);
v_n_1787_ = lean_ctor_get(v_x_1711_, 2);
lean_inc_ref(v_n_1787_);
v_d_1788_ = lean_ctor_get(v_x_1711_, 3);
lean_inc_ref_n(v_d_1788_, 2);
v_proof_1789_ = lean_ctor_get(v_x_1711_, 4);
lean_inc_ref(v_proof_1789_);
lean_dec_ref_known(v_x_1711_, 5);
v___x_1790_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__20));
v___x_1791_ = lean_box(0);
v___x_1792_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1792_, 0, v_u_1708_);
lean_ctor_set(v___x_1792_, 1, v___x_1791_);
lean_inc_ref(v___x_1792_);
v___x_1793_ = l_Lean_Expr_const___override(v___x_1790_, v___x_1792_);
lean_inc_ref(v_00_u03b1_1709_);
v___x_1794_ = l_Lean_Expr_app___override(v___x_1793_, v_00_u03b1_1709_);
v___x_1795_ = l_Lean_Expr_app___override(v___x_1794_, v_inst_1786_);
v___x_1796_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1797_ = l_Lean_Expr_app___override(v___x_1796_, v_n_1787_);
lean_inc_ref(v___x_1797_);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1795_, v___x_1797_);
v___x_1799_ = l_Lean_Expr_app___override(v___x_1798_, v_d_1788_);
v___x_1800_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__21));
v___x_1801_ = l_Lean_Expr_const___override(v___x_1800_, v___x_1792_);
v___x_1802_ = l_Lean_Expr_app___override(v___x_1801_, v_00_u03b1_1709_);
v___x_1803_ = l_Lean_Expr_app___override(v___x_1802_, v___x_1797_);
v___x_1804_ = l_Lean_Expr_app___override(v___x_1803_, v_d_1788_);
v___x_1805_ = l_Lean_Expr_app___override(v___x_1804_, v_inst_1786_);
v___x_1806_ = l_Lean_Expr_app___override(v___x_1805_, v_e_1710_);
v___x_1807_ = l_Lean_Expr_app___override(v___x_1806_, v_proof_1789_);
v___x_1808_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1799_);
lean_ctor_set(v___x_1808_, 1, v___x_1807_);
return v___x_1808_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawIntEq(lean_object* v_u_1809_, lean_object* v_00_u03b1_1810_, lean_object* v_e_1811_, lean_object* v_x_1812_){
_start:
{
switch(lean_obj_tag(v_x_1812_))
{
case 1:
{
lean_object* v_inst_1813_; lean_object* v_lit_1814_; lean_object* v_proof_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; 
v_inst_1813_ = lean_ctor_get(v_x_1812_, 0);
lean_inc_ref_n(v_inst_1813_, 2);
v_lit_1814_ = lean_ctor_get(v_x_1812_, 1);
lean_inc_ref_n(v_lit_1814_, 2);
v_proof_1815_ = lean_ctor_get(v_x_1812_, 2);
lean_inc_ref(v_proof_1815_);
lean_dec_ref_known(v_x_1812_, 3);
v___x_1816_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1814_);
v___x_1817_ = lean_nat_to_int(v___x_1816_);
v___x_1818_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_rawIntLitNatAbs_spec__0___closed__0));
v___x_1819_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__12));
v___x_1820_ = l_Lean_Name_mkStr2(v___x_1818_, v___x_1819_);
v___x_1821_ = lean_box(0);
v___x_1822_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1822_, 0, v_u_1809_);
lean_ctor_set(v___x_1822_, 1, v___x_1821_);
lean_inc_ref(v___x_1822_);
v___x_1823_ = l_Lean_Expr_const___override(v___x_1820_, v___x_1822_);
lean_inc_ref(v_00_u03b1_1810_);
v___x_1824_ = l_Lean_Expr_app___override(v___x_1823_, v_00_u03b1_1810_);
v___x_1825_ = l_Lean_Expr_app___override(v___x_1824_, v_inst_1813_);
v___x_1826_ = l_Lean_Expr_app___override(v___x_1825_, v_lit_1814_);
v___x_1827_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__14));
v___x_1828_ = l_Lean_Expr_const___override(v___x_1827_, v___x_1822_);
v___x_1829_ = l_Lean_Expr_app___override(v___x_1828_, v_00_u03b1_1810_);
v___x_1830_ = l_Lean_Expr_app___override(v___x_1829_, v_e_1811_);
v___x_1831_ = l_Lean_Expr_app___override(v___x_1830_, v_lit_1814_);
v___x_1832_ = l_Lean_Expr_app___override(v___x_1831_, v_inst_1813_);
v___x_1833_ = l_Lean_Expr_app___override(v___x_1832_, v_proof_1815_);
v___x_1834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1826_);
lean_ctor_set(v___x_1834_, 1, v___x_1833_);
v___x_1835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1817_);
lean_ctor_set(v___x_1835_, 1, v___x_1834_);
v___x_1836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1835_);
return v___x_1836_;
}
case 2:
{
lean_object* v_inst_1837_; lean_object* v_lit_1838_; lean_object* v_proof_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; 
v_inst_1837_ = lean_ctor_get(v_x_1812_, 0);
lean_inc_ref_n(v_inst_1837_, 2);
v_lit_1838_ = lean_ctor_get(v_x_1812_, 1);
lean_inc_ref(v_lit_1838_);
v_proof_1839_ = lean_ctor_get(v_x_1812_, 2);
lean_inc_ref(v_proof_1839_);
lean_dec_ref_known(v_x_1812_, 3);
v___x_1840_ = lp_batteries_Lean_Expr_natLit_x21(v_lit_1838_);
v___x_1841_ = lean_nat_to_int(v___x_1840_);
v___x_1842_ = lean_int_neg(v___x_1841_);
lean_dec(v___x_1841_);
v___x_1843_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__15));
v___x_1844_ = lean_box(0);
v___x_1845_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1845_, 0, v_u_1809_);
lean_ctor_set(v___x_1845_, 1, v___x_1844_);
lean_inc_ref(v___x_1845_);
v___x_1846_ = l_Lean_Expr_const___override(v___x_1843_, v___x_1845_);
lean_inc_ref(v_00_u03b1_1810_);
v___x_1847_ = l_Lean_Expr_app___override(v___x_1846_, v_00_u03b1_1810_);
v___x_1848_ = l_Lean_Expr_app___override(v___x_1847_, v_inst_1837_);
v___x_1849_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1850_ = l_Lean_Expr_app___override(v___x_1849_, v_lit_1838_);
lean_inc_ref(v___x_1850_);
v___x_1851_ = l_Lean_Expr_app___override(v___x_1848_, v___x_1850_);
v___x_1852_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq___closed__16));
v___x_1853_ = l_Lean_Expr_const___override(v___x_1852_, v___x_1845_);
v___x_1854_ = l_Lean_Expr_app___override(v___x_1853_, v_00_u03b1_1810_);
v___x_1855_ = l_Lean_Expr_app___override(v___x_1854_, v_e_1811_);
v___x_1856_ = l_Lean_Expr_app___override(v___x_1855_, v___x_1850_);
v___x_1857_ = l_Lean_Expr_app___override(v___x_1856_, v_inst_1837_);
v___x_1858_ = l_Lean_Expr_app___override(v___x_1857_, v_proof_1839_);
v___x_1859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1859_, 0, v___x_1851_);
lean_ctor_set(v___x_1859_, 1, v___x_1858_);
v___x_1860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1860_, 0, v___x_1842_);
lean_ctor_set(v___x_1860_, 1, v___x_1859_);
v___x_1861_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1861_, 0, v___x_1860_);
return v___x_1861_;
}
default: 
{
lean_object* v___x_1862_; 
lean_dec_ref(v_x_1812_);
lean_dec_ref(v_e_1811_);
lean_dec_ref(v_00_u03b1_1810_);
lean_dec(v_u_1809_);
v___x_1862_ = lean_box(0);
return v___x_1862_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(lean_object* v_msg_1870_){
_start:
{
lean_object* v___f_1871_; lean_object* v___f_1872_; lean_object* v___f_1873_; lean_object* v___f_1874_; lean_object* v___f_1875_; lean_object* v___f_1876_; lean_object* v___f_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; 
v___f_1871_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__0));
v___f_1872_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__1));
v___f_1873_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__2));
v___f_1874_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__3));
v___f_1875_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__4));
v___f_1876_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__5));
v___f_1877_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg___closed__6));
v___x_1878_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1878_, 0, v___f_1871_);
lean_ctor_set(v___x_1878_, 1, v___f_1872_);
v___x_1879_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1879_, 0, v___x_1878_);
lean_ctor_set(v___x_1879_, 1, v___f_1873_);
lean_ctor_set(v___x_1879_, 2, v___f_1874_);
lean_ctor_set(v___x_1879_, 3, v___f_1875_);
lean_ctor_set(v___x_1879_, 4, v___f_1876_);
v___x_1880_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1880_, 0, v___x_1879_);
lean_ctor_set(v___x_1880_, 1, v___f_1877_);
v___x_1881_ = lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default;
v___x_1882_ = l_instInhabitedOfMonad___redArg(v___x_1880_, v___x_1881_);
v___x_1883_ = lean_panic_fn_borrowed(v___x_1882_, v_msg_1870_);
lean_dec(v___x_1882_);
return v___x_1883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0(lean_object* v_u_1884_, lean_object* v_00_u03b1_1885_, lean_object* v_e_1886_, lean_object* v_msg_1887_){
_start:
{
lean_object* v___x_1888_; 
v___x_1888_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(v_msg_1887_);
return v___x_1888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___boxed(lean_object* v_u_1889_, lean_object* v_00_u03b1_1890_, lean_object* v_e_1891_, lean_object* v_msg_1892_){
_start:
{
lean_object* v_res_1893_; 
v_res_1893_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0(v_u_1889_, v_00_u03b1_1890_, v_e_1891_, v_msg_1892_);
lean_dec_ref(v_e_1891_);
lean_dec_ref(v_00_u03b1_1890_);
lean_dec(v_u_1889_);
return v_res_1893_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2(void){
_start:
{
lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; 
v___x_1896_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__1));
v___x_1897_ = lean_unsigned_to_nat(70u);
v___x_1898_ = lean_unsigned_to_nat(497u);
v___x_1899_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__0));
v___x_1900_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11));
v___x_1901_ = l_mkPanicMessageWithDecl(v___x_1900_, v___x_1899_, v___x_1898_, v___x_1897_, v___x_1896_);
return v___x_1901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat(lean_object* v_u_1909_, lean_object* v_00_u03b1_1910_, lean_object* v_e_1911_){
_start:
{
if (lean_obj_tag(v_e_1911_) == 5)
{
lean_object* v_fn_1915_; 
v_fn_1915_ = lean_ctor_get(v_e_1911_, 0);
lean_inc_ref(v_fn_1915_);
if (lean_obj_tag(v_fn_1915_) == 5)
{
lean_object* v_arg_1916_; lean_object* v_arg_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; 
v_arg_1916_ = lean_ctor_get(v_e_1911_, 1);
lean_inc_ref_n(v_arg_1916_, 2);
lean_dec_ref_known(v_e_1911_, 2);
v_arg_1917_ = lean_ctor_get(v_fn_1915_, 1);
lean_inc_ref_n(v_arg_1917_, 2);
lean_dec_ref_known(v_fn_1915_, 2);
v___x_1918_ = lean_box(0);
v___x_1919_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1919_, 0, v_u_1909_);
lean_ctor_set(v___x_1919_, 1, v___x_1918_);
v___x_1920_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__4));
v___x_1921_ = l_Lean_Expr_const___override(v___x_1920_, v___x_1919_);
v___x_1922_ = l_Lean_Expr_app___override(v___x_1921_, v_00_u03b1_1910_);
v___x_1923_ = l_Lean_Expr_app___override(v___x_1922_, v_arg_1917_);
v___x_1924_ = l_Lean_Expr_app___override(v___x_1923_, v_arg_1916_);
v___x_1925_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1925_, 0, v_arg_1917_);
lean_ctor_set(v___x_1925_, 1, v_arg_1916_);
lean_ctor_set(v___x_1925_, 2, v___x_1924_);
return v___x_1925_;
}
else
{
lean_dec_ref(v_fn_1915_);
lean_dec_ref_known(v_e_1911_, 2);
lean_dec_ref(v_00_u03b1_1910_);
lean_dec(v_u_1909_);
goto v___jp_1912_;
}
}
else
{
lean_dec_ref(v_e_1911_);
lean_dec_ref(v_00_u03b1_1910_);
lean_dec(v_u_1909_);
goto v___jp_1912_;
}
v___jp_1912_:
{
lean_object* v___x_1913_; lean_object* v___x_1914_; 
v___x_1913_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat___closed__2);
v___x_1914_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(v___x_1913_);
return v___x_1914_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2(void){
_start:
{
lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; 
v___x_1928_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__1));
v___x_1929_ = lean_unsigned_to_nat(69u);
v___x_1930_ = lean_unsigned_to_nat(506u);
v___x_1931_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__0));
v___x_1932_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11));
v___x_1933_ = l_mkPanicMessageWithDecl(v___x_1932_, v___x_1931_, v___x_1930_, v___x_1929_, v___x_1928_);
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt(lean_object* v_u_1940_, lean_object* v_00_u03b1_1941_, lean_object* v_n_1942_, lean_object* v_e_1943_){
_start:
{
lean_object* v___x_1947_; uint8_t v___x_1948_; 
v___x_1947_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__0);
v___x_1948_ = lean_int_dec_le(v___x_1947_, v_n_1942_);
if (v___x_1948_ == 0)
{
if (lean_obj_tag(v_e_1943_) == 5)
{
lean_object* v_fn_1949_; 
v_fn_1949_ = lean_ctor_get(v_e_1943_, 0);
lean_inc_ref(v_fn_1949_);
if (lean_obj_tag(v_fn_1949_) == 5)
{
lean_object* v_arg_1950_; 
v_arg_1950_ = lean_ctor_get(v_e_1943_, 1);
lean_inc_ref(v_arg_1950_);
lean_dec_ref_known(v_e_1943_, 2);
if (lean_obj_tag(v_arg_1950_) == 5)
{
lean_object* v_arg_1951_; lean_object* v_arg_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; 
v_arg_1951_ = lean_ctor_get(v_fn_1949_, 1);
lean_inc_ref_n(v_arg_1951_, 2);
lean_dec_ref_known(v_fn_1949_, 2);
v_arg_1952_ = lean_ctor_get(v_arg_1950_, 1);
lean_inc_ref_n(v_arg_1952_, 2);
lean_dec_ref_known(v_arg_1950_, 2);
v___x_1953_ = lean_box(0);
v___x_1954_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1954_, 0, v_u_1940_);
lean_ctor_set(v___x_1954_, 1, v___x_1953_);
v___x_1955_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
v___x_1956_ = l_Lean_Expr_app___override(v___x_1955_, v_arg_1952_);
v___x_1957_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__3));
v___x_1958_ = l_Lean_Expr_const___override(v___x_1957_, v___x_1954_);
v___x_1959_ = l_Lean_Expr_app___override(v___x_1958_, v_00_u03b1_1941_);
v___x_1960_ = l_Lean_Expr_app___override(v___x_1959_, v_arg_1951_);
v___x_1961_ = l_Lean_Expr_app___override(v___x_1960_, v___x_1956_);
v___x_1962_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1962_, 0, v_arg_1951_);
lean_ctor_set(v___x_1962_, 1, v_arg_1952_);
lean_ctor_set(v___x_1962_, 2, v___x_1961_);
return v___x_1962_;
}
else
{
lean_dec_ref_known(v_fn_1949_, 2);
lean_dec_ref(v_arg_1950_);
lean_dec_ref(v_00_u03b1_1941_);
lean_dec(v_u_1940_);
goto v___jp_1944_;
}
}
else
{
lean_dec_ref(v_fn_1949_);
lean_dec_ref_known(v_e_1943_, 2);
lean_dec_ref(v_00_u03b1_1941_);
lean_dec(v_u_1940_);
goto v___jp_1944_;
}
}
else
{
lean_dec_ref(v_e_1943_);
lean_dec_ref(v_00_u03b1_1941_);
lean_dec(v_u_1940_);
goto v___jp_1944_;
}
}
else
{
lean_object* v___x_1963_; 
v___x_1963_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat(v_u_1940_, v_00_u03b1_1941_, v_e_1943_);
return v___x_1963_;
}
v___jp_1944_:
{
lean_object* v___x_1945_; lean_object* v___x_1946_; 
v___x_1945_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___closed__2);
v___x_1946_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(v___x_1945_);
return v___x_1946_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt___boxed(lean_object* v_u_1964_, lean_object* v_00_u03b1_1965_, lean_object* v_n_1966_, lean_object* v_e_1967_){
_start:
{
lean_object* v_res_1968_; 
v_res_1968_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt(v_u_1964_, v_00_u03b1_1965_, v_n_1966_, v_e_1967_);
lean_dec(v_n_1966_);
return v_res_1968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNNRat_spec__0(lean_object* v_msg_1969_){
_start:
{
lean_object* v___x_1970_; lean_object* v___x_1971_; 
v___x_1970_ = l_Lean_instInhabitedExpr;
v___x_1971_ = lean_panic_fn_borrowed(v___x_1970_, v_msg_1969_);
return v___x_1971_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2(void){
_start:
{
lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; 
v___x_1974_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__1));
v___x_1975_ = lean_unsigned_to_nat(8u);
v___x_1976_ = lean_unsigned_to_nat(517u);
v___x_1977_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__0));
v___x_1978_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11));
v___x_1979_ = l_mkPanicMessageWithDecl(v___x_1978_, v___x_1977_, v___x_1976_, v___x_1975_, v___x_1974_);
return v___x_1979_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7(void){
_start:
{
lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; 
v___x_1989_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__6));
v___x_1990_ = lean_unsigned_to_nat(14u);
v___x_1991_ = lean_unsigned_to_nat(22u);
v___x_1992_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__5));
v___x_1993_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__4));
v___x_1994_ = l_mkPanicMessageWithDecl(v___x_1993_, v___x_1992_, v___x_1991_, v___x_1990_, v___x_1989_);
return v___x_1994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat(lean_object* v_u_1995_, lean_object* v_00_u03b1_1996_, lean_object* v_q_1997_, lean_object* v_e_1998_, lean_object* v_hyp_1999_){
_start:
{
lean_object* v_den_2003_; lean_object* v___x_2004_; uint8_t v___x_2005_; 
v_den_2003_ = lean_ctor_get(v_q_1997_, 1);
v___x_2004_ = lean_unsigned_to_nat(1u);
v___x_2005_ = lean_nat_dec_eq(v_den_2003_, v___x_2004_);
if (v___x_2005_ == 0)
{
if (lean_obj_tag(v_e_1998_) == 5)
{
lean_object* v_fn_2006_; 
v_fn_2006_ = lean_ctor_get(v_e_1998_, 0);
lean_inc_ref(v_fn_2006_);
if (lean_obj_tag(v_fn_2006_) == 5)
{
lean_object* v_fn_2007_; 
v_fn_2007_ = lean_ctor_get(v_fn_2006_, 0);
lean_inc_ref(v_fn_2007_);
if (lean_obj_tag(v_fn_2007_) == 5)
{
lean_object* v_arg_2008_; lean_object* v_arg_2009_; lean_object* v_arg_2010_; lean_object* v___y_2012_; 
v_arg_2008_ = lean_ctor_get(v_e_1998_, 1);
lean_inc_ref(v_arg_2008_);
lean_dec_ref_known(v_e_1998_, 2);
v_arg_2009_ = lean_ctor_get(v_fn_2006_, 1);
lean_inc_ref(v_arg_2009_);
lean_dec_ref_known(v_fn_2006_, 2);
v_arg_2010_ = lean_ctor_get(v_fn_2007_, 1);
lean_inc_ref(v_arg_2010_);
lean_dec_ref_known(v_fn_2007_, 2);
if (lean_obj_tag(v_hyp_1999_) == 0)
{
lean_object* v___x_2023_; lean_object* v___x_2024_; 
v___x_2023_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7);
v___x_2024_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNNRat_spec__0(v___x_2023_);
v___y_2012_ = v___x_2024_;
goto v___jp_2011_;
}
else
{
lean_object* v_val_2025_; 
v_val_2025_ = lean_ctor_get(v_hyp_1999_, 0);
lean_inc(v_val_2025_);
lean_dec_ref_known(v_hyp_1999_, 1);
v___y_2012_ = v_val_2025_;
goto v___jp_2011_;
}
v___jp_2011_:
{
lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; 
v___x_2013_ = lean_box(0);
v___x_2014_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2014_, 0, v_u_1995_);
lean_ctor_set(v___x_2014_, 1, v___x_2013_);
v___x_2015_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__3));
v___x_2016_ = l_Lean_Expr_const___override(v___x_2015_, v___x_2014_);
v___x_2017_ = l_Lean_Expr_app___override(v___x_2016_, v_00_u03b1_1996_);
lean_inc_ref(v_arg_2010_);
v___x_2018_ = l_Lean_Expr_app___override(v___x_2017_, v_arg_2010_);
lean_inc_ref(v_arg_2009_);
v___x_2019_ = l_Lean_Expr_app___override(v___x_2018_, v_arg_2009_);
lean_inc_ref(v_arg_2008_);
v___x_2020_ = l_Lean_Expr_app___override(v___x_2019_, v_arg_2008_);
v___x_2021_ = l_Lean_Expr_app___override(v___x_2020_, v___y_2012_);
v___x_2022_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v___x_2022_, 0, v_arg_2010_);
lean_ctor_set(v___x_2022_, 1, v_q_1997_);
lean_ctor_set(v___x_2022_, 2, v_arg_2009_);
lean_ctor_set(v___x_2022_, 3, v_arg_2008_);
lean_ctor_set(v___x_2022_, 4, v___x_2021_);
return v___x_2022_;
}
}
else
{
lean_dec_ref(v_fn_2007_);
lean_dec_ref_known(v_fn_2006_, 2);
lean_dec_ref_known(v_e_1998_, 2);
lean_dec(v_hyp_1999_);
lean_dec_ref(v_q_1997_);
lean_dec_ref(v_00_u03b1_1996_);
lean_dec(v_u_1995_);
goto v___jp_2000_;
}
}
else
{
lean_dec_ref_known(v_e_1998_, 2);
lean_dec_ref(v_fn_2006_);
lean_dec(v_hyp_1999_);
lean_dec_ref(v_q_1997_);
lean_dec_ref(v_00_u03b1_1996_);
lean_dec(v_u_1995_);
goto v___jp_2000_;
}
}
else
{
lean_dec(v_hyp_1999_);
lean_dec_ref(v_e_1998_);
lean_dec_ref(v_q_1997_);
lean_dec_ref(v_00_u03b1_1996_);
lean_dec(v_u_1995_);
goto v___jp_2000_;
}
}
else
{
lean_object* v___x_2026_; 
lean_dec(v_hyp_1999_);
lean_dec_ref(v_q_1997_);
v___x_2026_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNat(v_u_1995_, v_00_u03b1_1996_, v_e_1998_);
return v___x_2026_;
}
v___jp_2000_:
{
lean_object* v___x_2001_; lean_object* v___x_2002_; 
v___x_2001_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__2);
v___x_2002_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(v___x_2001_);
return v___x_2002_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2(void){
_start:
{
lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; 
v___x_2029_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__1));
v___x_2030_ = lean_unsigned_to_nat(8u);
v___x_2031_ = lean_unsigned_to_nat(530u);
v___x_2032_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__0));
v___x_2033_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__11));
v___x_2034_ = l_mkPanicMessageWithDecl(v___x_2033_, v___x_2032_, v___x_2031_, v___x_2030_, v___x_2029_);
return v___x_2034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat(lean_object* v_u_2041_, lean_object* v_00_u03b1_2042_, lean_object* v_q_2043_, lean_object* v_e_2044_, lean_object* v_hyp_2045_){
_start:
{
lean_object* v_num_2049_; lean_object* v_den_2050_; lean_object* v___x_2051_; uint8_t v___x_2052_; 
v_num_2049_ = lean_ctor_get(v_q_2043_, 0);
v_den_2050_ = lean_ctor_get(v_q_2043_, 1);
v___x_2051_ = lean_unsigned_to_nat(1u);
v___x_2052_ = lean_nat_dec_eq(v_den_2050_, v___x_2051_);
if (v___x_2052_ == 0)
{
lean_object* v___x_2053_; uint8_t v___x_2054_; 
v___x_2053_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0, &lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__0);
lean_inc_ref(v_q_2043_);
v___x_2054_ = l_Rat_instDecidableLe(v___x_2053_, v_q_2043_);
if (v___x_2054_ == 0)
{
if (lean_obj_tag(v_e_2044_) == 5)
{
lean_object* v_fn_2055_; 
v_fn_2055_ = lean_ctor_get(v_e_2044_, 0);
if (lean_obj_tag(v_fn_2055_) == 5)
{
lean_object* v_fn_2056_; 
v_fn_2056_ = lean_ctor_get(v_fn_2055_, 0);
lean_inc_ref(v_fn_2056_);
if (lean_obj_tag(v_fn_2056_) == 5)
{
lean_object* v_arg_2057_; 
v_arg_2057_ = lean_ctor_get(v_fn_2055_, 1);
lean_inc_ref(v_arg_2057_);
if (lean_obj_tag(v_arg_2057_) == 5)
{
lean_object* v_arg_2058_; lean_object* v_arg_2059_; lean_object* v_arg_2060_; lean_object* v___y_2062_; 
v_arg_2058_ = lean_ctor_get(v_e_2044_, 1);
lean_inc_ref(v_arg_2058_);
lean_dec_ref_known(v_e_2044_, 2);
v_arg_2059_ = lean_ctor_get(v_fn_2056_, 1);
lean_inc_ref(v_arg_2059_);
lean_dec_ref_known(v_fn_2056_, 2);
v_arg_2060_ = lean_ctor_get(v_arg_2057_, 1);
lean_inc_ref(v_arg_2060_);
lean_dec_ref_known(v_arg_2057_, 2);
if (lean_obj_tag(v_hyp_2045_) == 0)
{
lean_object* v___x_2075_; lean_object* v___x_2076_; 
v___x_2075_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat___closed__7);
v___x_2076_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNNRat_spec__0(v___x_2075_);
v___y_2062_ = v___x_2076_;
goto v___jp_2061_;
}
else
{
lean_object* v_val_2077_; 
v_val_2077_ = lean_ctor_get(v_hyp_2045_, 0);
lean_inc(v_val_2077_);
lean_dec_ref_known(v_hyp_2045_, 1);
v___y_2062_ = v_val_2077_;
goto v___jp_2061_;
}
v___jp_2061_:
{
lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; 
v___x_2063_ = lean_box(0);
v___x_2064_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2064_, 0, v_u_2041_);
lean_ctor_set(v___x_2064_, 1, v___x_2063_);
v___x_2065_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
lean_inc_ref(v_arg_2060_);
v___x_2066_ = l_Lean_Expr_app___override(v___x_2065_, v_arg_2060_);
v___x_2067_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__3));
v___x_2068_ = l_Lean_Expr_const___override(v___x_2067_, v___x_2064_);
v___x_2069_ = l_Lean_Expr_app___override(v___x_2068_, v_00_u03b1_2042_);
lean_inc_ref(v_arg_2059_);
v___x_2070_ = l_Lean_Expr_app___override(v___x_2069_, v_arg_2059_);
v___x_2071_ = l_Lean_Expr_app___override(v___x_2070_, v___x_2066_);
lean_inc_ref(v_arg_2058_);
v___x_2072_ = l_Lean_Expr_app___override(v___x_2071_, v_arg_2058_);
v___x_2073_ = l_Lean_Expr_app___override(v___x_2072_, v___y_2062_);
v___x_2074_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v___x_2074_, 0, v_arg_2059_);
lean_ctor_set(v___x_2074_, 1, v_q_2043_);
lean_ctor_set(v___x_2074_, 2, v_arg_2060_);
lean_ctor_set(v___x_2074_, 3, v_arg_2058_);
lean_ctor_set(v___x_2074_, 4, v___x_2073_);
return v___x_2074_;
}
}
else
{
lean_dec_ref(v_arg_2057_);
lean_dec_ref_known(v_fn_2056_, 2);
lean_dec_ref_known(v_e_2044_, 2);
lean_dec(v_hyp_2045_);
lean_dec_ref(v_q_2043_);
lean_dec_ref(v_00_u03b1_2042_);
lean_dec(v_u_2041_);
goto v___jp_2046_;
}
}
else
{
lean_dec_ref(v_fn_2056_);
lean_dec_ref_known(v_e_2044_, 2);
lean_dec(v_hyp_2045_);
lean_dec_ref(v_q_2043_);
lean_dec_ref(v_00_u03b1_2042_);
lean_dec(v_u_2041_);
goto v___jp_2046_;
}
}
else
{
lean_dec_ref_known(v_e_2044_, 2);
lean_dec(v_hyp_2045_);
lean_dec_ref(v_q_2043_);
lean_dec_ref(v_00_u03b1_2042_);
lean_dec(v_u_2041_);
goto v___jp_2046_;
}
}
else
{
lean_dec(v_hyp_2045_);
lean_dec_ref(v_e_2044_);
lean_dec_ref(v_q_2043_);
lean_dec_ref(v_00_u03b1_2042_);
lean_dec(v_u_2041_);
goto v___jp_2046_;
}
}
else
{
lean_object* v___x_2078_; 
v___x_2078_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawNNRat(v_u_2041_, v_00_u03b1_2042_, v_q_2043_, v_e_2044_, v_hyp_2045_);
return v___x_2078_;
}
}
else
{
lean_object* v___x_2079_; 
lean_inc(v_num_2049_);
lean_dec(v_hyp_2045_);
lean_dec_ref(v_q_2043_);
v___x_2079_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawInt(v_u_2041_, v_00_u03b1_2042_, v_num_2049_, v_e_2044_);
lean_dec(v_num_2049_);
return v___x_2079_;
}
v___jp_2046_:
{
lean_object* v___x_2047_; lean_object* v___x_2048_; 
v___x_2047_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_ofRawRat___closed__2);
v___x_2048_ = lp_mathlib_panic___at___00Mathlib_Meta_NormNum_Result_ofRawNat_spec__0___redArg(v___x_2047_);
return v___x_2048_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(lean_object* v_u_2179_, lean_object* v_00_u03b1_2180_, lean_object* v_e_2181_, lean_object* v_x_2182_, lean_object* v_a_2183_, lean_object* v_a_2184_, lean_object* v_a_2185_, lean_object* v_a_2186_){
_start:
{
switch(lean_obj_tag(v_x_2182_))
{
case 0:
{
lean_object* v___x_2188_; lean_object* v_fst_2189_; lean_object* v_snd_2190_; lean_object* v___x_2191_; uint8_t v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; 
v___x_2188_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRawEq(v_u_2179_, v_00_u03b1_2180_, v_e_2181_, v_x_2182_);
v_fst_2189_ = lean_ctor_get(v___x_2188_, 0);
lean_inc(v_fst_2189_);
v_snd_2190_ = lean_ctor_get(v___x_2188_, 1);
lean_inc(v_snd_2190_);
lean_dec_ref(v___x_2188_);
v___x_2191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2191_, 0, v_snd_2190_);
v___x_2192_ = 1;
v___x_2193_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2193_, 0, v_fst_2189_);
lean_ctor_set(v___x_2193_, 1, v___x_2191_);
lean_ctor_set_uint8(v___x_2193_, sizeof(void*)*2, v___x_2192_);
v___x_2194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2193_);
return v___x_2194_;
}
case 1:
{
lean_object* v_inst_2195_; lean_object* v_lit_2196_; lean_object* v_proof_2197_; lean_object* v___x_2198_; 
v_inst_2195_ = lean_ctor_get(v_x_2182_, 0);
lean_inc_ref_n(v_inst_2195_, 2);
v_lit_2196_ = lean_ctor_get(v_x_2182_, 1);
lean_inc_ref_n(v_lit_2196_, 2);
v_proof_2197_ = lean_ctor_get(v_x_2182_, 2);
lean_inc_ref(v_proof_2197_);
lean_dec_ref_known(v_x_2182_, 3);
lean_inc_ref(v_00_u03b1_2180_);
lean_inc(v_u_2179_);
v___x_2198_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v_inst_2195_, v_lit_2196_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2198_) == 0)
{
lean_object* v_a_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2228_; 
v_a_2199_ = lean_ctor_get(v___x_2198_, 0);
v_isSharedCheck_2228_ = !lean_is_exclusive(v___x_2198_);
if (v_isSharedCheck_2228_ == 0)
{
v___x_2201_ = v___x_2198_;
v_isShared_2202_ = v_isSharedCheck_2228_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_a_2199_);
lean_dec(v___x_2198_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2228_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v_fst_2203_; lean_object* v_snd_2204_; lean_object* v___x_2206_; uint8_t v_isShared_2207_; uint8_t v_isSharedCheck_2227_; 
v_fst_2203_ = lean_ctor_get(v_a_2199_, 0);
v_snd_2204_ = lean_ctor_get(v_a_2199_, 1);
v_isSharedCheck_2227_ = !lean_is_exclusive(v_a_2199_);
if (v_isSharedCheck_2227_ == 0)
{
v___x_2206_ = v_a_2199_;
v_isShared_2207_ = v_isSharedCheck_2227_;
goto v_resetjp_2205_;
}
else
{
lean_inc(v_snd_2204_);
lean_inc(v_fst_2203_);
lean_dec(v_a_2199_);
v___x_2206_ = lean_box(0);
v_isShared_2207_ = v_isSharedCheck_2227_;
goto v_resetjp_2205_;
}
v_resetjp_2205_:
{
lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2211_; 
v___x_2208_ = lean_box(0);
v___x_2209_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__1));
if (v_isShared_2207_ == 0)
{
lean_ctor_set_tag(v___x_2206_, 1);
lean_ctor_set(v___x_2206_, 1, v___x_2208_);
lean_ctor_set(v___x_2206_, 0, v_u_2179_);
v___x_2211_ = v___x_2206_;
goto v_reusejp_2210_;
}
else
{
lean_object* v_reuseFailAlloc_2226_; 
v_reuseFailAlloc_2226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2226_, 0, v_u_2179_);
lean_ctor_set(v_reuseFailAlloc_2226_, 1, v___x_2208_);
v___x_2211_ = v_reuseFailAlloc_2226_;
goto v_reusejp_2210_;
}
v_reusejp_2210_:
{
lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; uint8_t v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2224_; 
v___x_2212_ = l_Lean_Expr_const___override(v___x_2209_, v___x_2211_);
v___x_2213_ = l_Lean_Expr_app___override(v___x_2212_, v_00_u03b1_2180_);
v___x_2214_ = l_Lean_Expr_app___override(v___x_2213_, v_inst_2195_);
v___x_2215_ = l_Lean_Expr_app___override(v___x_2214_, v_lit_2196_);
v___x_2216_ = l_Lean_Expr_app___override(v___x_2215_, v_e_2181_);
lean_inc(v_fst_2203_);
v___x_2217_ = l_Lean_Expr_app___override(v___x_2216_, v_fst_2203_);
v___x_2218_ = l_Lean_Expr_app___override(v___x_2217_, v_proof_2197_);
v___x_2219_ = l_Lean_Expr_app___override(v___x_2218_, v_snd_2204_);
v___x_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2220_, 0, v___x_2219_);
v___x_2221_ = 1;
v___x_2222_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2222_, 0, v_fst_2203_);
lean_ctor_set(v___x_2222_, 1, v___x_2220_);
lean_ctor_set_uint8(v___x_2222_, sizeof(void*)*2, v___x_2221_);
if (v_isShared_2202_ == 0)
{
lean_ctor_set(v___x_2201_, 0, v___x_2222_);
v___x_2224_ = v___x_2201_;
goto v_reusejp_2223_;
}
else
{
lean_object* v_reuseFailAlloc_2225_; 
v_reuseFailAlloc_2225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2225_, 0, v___x_2222_);
v___x_2224_ = v_reuseFailAlloc_2225_;
goto v_reusejp_2223_;
}
v_reusejp_2223_:
{
return v___x_2224_;
}
}
}
}
}
else
{
lean_object* v_a_2229_; lean_object* v___x_2231_; uint8_t v_isShared_2232_; uint8_t v_isSharedCheck_2236_; 
lean_dec_ref(v_proof_2197_);
lean_dec_ref(v_lit_2196_);
lean_dec_ref(v_inst_2195_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
lean_dec(v_u_2179_);
v_a_2229_ = lean_ctor_get(v___x_2198_, 0);
v_isSharedCheck_2236_ = !lean_is_exclusive(v___x_2198_);
if (v_isSharedCheck_2236_ == 0)
{
v___x_2231_ = v___x_2198_;
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
else
{
lean_inc(v_a_2229_);
lean_dec(v___x_2198_);
v___x_2231_ = lean_box(0);
v_isShared_2232_ = v_isSharedCheck_2236_;
goto v_resetjp_2230_;
}
v_resetjp_2230_:
{
lean_object* v___x_2234_; 
if (v_isShared_2232_ == 0)
{
v___x_2234_ = v___x_2231_;
goto v_reusejp_2233_;
}
else
{
lean_object* v_reuseFailAlloc_2235_; 
v_reuseFailAlloc_2235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2235_, 0, v_a_2229_);
v___x_2234_ = v_reuseFailAlloc_2235_;
goto v_reusejp_2233_;
}
v_reusejp_2233_:
{
return v___x_2234_;
}
}
}
}
case 2:
{
lean_object* v_inst_2237_; lean_object* v_lit_2238_; lean_object* v_proof_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; 
v_inst_2237_ = lean_ctor_get(v_x_2182_, 0);
lean_inc_ref_n(v_inst_2237_, 2);
v_lit_2238_ = lean_ctor_get(v_x_2182_, 1);
lean_inc_ref_n(v_lit_2238_, 2);
v_proof_2239_ = lean_ctor_get(v_x_2182_, 2);
lean_inc_ref(v_proof_2239_);
lean_dec_ref_known(v_x_2182_, 3);
v___x_2240_ = lean_box(0);
lean_inc(v_u_2179_);
v___x_2241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2241_, 0, v_u_2179_);
lean_ctor_set(v___x_2241_, 1, v___x_2240_);
v___x_2242_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4));
lean_inc_ref_n(v___x_2241_, 4);
v___x_2243_ = l_Lean_Expr_const___override(v___x_2242_, v___x_2241_);
lean_inc_ref_n(v_00_u03b1_2180_, 5);
v___x_2244_ = l_Lean_Expr_app___override(v___x_2243_, v_00_u03b1_2180_);
v___x_2245_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7));
v___x_2246_ = l_Lean_Expr_const___override(v___x_2245_, v___x_2241_);
v___x_2247_ = l_Lean_Expr_app___override(v___x_2246_, v_00_u03b1_2180_);
v___x_2248_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9));
v___x_2249_ = l_Lean_Expr_const___override(v___x_2248_, v___x_2241_);
v___x_2250_ = l_Lean_Expr_app___override(v___x_2249_, v_00_u03b1_2180_);
v___x_2251_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2));
v___x_2252_ = l_Lean_Expr_const___override(v___x_2251_, v___x_2241_);
v___x_2253_ = l_Lean_Expr_app___override(v___x_2252_, v_00_u03b1_2180_);
v___x_2254_ = l_Lean_Expr_app___override(v___x_2253_, v_inst_2237_);
v___x_2255_ = l_Lean_Expr_app___override(v___x_2250_, v___x_2254_);
v___x_2256_ = l_Lean_Expr_app___override(v___x_2247_, v___x_2255_);
v___x_2257_ = l_Lean_Expr_app___override(v___x_2244_, v___x_2256_);
v___x_2258_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v___x_2257_, v_lit_2238_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_object* v_a_2259_; lean_object* v___x_2261_; uint8_t v_isShared_2262_; uint8_t v_isSharedCheck_2309_; 
v_a_2259_ = lean_ctor_get(v___x_2258_, 0);
v_isSharedCheck_2309_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2309_ == 0)
{
v___x_2261_ = v___x_2258_;
v_isShared_2262_ = v_isSharedCheck_2309_;
goto v_resetjp_2260_;
}
else
{
lean_inc(v_a_2259_);
lean_dec(v___x_2258_);
v___x_2261_ = lean_box(0);
v_isShared_2262_ = v_isSharedCheck_2309_;
goto v_resetjp_2260_;
}
v_resetjp_2260_:
{
lean_object* v_fst_2263_; lean_object* v_snd_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; uint8_t v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2307_; 
v_fst_2263_ = lean_ctor_get(v_a_2259_, 0);
lean_inc_n(v_fst_2263_, 2);
v_snd_2264_ = lean_ctor_get(v_a_2259_, 1);
lean_inc(v_snd_2264_);
lean_dec(v_a_2259_);
v___x_2265_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12));
lean_inc_ref_n(v___x_2241_, 7);
v___x_2266_ = l_Lean_Expr_const___override(v___x_2265_, v___x_2241_);
lean_inc_ref_n(v_00_u03b1_2180_, 7);
v___x_2267_ = l_Lean_Expr_app___override(v___x_2266_, v_00_u03b1_2180_);
v___x_2268_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15));
v___x_2269_ = l_Lean_Expr_const___override(v___x_2268_, v___x_2241_);
v___x_2270_ = l_Lean_Expr_app___override(v___x_2269_, v_00_u03b1_2180_);
v___x_2271_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18));
v___x_2272_ = l_Lean_Expr_const___override(v___x_2271_, v___x_2241_);
v___x_2273_ = l_Lean_Expr_app___override(v___x_2272_, v_00_u03b1_2180_);
v___x_2274_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21));
v___x_2275_ = l_Lean_Expr_const___override(v___x_2274_, v___x_2241_);
v___x_2276_ = l_Lean_Expr_app___override(v___x_2275_, v_00_u03b1_2180_);
v___x_2277_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24));
v___x_2278_ = l_Lean_Expr_const___override(v___x_2277_, v___x_2241_);
v___x_2279_ = l_Lean_Expr_app___override(v___x_2278_, v_00_u03b1_2180_);
v___x_2280_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27));
v___x_2281_ = l_Lean_Expr_const___override(v___x_2280_, v___x_2241_);
v___x_2282_ = l_Lean_Expr_app___override(v___x_2281_, v_00_u03b1_2180_);
v___x_2283_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29));
v___x_2284_ = l_Lean_Expr_const___override(v___x_2283_, v___x_2241_);
v___x_2285_ = l_Lean_Expr_app___override(v___x_2284_, v_00_u03b1_2180_);
lean_inc_ref(v_inst_2237_);
v___x_2286_ = l_Lean_Expr_app___override(v___x_2285_, v_inst_2237_);
v___x_2287_ = l_Lean_Expr_app___override(v___x_2282_, v___x_2286_);
v___x_2288_ = l_Lean_Expr_app___override(v___x_2279_, v___x_2287_);
v___x_2289_ = l_Lean_Expr_app___override(v___x_2276_, v___x_2288_);
v___x_2290_ = l_Lean_Expr_app___override(v___x_2273_, v___x_2289_);
v___x_2291_ = l_Lean_Expr_app___override(v___x_2270_, v___x_2290_);
v___x_2292_ = l_Lean_Expr_app___override(v___x_2267_, v___x_2291_);
v___x_2293_ = l_Lean_Expr_app___override(v___x_2292_, v_fst_2263_);
v___x_2294_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__31));
v___x_2295_ = l_Lean_Expr_const___override(v___x_2294_, v___x_2241_);
v___x_2296_ = l_Lean_Expr_app___override(v___x_2295_, v_00_u03b1_2180_);
v___x_2297_ = l_Lean_Expr_app___override(v___x_2296_, v_inst_2237_);
v___x_2298_ = l_Lean_Expr_app___override(v___x_2297_, v_lit_2238_);
v___x_2299_ = l_Lean_Expr_app___override(v___x_2298_, v_e_2181_);
v___x_2300_ = l_Lean_Expr_app___override(v___x_2299_, v_fst_2263_);
v___x_2301_ = l_Lean_Expr_app___override(v___x_2300_, v_proof_2239_);
v___x_2302_ = l_Lean_Expr_app___override(v___x_2301_, v_snd_2264_);
v___x_2303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2302_);
v___x_2304_ = 1;
v___x_2305_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2305_, 0, v___x_2293_);
lean_ctor_set(v___x_2305_, 1, v___x_2303_);
lean_ctor_set_uint8(v___x_2305_, sizeof(void*)*2, v___x_2304_);
if (v_isShared_2262_ == 0)
{
lean_ctor_set(v___x_2261_, 0, v___x_2305_);
v___x_2307_ = v___x_2261_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2308_; 
v_reuseFailAlloc_2308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2308_, 0, v___x_2305_);
v___x_2307_ = v_reuseFailAlloc_2308_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
return v___x_2307_;
}
}
}
else
{
lean_object* v_a_2310_; lean_object* v___x_2312_; uint8_t v_isShared_2313_; uint8_t v_isSharedCheck_2317_; 
lean_dec_ref_known(v___x_2241_, 2);
lean_dec_ref(v_proof_2239_);
lean_dec_ref(v_lit_2238_);
lean_dec_ref(v_inst_2237_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
v_a_2310_ = lean_ctor_get(v___x_2258_, 0);
v_isSharedCheck_2317_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2317_ == 0)
{
v___x_2312_ = v___x_2258_;
v_isShared_2313_ = v_isSharedCheck_2317_;
goto v_resetjp_2311_;
}
else
{
lean_inc(v_a_2310_);
lean_dec(v___x_2258_);
v___x_2312_ = lean_box(0);
v_isShared_2313_ = v_isSharedCheck_2317_;
goto v_resetjp_2311_;
}
v_resetjp_2311_:
{
lean_object* v___x_2315_; 
if (v_isShared_2313_ == 0)
{
v___x_2315_ = v___x_2312_;
goto v_reusejp_2314_;
}
else
{
lean_object* v_reuseFailAlloc_2316_; 
v_reuseFailAlloc_2316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2316_, 0, v_a_2310_);
v___x_2315_ = v_reuseFailAlloc_2316_;
goto v_reusejp_2314_;
}
v_reusejp_2314_:
{
return v___x_2315_;
}
}
}
}
case 3:
{
lean_object* v_inst_2318_; lean_object* v_n_2319_; lean_object* v_d_2320_; lean_object* v_proof_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; 
v_inst_2318_ = lean_ctor_get(v_x_2182_, 0);
lean_inc_ref_n(v_inst_2318_, 2);
v_n_2319_ = lean_ctor_get(v_x_2182_, 2);
lean_inc_ref_n(v_n_2319_, 2);
v_d_2320_ = lean_ctor_get(v_x_2182_, 3);
lean_inc_ref(v_d_2320_);
v_proof_2321_ = lean_ctor_get(v_x_2182_, 4);
lean_inc_ref(v_proof_2321_);
lean_dec_ref_known(v_x_2182_, 5);
v___x_2322_ = lean_box(0);
lean_inc_n(v_u_2179_, 2);
v___x_2323_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2323_, 0, v_u_2179_);
lean_ctor_set(v___x_2323_, 1, v___x_2322_);
v___x_2324_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4));
lean_inc_ref_n(v___x_2323_, 4);
v___x_2325_ = l_Lean_Expr_const___override(v___x_2324_, v___x_2323_);
lean_inc_ref_n(v_00_u03b1_2180_, 5);
v___x_2326_ = l_Lean_Expr_app___override(v___x_2325_, v_00_u03b1_2180_);
v___x_2327_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7));
v___x_2328_ = l_Lean_Expr_const___override(v___x_2327_, v___x_2323_);
v___x_2329_ = l_Lean_Expr_app___override(v___x_2328_, v_00_u03b1_2180_);
v___x_2330_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9));
v___x_2331_ = l_Lean_Expr_const___override(v___x_2330_, v___x_2323_);
v___x_2332_ = l_Lean_Expr_app___override(v___x_2331_, v_00_u03b1_2180_);
v___x_2333_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4));
v___x_2334_ = l_Lean_Expr_const___override(v___x_2333_, v___x_2323_);
v___x_2335_ = l_Lean_Expr_app___override(v___x_2334_, v_00_u03b1_2180_);
v___x_2336_ = l_Lean_Expr_app___override(v___x_2335_, v_inst_2318_);
v___x_2337_ = l_Lean_Expr_app___override(v___x_2332_, v___x_2336_);
v___x_2338_ = l_Lean_Expr_app___override(v___x_2329_, v___x_2337_);
v___x_2339_ = l_Lean_Expr_app___override(v___x_2326_, v___x_2338_);
lean_inc_ref(v___x_2339_);
v___x_2340_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v___x_2339_, v_n_2319_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2340_) == 0)
{
lean_object* v_a_2341_; lean_object* v_fst_2342_; lean_object* v_snd_2343_; lean_object* v___x_2345_; uint8_t v_isShared_2346_; uint8_t v_isSharedCheck_2415_; 
v_a_2341_ = lean_ctor_get(v___x_2340_, 0);
lean_inc(v_a_2341_);
lean_dec_ref_known(v___x_2340_, 1);
v_fst_2342_ = lean_ctor_get(v_a_2341_, 0);
v_snd_2343_ = lean_ctor_get(v_a_2341_, 1);
v_isSharedCheck_2415_ = !lean_is_exclusive(v_a_2341_);
if (v_isSharedCheck_2415_ == 0)
{
v___x_2345_ = v_a_2341_;
v_isShared_2346_ = v_isSharedCheck_2415_;
goto v_resetjp_2344_;
}
else
{
lean_inc(v_snd_2343_);
lean_inc(v_fst_2342_);
lean_dec(v_a_2341_);
v___x_2345_ = lean_box(0);
v_isShared_2346_ = v_isSharedCheck_2415_;
goto v_resetjp_2344_;
}
v_resetjp_2344_:
{
lean_object* v___x_2347_; 
lean_inc_ref(v_d_2320_);
lean_inc_ref(v_00_u03b1_2180_);
lean_inc(v_u_2179_);
v___x_2347_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v___x_2339_, v_d_2320_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2347_) == 0)
{
lean_object* v_a_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2406_; 
v_a_2348_ = lean_ctor_get(v___x_2347_, 0);
v_isSharedCheck_2406_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2406_ == 0)
{
v___x_2350_ = v___x_2347_;
v_isShared_2351_ = v_isSharedCheck_2406_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_a_2348_);
lean_dec(v___x_2347_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2406_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v_fst_2352_; lean_object* v_snd_2353_; lean_object* v___x_2355_; uint8_t v_isShared_2356_; uint8_t v_isSharedCheck_2405_; 
v_fst_2352_ = lean_ctor_get(v_a_2348_, 0);
v_snd_2353_ = lean_ctor_get(v_a_2348_, 1);
v_isSharedCheck_2405_ = !lean_is_exclusive(v_a_2348_);
if (v_isSharedCheck_2405_ == 0)
{
v___x_2355_ = v_a_2348_;
v_isShared_2356_ = v_isSharedCheck_2405_;
goto v_resetjp_2354_;
}
else
{
lean_inc(v_snd_2353_);
lean_inc(v_fst_2352_);
lean_dec(v_a_2348_);
v___x_2355_ = lean_box(0);
v_isShared_2356_ = v_isSharedCheck_2405_;
goto v_resetjp_2354_;
}
v_resetjp_2354_:
{
lean_object* v___x_2357_; lean_object* v___x_2359_; 
v___x_2357_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34));
lean_inc_ref(v___x_2323_);
lean_inc(v_u_2179_);
if (v_isShared_2356_ == 0)
{
lean_ctor_set_tag(v___x_2355_, 1);
lean_ctor_set(v___x_2355_, 1, v___x_2323_);
lean_ctor_set(v___x_2355_, 0, v_u_2179_);
v___x_2359_ = v___x_2355_;
goto v_reusejp_2358_;
}
else
{
lean_object* v_reuseFailAlloc_2404_; 
v_reuseFailAlloc_2404_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2404_, 0, v_u_2179_);
lean_ctor_set(v_reuseFailAlloc_2404_, 1, v___x_2323_);
v___x_2359_ = v_reuseFailAlloc_2404_;
goto v_reusejp_2358_;
}
v_reusejp_2358_:
{
lean_object* v___x_2361_; 
if (v_isShared_2346_ == 0)
{
lean_ctor_set_tag(v___x_2345_, 1);
lean_ctor_set(v___x_2345_, 1, v___x_2359_);
lean_ctor_set(v___x_2345_, 0, v_u_2179_);
v___x_2361_ = v___x_2345_;
goto v_reusejp_2360_;
}
else
{
lean_object* v_reuseFailAlloc_2403_; 
v_reuseFailAlloc_2403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2403_, 0, v_u_2179_);
lean_ctor_set(v_reuseFailAlloc_2403_, 1, v___x_2359_);
v___x_2361_ = v_reuseFailAlloc_2403_;
goto v_reusejp_2360_;
}
v_reusejp_2360_:
{
lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; uint8_t v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2401_; 
v___x_2362_ = l_Lean_Expr_const___override(v___x_2357_, v___x_2361_);
lean_inc_ref_n(v_00_u03b1_2180_, 7);
v___x_2363_ = l_Lean_Expr_app___override(v___x_2362_, v_00_u03b1_2180_);
v___x_2364_ = l_Lean_Expr_app___override(v___x_2363_, v_00_u03b1_2180_);
v___x_2365_ = l_Lean_Expr_app___override(v___x_2364_, v_00_u03b1_2180_);
v___x_2366_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__36));
lean_inc_ref_n(v___x_2323_, 4);
v___x_2367_ = l_Lean_Expr_const___override(v___x_2366_, v___x_2323_);
v___x_2368_ = l_Lean_Expr_app___override(v___x_2367_, v_00_u03b1_2180_);
v___x_2369_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39));
v___x_2370_ = l_Lean_Expr_const___override(v___x_2369_, v___x_2323_);
v___x_2371_ = l_Lean_Expr_app___override(v___x_2370_, v_00_u03b1_2180_);
v___x_2372_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__42));
v___x_2373_ = l_Lean_Expr_const___override(v___x_2372_, v___x_2323_);
v___x_2374_ = l_Lean_Expr_app___override(v___x_2373_, v_00_u03b1_2180_);
v___x_2375_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__44));
v___x_2376_ = l_Lean_Expr_const___override(v___x_2375_, v___x_2323_);
v___x_2377_ = l_Lean_Expr_app___override(v___x_2376_, v_00_u03b1_2180_);
lean_inc_ref(v_inst_2318_);
v___x_2378_ = l_Lean_Expr_app___override(v___x_2377_, v_inst_2318_);
v___x_2379_ = l_Lean_Expr_app___override(v___x_2374_, v___x_2378_);
v___x_2380_ = l_Lean_Expr_app___override(v___x_2371_, v___x_2379_);
v___x_2381_ = l_Lean_Expr_app___override(v___x_2368_, v___x_2380_);
v___x_2382_ = l_Lean_Expr_app___override(v___x_2365_, v___x_2381_);
lean_inc(v_fst_2342_);
v___x_2383_ = l_Lean_Expr_app___override(v___x_2382_, v_fst_2342_);
lean_inc(v_fst_2352_);
v___x_2384_ = l_Lean_Expr_app___override(v___x_2383_, v_fst_2352_);
v___x_2385_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__45));
v___x_2386_ = l_Lean_Expr_const___override(v___x_2385_, v___x_2323_);
v___x_2387_ = l_Lean_Expr_app___override(v___x_2386_, v_00_u03b1_2180_);
v___x_2388_ = l_Lean_Expr_app___override(v___x_2387_, v_inst_2318_);
v___x_2389_ = l_Lean_Expr_app___override(v___x_2388_, v_n_2319_);
v___x_2390_ = l_Lean_Expr_app___override(v___x_2389_, v_d_2320_);
v___x_2391_ = l_Lean_Expr_app___override(v___x_2390_, v_e_2181_);
v___x_2392_ = l_Lean_Expr_app___override(v___x_2391_, v_fst_2342_);
v___x_2393_ = l_Lean_Expr_app___override(v___x_2392_, v_fst_2352_);
v___x_2394_ = l_Lean_Expr_app___override(v___x_2393_, v_proof_2321_);
v___x_2395_ = l_Lean_Expr_app___override(v___x_2394_, v_snd_2343_);
v___x_2396_ = l_Lean_Expr_app___override(v___x_2395_, v_snd_2353_);
v___x_2397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2397_, 0, v___x_2396_);
v___x_2398_ = 1;
v___x_2399_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2399_, 0, v___x_2384_);
lean_ctor_set(v___x_2399_, 1, v___x_2397_);
lean_ctor_set_uint8(v___x_2399_, sizeof(void*)*2, v___x_2398_);
if (v_isShared_2351_ == 0)
{
lean_ctor_set(v___x_2350_, 0, v___x_2399_);
v___x_2401_ = v___x_2350_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v___x_2399_);
v___x_2401_ = v_reuseFailAlloc_2402_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
return v___x_2401_;
}
}
}
}
}
}
else
{
lean_object* v_a_2407_; lean_object* v___x_2409_; uint8_t v_isShared_2410_; uint8_t v_isSharedCheck_2414_; 
lean_del_object(v___x_2345_);
lean_dec(v_snd_2343_);
lean_dec(v_fst_2342_);
lean_dec_ref_known(v___x_2323_, 2);
lean_dec_ref(v_proof_2321_);
lean_dec_ref(v_d_2320_);
lean_dec_ref(v_n_2319_);
lean_dec_ref(v_inst_2318_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
lean_dec(v_u_2179_);
v_a_2407_ = lean_ctor_get(v___x_2347_, 0);
v_isSharedCheck_2414_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2414_ == 0)
{
v___x_2409_ = v___x_2347_;
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
else
{
lean_inc(v_a_2407_);
lean_dec(v___x_2347_);
v___x_2409_ = lean_box(0);
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
v_resetjp_2408_:
{
lean_object* v___x_2412_; 
if (v_isShared_2410_ == 0)
{
v___x_2412_ = v___x_2409_;
goto v_reusejp_2411_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v_a_2407_);
v___x_2412_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2411_;
}
v_reusejp_2411_:
{
return v___x_2412_;
}
}
}
}
}
else
{
lean_object* v_a_2416_; lean_object* v___x_2418_; uint8_t v_isShared_2419_; uint8_t v_isSharedCheck_2423_; 
lean_dec_ref(v___x_2339_);
lean_dec_ref_known(v___x_2323_, 2);
lean_dec_ref(v_proof_2321_);
lean_dec_ref(v_d_2320_);
lean_dec_ref(v_n_2319_);
lean_dec_ref(v_inst_2318_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
lean_dec(v_u_2179_);
v_a_2416_ = lean_ctor_get(v___x_2340_, 0);
v_isSharedCheck_2423_ = !lean_is_exclusive(v___x_2340_);
if (v_isSharedCheck_2423_ == 0)
{
v___x_2418_ = v___x_2340_;
v_isShared_2419_ = v_isSharedCheck_2423_;
goto v_resetjp_2417_;
}
else
{
lean_inc(v_a_2416_);
lean_dec(v___x_2340_);
v___x_2418_ = lean_box(0);
v_isShared_2419_ = v_isSharedCheck_2423_;
goto v_resetjp_2417_;
}
v_resetjp_2417_:
{
lean_object* v___x_2421_; 
if (v_isShared_2419_ == 0)
{
v___x_2421_ = v___x_2418_;
goto v_reusejp_2420_;
}
else
{
lean_object* v_reuseFailAlloc_2422_; 
v_reuseFailAlloc_2422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2422_, 0, v_a_2416_);
v___x_2421_ = v_reuseFailAlloc_2422_;
goto v_reusejp_2420_;
}
v_reusejp_2420_:
{
return v___x_2421_;
}
}
}
}
default: 
{
lean_object* v_inst_2424_; lean_object* v_n_2425_; lean_object* v_d_2426_; lean_object* v_proof_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; 
v_inst_2424_ = lean_ctor_get(v_x_2182_, 0);
lean_inc_ref_n(v_inst_2424_, 2);
v_n_2425_ = lean_ctor_get(v_x_2182_, 2);
lean_inc_ref_n(v_n_2425_, 2);
v_d_2426_ = lean_ctor_get(v_x_2182_, 3);
lean_inc_ref(v_d_2426_);
v_proof_2427_ = lean_ctor_get(v_x_2182_, 4);
lean_inc_ref(v_proof_2427_);
lean_dec_ref_known(v_x_2182_, 5);
v___x_2428_ = lean_box(0);
lean_inc_n(v_u_2179_, 2);
v___x_2429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2429_, 0, v_u_2179_);
lean_ctor_set(v___x_2429_, 1, v___x_2428_);
v___x_2430_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__4));
lean_inc_ref_n(v___x_2429_, 5);
v___x_2431_ = l_Lean_Expr_const___override(v___x_2430_, v___x_2429_);
lean_inc_ref_n(v_00_u03b1_2180_, 6);
v___x_2432_ = l_Lean_Expr_app___override(v___x_2431_, v_00_u03b1_2180_);
v___x_2433_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__7));
v___x_2434_ = l_Lean_Expr_const___override(v___x_2433_, v___x_2429_);
v___x_2435_ = l_Lean_Expr_app___override(v___x_2434_, v_00_u03b1_2180_);
v___x_2436_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__9));
v___x_2437_ = l_Lean_Expr_const___override(v___x_2436_, v___x_2429_);
v___x_2438_ = l_Lean_Expr_app___override(v___x_2437_, v_00_u03b1_2180_);
v___x_2439_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___closed__2));
v___x_2440_ = l_Lean_Expr_const___override(v___x_2439_, v___x_2429_);
v___x_2441_ = l_Lean_Expr_app___override(v___x_2440_, v_00_u03b1_2180_);
v___x_2442_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
v___x_2443_ = l_Lean_Expr_const___override(v___x_2442_, v___x_2429_);
v___x_2444_ = l_Lean_Expr_app___override(v___x_2443_, v_00_u03b1_2180_);
v___x_2445_ = l_Lean_Expr_app___override(v___x_2444_, v_inst_2424_);
lean_inc_ref(v___x_2445_);
v___x_2446_ = l_Lean_Expr_app___override(v___x_2441_, v___x_2445_);
v___x_2447_ = l_Lean_Expr_app___override(v___x_2438_, v___x_2446_);
v___x_2448_ = l_Lean_Expr_app___override(v___x_2435_, v___x_2447_);
v___x_2449_ = l_Lean_Expr_app___override(v___x_2432_, v___x_2448_);
lean_inc_ref(v___x_2449_);
v___x_2450_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v___x_2449_, v_n_2425_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2450_) == 0)
{
lean_object* v_a_2451_; lean_object* v_fst_2452_; lean_object* v_snd_2453_; lean_object* v___x_2455_; uint8_t v_isShared_2456_; uint8_t v_isSharedCheck_2550_; 
v_a_2451_ = lean_ctor_get(v___x_2450_, 0);
lean_inc(v_a_2451_);
lean_dec_ref_known(v___x_2450_, 1);
v_fst_2452_ = lean_ctor_get(v_a_2451_, 0);
v_snd_2453_ = lean_ctor_get(v_a_2451_, 1);
v_isSharedCheck_2550_ = !lean_is_exclusive(v_a_2451_);
if (v_isSharedCheck_2550_ == 0)
{
v___x_2455_ = v_a_2451_;
v_isShared_2456_ = v_isSharedCheck_2550_;
goto v_resetjp_2454_;
}
else
{
lean_inc(v_snd_2453_);
lean_inc(v_fst_2452_);
lean_dec(v_a_2451_);
v___x_2455_ = lean_box(0);
v_isShared_2456_ = v_isSharedCheck_2550_;
goto v_resetjp_2454_;
}
v_resetjp_2454_:
{
lean_object* v___x_2457_; 
lean_inc_ref(v_d_2426_);
lean_inc_ref(v_00_u03b1_2180_);
lean_inc(v_u_2179_);
v___x_2457_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_2179_, v_00_u03b1_2180_, v___x_2449_, v_d_2426_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
if (lean_obj_tag(v___x_2457_) == 0)
{
lean_object* v_a_2458_; lean_object* v___x_2460_; uint8_t v_isShared_2461_; uint8_t v_isSharedCheck_2541_; 
v_a_2458_ = lean_ctor_get(v___x_2457_, 0);
v_isSharedCheck_2541_ = !lean_is_exclusive(v___x_2457_);
if (v_isSharedCheck_2541_ == 0)
{
v___x_2460_ = v___x_2457_;
v_isShared_2461_ = v_isSharedCheck_2541_;
goto v_resetjp_2459_;
}
else
{
lean_inc(v_a_2458_);
lean_dec(v___x_2457_);
v___x_2460_ = lean_box(0);
v_isShared_2461_ = v_isSharedCheck_2541_;
goto v_resetjp_2459_;
}
v_resetjp_2459_:
{
lean_object* v_fst_2462_; lean_object* v_snd_2463_; lean_object* v___x_2465_; uint8_t v_isShared_2466_; uint8_t v_isSharedCheck_2540_; 
v_fst_2462_ = lean_ctor_get(v_a_2458_, 0);
v_snd_2463_ = lean_ctor_get(v_a_2458_, 1);
v_isSharedCheck_2540_ = !lean_is_exclusive(v_a_2458_);
if (v_isSharedCheck_2540_ == 0)
{
v___x_2465_ = v_a_2458_;
v_isShared_2466_ = v_isSharedCheck_2540_;
goto v_resetjp_2464_;
}
else
{
lean_inc(v_snd_2463_);
lean_inc(v_fst_2462_);
lean_dec(v_a_2458_);
v___x_2465_ = lean_box(0);
v_isShared_2466_ = v_isSharedCheck_2540_;
goto v_resetjp_2464_;
}
v_resetjp_2464_:
{
lean_object* v___x_2467_; lean_object* v___x_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2497_; 
v___x_2467_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__12));
lean_inc_ref_n(v___x_2429_, 8);
v___x_2468_ = l_Lean_Expr_const___override(v___x_2467_, v___x_2429_);
lean_inc_ref_n(v_00_u03b1_2180_, 7);
v___x_2469_ = l_Lean_Expr_app___override(v___x_2468_, v_00_u03b1_2180_);
v___x_2470_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__15));
v___x_2471_ = l_Lean_Expr_const___override(v___x_2470_, v___x_2429_);
v___x_2472_ = l_Lean_Expr_app___override(v___x_2471_, v_00_u03b1_2180_);
v___x_2473_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__18));
v___x_2474_ = l_Lean_Expr_const___override(v___x_2473_, v___x_2429_);
v___x_2475_ = l_Lean_Expr_app___override(v___x_2474_, v_00_u03b1_2180_);
v___x_2476_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__21));
v___x_2477_ = l_Lean_Expr_const___override(v___x_2476_, v___x_2429_);
v___x_2478_ = l_Lean_Expr_app___override(v___x_2477_, v_00_u03b1_2180_);
v___x_2479_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__24));
v___x_2480_ = l_Lean_Expr_const___override(v___x_2479_, v___x_2429_);
v___x_2481_ = l_Lean_Expr_app___override(v___x_2480_, v_00_u03b1_2180_);
v___x_2482_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__27));
v___x_2483_ = l_Lean_Expr_const___override(v___x_2482_, v___x_2429_);
v___x_2484_ = l_Lean_Expr_app___override(v___x_2483_, v_00_u03b1_2180_);
v___x_2485_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__29));
v___x_2486_ = l_Lean_Expr_const___override(v___x_2485_, v___x_2429_);
v___x_2487_ = l_Lean_Expr_app___override(v___x_2486_, v_00_u03b1_2180_);
v___x_2488_ = l_Lean_Expr_app___override(v___x_2487_, v___x_2445_);
v___x_2489_ = l_Lean_Expr_app___override(v___x_2484_, v___x_2488_);
v___x_2490_ = l_Lean_Expr_app___override(v___x_2481_, v___x_2489_);
v___x_2491_ = l_Lean_Expr_app___override(v___x_2478_, v___x_2490_);
v___x_2492_ = l_Lean_Expr_app___override(v___x_2475_, v___x_2491_);
v___x_2493_ = l_Lean_Expr_app___override(v___x_2472_, v___x_2492_);
v___x_2494_ = l_Lean_Expr_app___override(v___x_2469_, v___x_2493_);
v___x_2495_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__34));
lean_inc(v_u_2179_);
if (v_isShared_2466_ == 0)
{
lean_ctor_set_tag(v___x_2465_, 1);
lean_ctor_set(v___x_2465_, 1, v___x_2429_);
lean_ctor_set(v___x_2465_, 0, v_u_2179_);
v___x_2497_ = v___x_2465_;
goto v_reusejp_2496_;
}
else
{
lean_object* v_reuseFailAlloc_2539_; 
v_reuseFailAlloc_2539_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2539_, 0, v_u_2179_);
lean_ctor_set(v_reuseFailAlloc_2539_, 1, v___x_2429_);
v___x_2497_ = v_reuseFailAlloc_2539_;
goto v_reusejp_2496_;
}
v_reusejp_2496_:
{
lean_object* v___x_2499_; 
if (v_isShared_2456_ == 0)
{
lean_ctor_set_tag(v___x_2455_, 1);
lean_ctor_set(v___x_2455_, 1, v___x_2497_);
lean_ctor_set(v___x_2455_, 0, v_u_2179_);
v___x_2499_ = v___x_2455_;
goto v_reusejp_2498_;
}
else
{
lean_object* v_reuseFailAlloc_2538_; 
v_reuseFailAlloc_2538_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2538_, 0, v_u_2179_);
lean_ctor_set(v_reuseFailAlloc_2538_, 1, v___x_2497_);
v___x_2499_ = v_reuseFailAlloc_2538_;
goto v_reusejp_2498_;
}
v_reusejp_2498_:
{
lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; uint8_t v___x_2533_; lean_object* v___x_2534_; lean_object* v___x_2536_; 
v___x_2500_ = l_Lean_Expr_const___override(v___x_2495_, v___x_2499_);
lean_inc_ref_n(v_00_u03b1_2180_, 6);
v___x_2501_ = l_Lean_Expr_app___override(v___x_2500_, v_00_u03b1_2180_);
v___x_2502_ = l_Lean_Expr_app___override(v___x_2501_, v_00_u03b1_2180_);
v___x_2503_ = l_Lean_Expr_app___override(v___x_2502_, v_00_u03b1_2180_);
v___x_2504_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__36));
lean_inc_ref_n(v___x_2429_, 3);
v___x_2505_ = l_Lean_Expr_const___override(v___x_2504_, v___x_2429_);
v___x_2506_ = l_Lean_Expr_app___override(v___x_2505_, v_00_u03b1_2180_);
v___x_2507_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__39));
v___x_2508_ = l_Lean_Expr_const___override(v___x_2507_, v___x_2429_);
v___x_2509_ = l_Lean_Expr_app___override(v___x_2508_, v_00_u03b1_2180_);
v___x_2510_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__46));
v___x_2511_ = l_Lean_Expr_const___override(v___x_2510_, v___x_2429_);
v___x_2512_ = l_Lean_Expr_app___override(v___x_2511_, v_00_u03b1_2180_);
lean_inc_ref(v_inst_2424_);
v___x_2513_ = l_Lean_Expr_app___override(v___x_2512_, v_inst_2424_);
v___x_2514_ = l_Lean_Expr_app___override(v___x_2509_, v___x_2513_);
v___x_2515_ = l_Lean_Expr_app___override(v___x_2506_, v___x_2514_);
v___x_2516_ = l_Lean_Expr_app___override(v___x_2503_, v___x_2515_);
lean_inc(v_fst_2452_);
v___x_2517_ = l_Lean_Expr_app___override(v___x_2516_, v_fst_2452_);
lean_inc(v_fst_2462_);
v___x_2518_ = l_Lean_Expr_app___override(v___x_2517_, v_fst_2462_);
v___x_2519_ = l_Lean_Expr_app___override(v___x_2494_, v___x_2518_);
v___x_2520_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___closed__47));
v___x_2521_ = l_Lean_Expr_const___override(v___x_2520_, v___x_2429_);
v___x_2522_ = l_Lean_Expr_app___override(v___x_2521_, v_00_u03b1_2180_);
v___x_2523_ = l_Lean_Expr_app___override(v___x_2522_, v_inst_2424_);
v___x_2524_ = l_Lean_Expr_app___override(v___x_2523_, v_n_2425_);
v___x_2525_ = l_Lean_Expr_app___override(v___x_2524_, v_d_2426_);
v___x_2526_ = l_Lean_Expr_app___override(v___x_2525_, v_e_2181_);
v___x_2527_ = l_Lean_Expr_app___override(v___x_2526_, v_fst_2452_);
v___x_2528_ = l_Lean_Expr_app___override(v___x_2527_, v_fst_2462_);
v___x_2529_ = l_Lean_Expr_app___override(v___x_2528_, v_proof_2427_);
v___x_2530_ = l_Lean_Expr_app___override(v___x_2529_, v_snd_2453_);
v___x_2531_ = l_Lean_Expr_app___override(v___x_2530_, v_snd_2463_);
v___x_2532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2532_, 0, v___x_2531_);
v___x_2533_ = 1;
v___x_2534_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2534_, 0, v___x_2519_);
lean_ctor_set(v___x_2534_, 1, v___x_2532_);
lean_ctor_set_uint8(v___x_2534_, sizeof(void*)*2, v___x_2533_);
if (v_isShared_2461_ == 0)
{
lean_ctor_set(v___x_2460_, 0, v___x_2534_);
v___x_2536_ = v___x_2460_;
goto v_reusejp_2535_;
}
else
{
lean_object* v_reuseFailAlloc_2537_; 
v_reuseFailAlloc_2537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2537_, 0, v___x_2534_);
v___x_2536_ = v_reuseFailAlloc_2537_;
goto v_reusejp_2535_;
}
v_reusejp_2535_:
{
return v___x_2536_;
}
}
}
}
}
}
else
{
lean_object* v_a_2542_; lean_object* v___x_2544_; uint8_t v_isShared_2545_; uint8_t v_isSharedCheck_2549_; 
lean_del_object(v___x_2455_);
lean_dec(v_snd_2453_);
lean_dec(v_fst_2452_);
lean_dec_ref(v___x_2445_);
lean_dec_ref_known(v___x_2429_, 2);
lean_dec_ref(v_proof_2427_);
lean_dec_ref(v_d_2426_);
lean_dec_ref(v_n_2425_);
lean_dec_ref(v_inst_2424_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
lean_dec(v_u_2179_);
v_a_2542_ = lean_ctor_get(v___x_2457_, 0);
v_isSharedCheck_2549_ = !lean_is_exclusive(v___x_2457_);
if (v_isSharedCheck_2549_ == 0)
{
v___x_2544_ = v___x_2457_;
v_isShared_2545_ = v_isSharedCheck_2549_;
goto v_resetjp_2543_;
}
else
{
lean_inc(v_a_2542_);
lean_dec(v___x_2457_);
v___x_2544_ = lean_box(0);
v_isShared_2545_ = v_isSharedCheck_2549_;
goto v_resetjp_2543_;
}
v_resetjp_2543_:
{
lean_object* v___x_2547_; 
if (v_isShared_2545_ == 0)
{
v___x_2547_ = v___x_2544_;
goto v_reusejp_2546_;
}
else
{
lean_object* v_reuseFailAlloc_2548_; 
v_reuseFailAlloc_2548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2548_, 0, v_a_2542_);
v___x_2547_ = v_reuseFailAlloc_2548_;
goto v_reusejp_2546_;
}
v_reusejp_2546_:
{
return v___x_2547_;
}
}
}
}
}
else
{
lean_object* v_a_2551_; lean_object* v___x_2553_; uint8_t v_isShared_2554_; uint8_t v_isSharedCheck_2558_; 
lean_dec_ref(v___x_2449_);
lean_dec_ref(v___x_2445_);
lean_dec_ref_known(v___x_2429_, 2);
lean_dec_ref(v_proof_2427_);
lean_dec_ref(v_d_2426_);
lean_dec_ref(v_n_2425_);
lean_dec_ref(v_inst_2424_);
lean_dec_ref(v_e_2181_);
lean_dec_ref(v_00_u03b1_2180_);
lean_dec(v_u_2179_);
v_a_2551_ = lean_ctor_get(v___x_2450_, 0);
v_isSharedCheck_2558_ = !lean_is_exclusive(v___x_2450_);
if (v_isSharedCheck_2558_ == 0)
{
v___x_2553_ = v___x_2450_;
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
else
{
lean_inc(v_a_2551_);
lean_dec(v___x_2450_);
v___x_2553_ = lean_box(0);
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
v_resetjp_2552_:
{
lean_object* v___x_2556_; 
if (v_isShared_2554_ == 0)
{
v___x_2556_ = v___x_2553_;
goto v_reusejp_2555_;
}
else
{
lean_object* v_reuseFailAlloc_2557_; 
v_reuseFailAlloc_2557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2557_, 0, v_a_2551_);
v___x_2556_ = v_reuseFailAlloc_2557_;
goto v_reusejp_2555_;
}
v_reusejp_2555_:
{
return v___x_2556_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult___boxed(lean_object* v_u_2559_, lean_object* v_00_u03b1_2560_, lean_object* v_e_2561_, lean_object* v_x_2562_, lean_object* v_a_2563_, lean_object* v_a_2564_, lean_object* v_a_2565_, lean_object* v_a_2566_, lean_object* v_a_2567_){
_start:
{
lean_object* v_res_2568_; 
v_res_2568_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toSimpResult(v_u_2559_, v_00_u03b1_2560_, v_e_2561_, v_x_2562_, v_a_2563_, v_a_2564_, v_a_2565_, v_a_2566_);
lean_dec(v_a_2566_);
lean_dec_ref(v_a_2565_);
lean_dec(v_a_2564_);
lean_dec_ref(v_a_2563_);
return v_res_2568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___redArg(uint8_t v_b_2569_, lean_object* v_prf_2570_){
_start:
{
lean_object* v___x_2571_; 
v___x_2571_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_2571_, 0, v_prf_2570_);
lean_ctor_set_uint8(v___x_2571_, sizeof(void*)*1, v_b_2569_);
return v___x_2571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___redArg___boxed(lean_object* v_b_2572_, lean_object* v_prf_2573_){
_start:
{
uint8_t v_b_boxed_2574_; lean_object* v_res_2575_; 
v_b_boxed_2574_ = lean_unbox(v_b_2572_);
v_res_2575_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___redArg(v_b_boxed_2574_, v_prf_2573_);
return v_res_2575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult(lean_object* v_p_2576_, uint8_t v_b_2577_, lean_object* v_prf_2578_){
_start:
{
lean_object* v___x_2579_; 
v___x_2579_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_2579_, 0, v_prf_2578_);
lean_ctor_set_uint8(v___x_2579_, sizeof(void*)*1, v_b_2577_);
return v___x_2579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult___boxed(lean_object* v_p_2580_, lean_object* v_b_2581_, lean_object* v_prf_2582_){
_start:
{
uint8_t v_b_boxed_2583_; lean_object* v_res_2584_; 
v_b_boxed_2583_ = lean_unbox(v_b_2581_);
v_res_2584_ = lp_mathlib_Mathlib_Meta_NormNum_Result_ofBoolResult(v_p_2580_, v_b_boxed_2583_, v_prf_2582_);
lean_dec_ref(v_p_2580_);
return v_res_2584_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2(void){
_start:
{
lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2590_; 
v___x_2588_ = lean_box(0);
v___x_2589_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__1));
v___x_2590_ = l_Lean_Expr_const___override(v___x_2589_, v___x_2588_);
return v___x_2590_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5(void){
_start:
{
lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; 
v___x_2595_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3);
v___x_2596_ = lean_box(0);
v___x_2597_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2597_, 0, v___x_2596_);
lean_ctor_set(v___x_2597_, 1, v___x_2595_);
return v___x_2597_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6(void){
_start:
{
lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; 
v___x_2598_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__5);
v___x_2599_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4));
v___x_2600_ = l_Lean_Expr_const___override(v___x_2599_, v___x_2598_);
return v___x_2600_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7(void){
_start:
{
lean_object* v___x_2601_; lean_object* v___x_2602_; 
v___x_2601_ = lean_box(0);
v___x_2602_ = l_Lean_Expr_sort___override(v___x_2601_);
return v___x_2602_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8(void){
_start:
{
lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; 
v___x_2603_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7);
v___x_2604_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__6);
v___x_2605_ = l_Lean_Expr_app___override(v___x_2604_, v___x_2603_);
return v___x_2605_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13(void){
_start:
{
lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; 
v___x_2612_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7);
v___x_2613_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__4);
v___x_2614_ = l_Lean_Expr_app___override(v___x_2613_, v___x_2612_);
return v___x_2614_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14(void){
_start:
{
lean_object* v___x_2615_; lean_object* v___x_2616_; 
v___x_2615_ = lean_unsigned_to_nat(0u);
v___x_2616_ = l_Lean_Expr_bvar___override(v___x_2615_);
return v___x_2616_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15(void){
_start:
{
lean_object* v___x_2617_; lean_object* v___x_2618_; 
v___x_2617_ = lean_unsigned_to_nat(1u);
v___x_2618_ = l_Lean_Expr_bvar___override(v___x_2617_);
return v___x_2618_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16(void){
_start:
{
lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2619_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2620_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__2);
v___x_2621_ = l_Lean_Expr_app___override(v___x_2620_, v___x_2619_);
return v___x_2621_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19(void){
_start:
{
lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; 
v___x_2626_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3, &lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3_once, _init_lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__3);
v___x_2627_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18));
v___x_2628_ = l_Lean_Expr_const___override(v___x_2627_, v___x_2626_);
return v___x_2628_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20(void){
_start:
{
lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; 
v___x_2629_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7);
v___x_2630_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__19);
v___x_2631_ = l_Lean_Expr_app___override(v___x_2630_, v___x_2629_);
return v___x_2631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans(lean_object* v_u_2652_, lean_object* v_00_u03b1_2653_, lean_object* v_a_2654_, lean_object* v_b_2655_, lean_object* v_eq_2656_, lean_object* v_x_2657_){
_start:
{
switch(lean_obj_tag(v_x_2657_))
{
case 0:
{
uint8_t v_val_2658_; 
lean_dec_ref(v_00_u03b1_2653_);
lean_dec(v_u_2652_);
v_val_2658_ = lean_ctor_get_uint8(v_x_2657_, sizeof(void*)*1);
if (v_val_2658_ == 0)
{
lean_object* v_proof_2659_; lean_object* v___x_2661_; uint8_t v_isShared_2662_; uint8_t v_isSharedCheck_2687_; 
v_proof_2659_ = lean_ctor_get(v_x_2657_, 0);
v_isSharedCheck_2687_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2661_ = v_x_2657_;
v_isShared_2662_ = v_isSharedCheck_2687_;
goto v_resetjp_2660_;
}
else
{
lean_inc(v_proof_2659_);
lean_dec(v_x_2657_);
v___x_2661_ = lean_box(0);
v_isShared_2662_ = v_isSharedCheck_2687_;
goto v_resetjp_2660_;
}
v_resetjp_2660_:
{
lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; uint8_t v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v___x_2679_; lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2685_; 
v___x_2663_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7);
v___x_2664_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2665_ = l_Lean_Expr_app___override(v___x_2664_, v_b_2655_);
v___x_2666_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2667_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2668_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13);
v___x_2669_ = l_Lean_Expr_app___override(v___x_2668_, v_b_2655_);
v___x_2670_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2671_ = l_Lean_Expr_app___override(v___x_2669_, v___x_2670_);
v___x_2672_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__16);
v___x_2673_ = 0;
v___x_2674_ = l_Lean_Expr_lam___override(v___x_2667_, v___x_2671_, v___x_2672_, v___x_2673_);
v___x_2675_ = l_Lean_Expr_lam___override(v___x_2666_, v___x_2663_, v___x_2674_, v___x_2673_);
v___x_2676_ = l_Lean_Expr_app___override(v___x_2665_, v___x_2675_);
v___x_2677_ = l_Lean_Expr_app___override(v___x_2676_, v_proof_2659_);
lean_inc_ref(v_a_2654_);
v___x_2678_ = l_Lean_Expr_app___override(v___x_2677_, v_a_2654_);
v___x_2679_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20);
v___x_2680_ = l_Lean_Expr_app___override(v___x_2679_, v_a_2654_);
v___x_2681_ = l_Lean_Expr_app___override(v___x_2680_, v_b_2655_);
v___x_2682_ = l_Lean_Expr_app___override(v___x_2681_, v_eq_2656_);
v___x_2683_ = l_Lean_Expr_app___override(v___x_2678_, v___x_2682_);
if (v_isShared_2662_ == 0)
{
lean_ctor_set(v___x_2661_, 0, v___x_2683_);
v___x_2685_ = v___x_2661_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v___x_2683_);
lean_ctor_set_uint8(v_reuseFailAlloc_2686_, sizeof(void*)*1, v_val_2658_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
else
{
lean_object* v_proof_2688_; lean_object* v___x_2690_; uint8_t v_isShared_2691_; uint8_t v_isSharedCheck_2716_; 
v_proof_2688_ = lean_ctor_get(v_x_2657_, 0);
v_isSharedCheck_2716_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2716_ == 0)
{
v___x_2690_ = v_x_2657_;
v_isShared_2691_ = v_isSharedCheck_2716_;
goto v_resetjp_2689_;
}
else
{
lean_inc(v_proof_2688_);
lean_dec(v_x_2657_);
v___x_2690_ = lean_box(0);
v_isShared_2691_ = v_isSharedCheck_2716_;
goto v_resetjp_2689_;
}
v_resetjp_2689_:
{
lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; uint8_t v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2714_; 
v___x_2692_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__7);
v___x_2693_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__8);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2694_ = l_Lean_Expr_app___override(v___x_2693_, v_b_2655_);
v___x_2695_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2696_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2697_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__13);
v___x_2698_ = l_Lean_Expr_app___override(v___x_2697_, v_b_2655_);
v___x_2699_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2700_ = l_Lean_Expr_app___override(v___x_2698_, v___x_2699_);
v___x_2701_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2702_ = 0;
v___x_2703_ = l_Lean_Expr_lam___override(v___x_2696_, v___x_2700_, v___x_2701_, v___x_2702_);
v___x_2704_ = l_Lean_Expr_lam___override(v___x_2695_, v___x_2692_, v___x_2703_, v___x_2702_);
v___x_2705_ = l_Lean_Expr_app___override(v___x_2694_, v___x_2704_);
v___x_2706_ = l_Lean_Expr_app___override(v___x_2705_, v_proof_2688_);
lean_inc_ref(v_a_2654_);
v___x_2707_ = l_Lean_Expr_app___override(v___x_2706_, v_a_2654_);
v___x_2708_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__20);
v___x_2709_ = l_Lean_Expr_app___override(v___x_2708_, v_a_2654_);
v___x_2710_ = l_Lean_Expr_app___override(v___x_2709_, v_b_2655_);
v___x_2711_ = l_Lean_Expr_app___override(v___x_2710_, v_eq_2656_);
v___x_2712_ = l_Lean_Expr_app___override(v___x_2707_, v___x_2711_);
if (v_isShared_2691_ == 0)
{
lean_ctor_set(v___x_2690_, 0, v___x_2712_);
v___x_2714_ = v___x_2690_;
goto v_reusejp_2713_;
}
else
{
lean_object* v_reuseFailAlloc_2715_; 
v_reuseFailAlloc_2715_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2715_, 0, v___x_2712_);
lean_ctor_set_uint8(v_reuseFailAlloc_2715_, sizeof(void*)*1, v_val_2658_);
v___x_2714_ = v_reuseFailAlloc_2715_;
goto v_reusejp_2713_;
}
v_reusejp_2713_:
{
return v___x_2714_;
}
}
}
}
case 1:
{
lean_object* v_inst_2717_; lean_object* v_lit_2718_; lean_object* v_proof_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2764_; 
v_inst_2717_ = lean_ctor_get(v_x_2657_, 0);
v_lit_2718_ = lean_ctor_get(v_x_2657_, 1);
v_proof_2719_ = lean_ctor_get(v_x_2657_, 2);
v_isSharedCheck_2764_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2764_ == 0)
{
v___x_2721_ = v_x_2657_;
v_isShared_2722_ = v_isSharedCheck_2764_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_proof_2719_);
lean_inc(v_lit_2718_);
lean_inc(v_inst_2717_);
lean_dec(v_x_2657_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2764_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; lean_object* v___x_2728_; lean_object* v___x_2729_; lean_object* v___x_2730_; lean_object* v___x_2731_; lean_object* v___x_2732_; lean_object* v___x_2733_; lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; uint8_t v___x_2748_; lean_object* v___x_2749_; lean_object* v___x_2750_; lean_object* v___x_2751_; lean_object* v___x_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2762_; 
v___x_2723_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__21));
v___x_2724_ = lean_box(0);
lean_inc(v_u_2652_);
v___x_2725_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2725_, 0, v_u_2652_);
lean_ctor_set(v___x_2725_, 1, v___x_2724_);
v___x_2726_ = l_Lean_Expr_const___override(v___x_2723_, v___x_2725_);
lean_inc_ref_n(v_00_u03b1_2653_, 4);
v___x_2727_ = l_Lean_Expr_app___override(v___x_2726_, v_00_u03b1_2653_);
lean_inc_ref(v_inst_2717_);
v___x_2728_ = l_Lean_Expr_app___override(v___x_2727_, v_inst_2717_);
v___x_2729_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4));
v___x_2730_ = lean_box(0);
v___x_2731_ = l_Lean_Level_succ___override(v_u_2652_);
v___x_2732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2732_, 0, v___x_2731_);
lean_ctor_set(v___x_2732_, 1, v___x_2724_);
lean_inc_ref_n(v___x_2732_, 2);
v___x_2733_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2733_, 0, v___x_2730_);
lean_ctor_set(v___x_2733_, 1, v___x_2732_);
v___x_2734_ = l_Lean_Expr_const___override(v___x_2729_, v___x_2733_);
v___x_2735_ = l_Lean_Expr_app___override(v___x_2734_, v_00_u03b1_2653_);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2736_ = l_Lean_Expr_app___override(v___x_2735_, v_b_2655_);
v___x_2737_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2738_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2739_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1));
v___x_2740_ = l_Lean_Expr_const___override(v___x_2739_, v___x_2732_);
v___x_2741_ = l_Lean_Expr_app___override(v___x_2740_, v_00_u03b1_2653_);
v___x_2742_ = l_Lean_Expr_app___override(v___x_2741_, v_b_2655_);
v___x_2743_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2744_ = l_Lean_Expr_app___override(v___x_2742_, v___x_2743_);
v___x_2745_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2746_ = l_Lean_Expr_app___override(v___x_2728_, v___x_2745_);
lean_inc_ref(v_lit_2718_);
v___x_2747_ = l_Lean_Expr_app___override(v___x_2746_, v_lit_2718_);
v___x_2748_ = 0;
v___x_2749_ = l_Lean_Expr_lam___override(v___x_2738_, v___x_2744_, v___x_2747_, v___x_2748_);
v___x_2750_ = l_Lean_Expr_lam___override(v___x_2737_, v_00_u03b1_2653_, v___x_2749_, v___x_2748_);
v___x_2751_ = l_Lean_Expr_app___override(v___x_2736_, v___x_2750_);
v___x_2752_ = l_Lean_Expr_app___override(v___x_2751_, v_proof_2719_);
lean_inc_ref(v_a_2654_);
v___x_2753_ = l_Lean_Expr_app___override(v___x_2752_, v_a_2654_);
v___x_2754_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18));
v___x_2755_ = l_Lean_Expr_const___override(v___x_2754_, v___x_2732_);
v___x_2756_ = l_Lean_Expr_app___override(v___x_2755_, v_00_u03b1_2653_);
v___x_2757_ = l_Lean_Expr_app___override(v___x_2756_, v_a_2654_);
v___x_2758_ = l_Lean_Expr_app___override(v___x_2757_, v_b_2655_);
v___x_2759_ = l_Lean_Expr_app___override(v___x_2758_, v_eq_2656_);
v___x_2760_ = l_Lean_Expr_app___override(v___x_2753_, v___x_2759_);
if (v_isShared_2722_ == 0)
{
lean_ctor_set(v___x_2721_, 2, v___x_2760_);
v___x_2762_ = v___x_2721_;
goto v_reusejp_2761_;
}
else
{
lean_object* v_reuseFailAlloc_2763_; 
v_reuseFailAlloc_2763_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2763_, 0, v_inst_2717_);
lean_ctor_set(v_reuseFailAlloc_2763_, 1, v_lit_2718_);
lean_ctor_set(v_reuseFailAlloc_2763_, 2, v___x_2760_);
v___x_2762_ = v_reuseFailAlloc_2763_;
goto v_reusejp_2761_;
}
v_reusejp_2761_:
{
return v___x_2762_;
}
}
}
case 2:
{
lean_object* v_inst_2765_; lean_object* v_lit_2766_; lean_object* v_proof_2767_; lean_object* v___x_2769_; uint8_t v_isShared_2770_; uint8_t v_isSharedCheck_2814_; 
v_inst_2765_ = lean_ctor_get(v_x_2657_, 0);
v_lit_2766_ = lean_ctor_get(v_x_2657_, 1);
v_proof_2767_ = lean_ctor_get(v_x_2657_, 2);
v_isSharedCheck_2814_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2814_ == 0)
{
v___x_2769_ = v_x_2657_;
v_isShared_2770_ = v_isSharedCheck_2814_;
goto v_resetjp_2768_;
}
else
{
lean_inc(v_proof_2767_);
lean_inc(v_lit_2766_);
lean_inc(v_inst_2765_);
lean_dec(v_x_2657_);
v___x_2769_ = lean_box(0);
v_isShared_2770_ = v_isSharedCheck_2814_;
goto v_resetjp_2768_;
}
v_resetjp_2768_:
{
lean_object* v___x_2771_; lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; lean_object* v___x_2793_; lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2797_; uint8_t v___x_2798_; lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2812_; 
v___x_2771_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__22));
v___x_2772_ = lean_box(0);
lean_inc(v_u_2652_);
v___x_2773_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2773_, 0, v_u_2652_);
lean_ctor_set(v___x_2773_, 1, v___x_2772_);
v___x_2774_ = l_Lean_Expr_const___override(v___x_2771_, v___x_2773_);
lean_inc_ref_n(v_00_u03b1_2653_, 4);
v___x_2775_ = l_Lean_Expr_app___override(v___x_2774_, v_00_u03b1_2653_);
lean_inc_ref(v_inst_2765_);
v___x_2776_ = l_Lean_Expr_app___override(v___x_2775_, v_inst_2765_);
v___x_2777_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
lean_inc_ref(v_lit_2766_);
v___x_2778_ = l_Lean_Expr_app___override(v___x_2777_, v_lit_2766_);
v___x_2779_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4));
v___x_2780_ = lean_box(0);
v___x_2781_ = l_Lean_Level_succ___override(v_u_2652_);
v___x_2782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2782_, 0, v___x_2781_);
lean_ctor_set(v___x_2782_, 1, v___x_2772_);
lean_inc_ref_n(v___x_2782_, 2);
v___x_2783_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2783_, 0, v___x_2780_);
lean_ctor_set(v___x_2783_, 1, v___x_2782_);
v___x_2784_ = l_Lean_Expr_const___override(v___x_2779_, v___x_2783_);
v___x_2785_ = l_Lean_Expr_app___override(v___x_2784_, v_00_u03b1_2653_);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2786_ = l_Lean_Expr_app___override(v___x_2785_, v_b_2655_);
v___x_2787_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2788_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2789_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1));
v___x_2790_ = l_Lean_Expr_const___override(v___x_2789_, v___x_2782_);
v___x_2791_ = l_Lean_Expr_app___override(v___x_2790_, v_00_u03b1_2653_);
v___x_2792_ = l_Lean_Expr_app___override(v___x_2791_, v_b_2655_);
v___x_2793_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2794_ = l_Lean_Expr_app___override(v___x_2792_, v___x_2793_);
v___x_2795_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2796_ = l_Lean_Expr_app___override(v___x_2776_, v___x_2795_);
v___x_2797_ = l_Lean_Expr_app___override(v___x_2796_, v___x_2778_);
v___x_2798_ = 0;
v___x_2799_ = l_Lean_Expr_lam___override(v___x_2788_, v___x_2794_, v___x_2797_, v___x_2798_);
v___x_2800_ = l_Lean_Expr_lam___override(v___x_2787_, v_00_u03b1_2653_, v___x_2799_, v___x_2798_);
v___x_2801_ = l_Lean_Expr_app___override(v___x_2786_, v___x_2800_);
v___x_2802_ = l_Lean_Expr_app___override(v___x_2801_, v_proof_2767_);
lean_inc_ref(v_a_2654_);
v___x_2803_ = l_Lean_Expr_app___override(v___x_2802_, v_a_2654_);
v___x_2804_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18));
v___x_2805_ = l_Lean_Expr_const___override(v___x_2804_, v___x_2782_);
v___x_2806_ = l_Lean_Expr_app___override(v___x_2805_, v_00_u03b1_2653_);
v___x_2807_ = l_Lean_Expr_app___override(v___x_2806_, v_a_2654_);
v___x_2808_ = l_Lean_Expr_app___override(v___x_2807_, v_b_2655_);
v___x_2809_ = l_Lean_Expr_app___override(v___x_2808_, v_eq_2656_);
v___x_2810_ = l_Lean_Expr_app___override(v___x_2803_, v___x_2809_);
if (v_isShared_2770_ == 0)
{
lean_ctor_set(v___x_2769_, 2, v___x_2810_);
v___x_2812_ = v___x_2769_;
goto v_reusejp_2811_;
}
else
{
lean_object* v_reuseFailAlloc_2813_; 
v_reuseFailAlloc_2813_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2813_, 0, v_inst_2765_);
lean_ctor_set(v_reuseFailAlloc_2813_, 1, v_lit_2766_);
lean_ctor_set(v_reuseFailAlloc_2813_, 2, v___x_2810_);
v___x_2812_ = v_reuseFailAlloc_2813_;
goto v_reusejp_2811_;
}
v_reusejp_2811_:
{
return v___x_2812_;
}
}
}
case 3:
{
lean_object* v_inst_2815_; lean_object* v_q_2816_; lean_object* v_n_2817_; lean_object* v_d_2818_; lean_object* v_proof_2819_; lean_object* v___x_2821_; uint8_t v_isShared_2822_; uint8_t v_isSharedCheck_2869_; 
v_inst_2815_ = lean_ctor_get(v_x_2657_, 0);
v_q_2816_ = lean_ctor_get(v_x_2657_, 1);
v_n_2817_ = lean_ctor_get(v_x_2657_, 2);
v_d_2818_ = lean_ctor_get(v_x_2657_, 3);
v_proof_2819_ = lean_ctor_get(v_x_2657_, 4);
v_isSharedCheck_2869_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2869_ == 0)
{
v___x_2821_ = v_x_2657_;
v_isShared_2822_ = v_isSharedCheck_2869_;
goto v_resetjp_2820_;
}
else
{
lean_inc(v_proof_2819_);
lean_inc(v_d_2818_);
lean_inc(v_n_2817_);
lean_inc(v_q_2816_);
lean_inc(v_inst_2815_);
lean_dec(v_x_2657_);
v___x_2821_ = lean_box(0);
v_isShared_2822_ = v_isSharedCheck_2869_;
goto v_resetjp_2820_;
}
v_resetjp_2820_:
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; lean_object* v___x_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; uint8_t v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; lean_object* v___x_2867_; 
v___x_2823_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__23));
v___x_2824_ = lean_box(0);
lean_inc(v_u_2652_);
v___x_2825_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2825_, 0, v_u_2652_);
lean_ctor_set(v___x_2825_, 1, v___x_2824_);
lean_inc_ref(v___x_2825_);
v___x_2826_ = l_Lean_Expr_const___override(v___x_2823_, v___x_2825_);
lean_inc_ref_n(v_00_u03b1_2653_, 5);
v___x_2827_ = l_Lean_Expr_app___override(v___x_2826_, v_00_u03b1_2653_);
v___x_2828_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___closed__4));
v___x_2829_ = l_Lean_Expr_const___override(v___x_2828_, v___x_2825_);
v___x_2830_ = l_Lean_Expr_app___override(v___x_2829_, v_00_u03b1_2653_);
lean_inc_ref(v_inst_2815_);
v___x_2831_ = l_Lean_Expr_app___override(v___x_2830_, v_inst_2815_);
v___x_2832_ = l_Lean_Expr_app___override(v___x_2827_, v___x_2831_);
v___x_2833_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4));
v___x_2834_ = lean_box(0);
v___x_2835_ = l_Lean_Level_succ___override(v_u_2652_);
v___x_2836_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2836_, 0, v___x_2835_);
lean_ctor_set(v___x_2836_, 1, v___x_2824_);
lean_inc_ref_n(v___x_2836_, 2);
v___x_2837_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2837_, 0, v___x_2834_);
lean_ctor_set(v___x_2837_, 1, v___x_2836_);
v___x_2838_ = l_Lean_Expr_const___override(v___x_2833_, v___x_2837_);
v___x_2839_ = l_Lean_Expr_app___override(v___x_2838_, v_00_u03b1_2653_);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2840_ = l_Lean_Expr_app___override(v___x_2839_, v_b_2655_);
v___x_2841_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2842_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2843_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1));
v___x_2844_ = l_Lean_Expr_const___override(v___x_2843_, v___x_2836_);
v___x_2845_ = l_Lean_Expr_app___override(v___x_2844_, v_00_u03b1_2653_);
v___x_2846_ = l_Lean_Expr_app___override(v___x_2845_, v_b_2655_);
v___x_2847_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2848_ = l_Lean_Expr_app___override(v___x_2846_, v___x_2847_);
v___x_2849_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2850_ = l_Lean_Expr_app___override(v___x_2832_, v___x_2849_);
lean_inc_ref(v_n_2817_);
v___x_2851_ = l_Lean_Expr_app___override(v___x_2850_, v_n_2817_);
lean_inc_ref(v_d_2818_);
v___x_2852_ = l_Lean_Expr_app___override(v___x_2851_, v_d_2818_);
v___x_2853_ = 0;
v___x_2854_ = l_Lean_Expr_lam___override(v___x_2842_, v___x_2848_, v___x_2852_, v___x_2853_);
v___x_2855_ = l_Lean_Expr_lam___override(v___x_2841_, v_00_u03b1_2653_, v___x_2854_, v___x_2853_);
v___x_2856_ = l_Lean_Expr_app___override(v___x_2840_, v___x_2855_);
v___x_2857_ = l_Lean_Expr_app___override(v___x_2856_, v_proof_2819_);
lean_inc_ref(v_a_2654_);
v___x_2858_ = l_Lean_Expr_app___override(v___x_2857_, v_a_2654_);
v___x_2859_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18));
v___x_2860_ = l_Lean_Expr_const___override(v___x_2859_, v___x_2836_);
v___x_2861_ = l_Lean_Expr_app___override(v___x_2860_, v_00_u03b1_2653_);
v___x_2862_ = l_Lean_Expr_app___override(v___x_2861_, v_a_2654_);
v___x_2863_ = l_Lean_Expr_app___override(v___x_2862_, v_b_2655_);
v___x_2864_ = l_Lean_Expr_app___override(v___x_2863_, v_eq_2656_);
v___x_2865_ = l_Lean_Expr_app___override(v___x_2858_, v___x_2864_);
if (v_isShared_2822_ == 0)
{
lean_ctor_set(v___x_2821_, 4, v___x_2865_);
v___x_2867_ = v___x_2821_;
goto v_reusejp_2866_;
}
else
{
lean_object* v_reuseFailAlloc_2868_; 
v_reuseFailAlloc_2868_ = lean_alloc_ctor(3, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2868_, 0, v_inst_2815_);
lean_ctor_set(v_reuseFailAlloc_2868_, 1, v_q_2816_);
lean_ctor_set(v_reuseFailAlloc_2868_, 2, v_n_2817_);
lean_ctor_set(v_reuseFailAlloc_2868_, 3, v_d_2818_);
lean_ctor_set(v_reuseFailAlloc_2868_, 4, v___x_2865_);
v___x_2867_ = v_reuseFailAlloc_2868_;
goto v_reusejp_2866_;
}
v_reusejp_2866_:
{
return v___x_2867_;
}
}
}
default: 
{
lean_object* v_inst_2870_; lean_object* v_q_2871_; lean_object* v_n_2872_; lean_object* v_d_2873_; lean_object* v_proof_2874_; lean_object* v___x_2876_; uint8_t v_isShared_2877_; uint8_t v_isSharedCheck_2926_; 
v_inst_2870_ = lean_ctor_get(v_x_2657_, 0);
v_q_2871_ = lean_ctor_get(v_x_2657_, 1);
v_n_2872_ = lean_ctor_get(v_x_2657_, 2);
v_d_2873_ = lean_ctor_get(v_x_2657_, 3);
v_proof_2874_ = lean_ctor_get(v_x_2657_, 4);
v_isSharedCheck_2926_ = !lean_is_exclusive(v_x_2657_);
if (v_isSharedCheck_2926_ == 0)
{
v___x_2876_ = v_x_2657_;
v_isShared_2877_ = v_isSharedCheck_2926_;
goto v_resetjp_2875_;
}
else
{
lean_inc(v_proof_2874_);
lean_inc(v_d_2873_);
lean_inc(v_n_2872_);
lean_inc(v_q_2871_);
lean_inc(v_inst_2870_);
lean_dec(v_x_2657_);
v___x_2876_ = lean_box(0);
v_isShared_2877_ = v_isSharedCheck_2926_;
goto v_resetjp_2875_;
}
v_resetjp_2875_:
{
lean_object* v___x_2878_; lean_object* v___x_2879_; lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2902_; lean_object* v___x_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2908_; lean_object* v___x_2909_; uint8_t v___x_2910_; lean_object* v___x_2911_; lean_object* v___x_2912_; lean_object* v___x_2913_; lean_object* v___x_2914_; lean_object* v___x_2915_; lean_object* v___x_2916_; lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; lean_object* v___x_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; lean_object* v___x_2924_; 
v___x_2878_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__24));
v___x_2879_ = lean_box(0);
lean_inc(v_u_2652_);
v___x_2880_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2880_, 0, v_u_2652_);
lean_ctor_set(v___x_2880_, 1, v___x_2879_);
lean_inc_ref(v___x_2880_);
v___x_2881_ = l_Lean_Expr_const___override(v___x_2878_, v___x_2880_);
lean_inc_ref_n(v_00_u03b1_2653_, 5);
v___x_2882_ = l_Lean_Expr_app___override(v___x_2881_, v_00_u03b1_2653_);
v___x_2883_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___closed__8));
v___x_2884_ = l_Lean_Expr_const___override(v___x_2883_, v___x_2880_);
v___x_2885_ = l_Lean_Expr_app___override(v___x_2884_, v_00_u03b1_2653_);
lean_inc_ref(v_inst_2870_);
v___x_2886_ = l_Lean_Expr_app___override(v___x_2885_, v_inst_2870_);
v___x_2887_ = l_Lean_Expr_app___override(v___x_2882_, v___x_2886_);
v___x_2888_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4, &lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4_once, _init_lp_mathlib_Mathlib_Meta_NormNum_mkRawIntLit___closed__4);
lean_inc_ref(v_n_2872_);
v___x_2889_ = l_Lean_Expr_app___override(v___x_2888_, v_n_2872_);
v___x_2890_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__4));
v___x_2891_ = lean_box(0);
v___x_2892_ = l_Lean_Level_succ___override(v_u_2652_);
v___x_2893_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2893_, 0, v___x_2892_);
lean_ctor_set(v___x_2893_, 1, v___x_2879_);
lean_inc_ref_n(v___x_2893_, 2);
v___x_2894_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2894_, 0, v___x_2891_);
lean_ctor_set(v___x_2894_, 1, v___x_2893_);
v___x_2895_ = l_Lean_Expr_const___override(v___x_2890_, v___x_2894_);
v___x_2896_ = l_Lean_Expr_app___override(v___x_2895_, v_00_u03b1_2653_);
lean_inc_ref_n(v_b_2655_, 2);
v___x_2897_ = l_Lean_Expr_app___override(v___x_2896_, v_b_2655_);
v___x_2898_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__10));
v___x_2899_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__12));
v___x_2900_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_rawIntLitNatAbs___closed__1));
v___x_2901_ = l_Lean_Expr_const___override(v___x_2900_, v___x_2893_);
v___x_2902_ = l_Lean_Expr_app___override(v___x_2901_, v_00_u03b1_2653_);
v___x_2903_ = l_Lean_Expr_app___override(v___x_2902_, v_b_2655_);
v___x_2904_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__14);
v___x_2905_ = l_Lean_Expr_app___override(v___x_2903_, v___x_2904_);
v___x_2906_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15, &lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15_once, _init_lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__15);
v___x_2907_ = l_Lean_Expr_app___override(v___x_2887_, v___x_2906_);
v___x_2908_ = l_Lean_Expr_app___override(v___x_2907_, v___x_2889_);
lean_inc_ref(v_d_2873_);
v___x_2909_ = l_Lean_Expr_app___override(v___x_2908_, v_d_2873_);
v___x_2910_ = 0;
v___x_2911_ = l_Lean_Expr_lam___override(v___x_2899_, v___x_2905_, v___x_2909_, v___x_2910_);
v___x_2912_ = l_Lean_Expr_lam___override(v___x_2898_, v_00_u03b1_2653_, v___x_2911_, v___x_2910_);
v___x_2913_ = l_Lean_Expr_app___override(v___x_2897_, v___x_2912_);
v___x_2914_ = l_Lean_Expr_app___override(v___x_2913_, v_proof_2874_);
lean_inc_ref(v_a_2654_);
v___x_2915_ = l_Lean_Expr_app___override(v___x_2914_, v_a_2654_);
v___x_2916_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_NormNum_Result_eqTrans___closed__18));
v___x_2917_ = l_Lean_Expr_const___override(v___x_2916_, v___x_2893_);
v___x_2918_ = l_Lean_Expr_app___override(v___x_2917_, v_00_u03b1_2653_);
v___x_2919_ = l_Lean_Expr_app___override(v___x_2918_, v_a_2654_);
v___x_2920_ = l_Lean_Expr_app___override(v___x_2919_, v_b_2655_);
v___x_2921_ = l_Lean_Expr_app___override(v___x_2920_, v_eq_2656_);
v___x_2922_ = l_Lean_Expr_app___override(v___x_2915_, v___x_2921_);
if (v_isShared_2877_ == 0)
{
lean_ctor_set(v___x_2876_, 4, v___x_2922_);
v___x_2924_ = v___x_2876_;
goto v_reusejp_2923_;
}
else
{
lean_object* v_reuseFailAlloc_2925_; 
v_reuseFailAlloc_2925_ = lean_alloc_ctor(4, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2925_, 0, v_inst_2870_);
lean_ctor_set(v_reuseFailAlloc_2925_, 1, v_q_2871_);
lean_ctor_set(v_reuseFailAlloc_2925_, 2, v_n_2872_);
lean_ctor_set(v_reuseFailAlloc_2925_, 3, v_d_2873_);
lean_ctor_set(v_reuseFailAlloc_2925_, 4, v___x_2922_);
v___x_2924_ = v_reuseFailAlloc_2925_;
goto v_reusejp_2923_;
}
v_reusejp_2923_:
{
return v___x_2924_;
}
}
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Result(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum_Result(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default = _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27_default);
lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27 = _init_lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_instInhabitedResult_x27);
lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isNat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isNegNNRat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isInt___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isNNRat_x27___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_isRat___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_toInt___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_toNNRat_x27___auto__1);
lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___auto__1 = _init_lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___auto__1();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_NormNum_Result_toRat_x27___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Result(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Result(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum_Result(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum_Result(builtin);
}
#ifdef __cplusplus
}
#endif
