// Lean compiler output
// Module: Mathlib.Lean.Meta.KAbstractPositions
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.HeadIndex public import Lean.Meta.ExprLens public import Lean.Meta.Check
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
lean_object* lean_array_to_list(lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isTypeCorrect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushBindingDomain(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushBindingBody(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushAppFn(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushAppArg(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushProj(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushLetVarType(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushLetValue(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushLetBody(lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_Expr_toHeadIndex(lean_object*);
uint8_t l_Lean_instBEqHeadIndex_beq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headNumArgs(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_SubExpr_Pos_isRoot(lean_object*);
lean_object* l_Lean_SubExpr_Pos_tail(lean_object*);
lean_object* l_Lean_SubExpr_Pos_head(lean_object*);
extern lean_object* l_Lean_SubExpr_Pos_root;
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions_visit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions_visit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_kabstractPositions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_kabstractPositions___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_kabstractPositions___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Bad coordinate "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " for "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Can't viewRaw the type of "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Mathlib.Lean.Meta.KAbstractPositions"};
static const lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Lean.Meta.viewKAbstractSubExpr"};
static const lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Lean.Expr"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "_private.Lean.Expr.0.Lean.Expr.updateMData!Impl"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mdata expected"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Invalid coordinate "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Lensing on types is not supported"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__6 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_a"};
static const lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 106, 112, 29, 6, 211, 214, 169)}};
static const lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions_visit(lean_object* v_p_1_, lean_object* v_mctx_2_, lean_object* v_pHeadIdx_3_, lean_object* v_pNumArgs_4_, lean_object* v_e_5_, lean_object* v_pos_6_, lean_object* v_positions_7_, lean_object* v_a_8_, lean_object* v_a_9_, lean_object* v_a_10_, lean_object* v_a_11_){
_start:
{
lean_object* v_binderType_14_; lean_object* v_body_15_; lean_object* v___y_16_; lean_object* v___y_17_; lean_object* v___y_18_; lean_object* v___y_19_; lean_object* v___y_20_; lean_object* v___y_27_; lean_object* v___y_28_; lean_object* v___y_29_; lean_object* v___y_30_; lean_object* v___y_31_; uint8_t v___x_60_; 
v___x_60_ = l_Lean_Expr_hasLooseBVars(v_e_5_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; uint8_t v___x_62_; 
lean_inc_ref(v_e_5_);
v___x_61_ = l_Lean_Expr_toHeadIndex(v_e_5_);
v___x_62_ = l_Lean_instBEqHeadIndex_beq(v___x_61_, v_pHeadIdx_3_);
lean_dec(v___x_61_);
if (v___x_62_ == 0)
{
v___y_27_ = v_positions_7_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
else
{
if (v___x_60_ == 0)
{
lean_object* v___x_63_; uint8_t v___x_64_; 
v___x_63_ = l_Lean_Expr_headNumArgs(v_e_5_);
v___x_64_ = lean_nat_dec_eq(v___x_63_, v_pNumArgs_4_);
lean_dec(v___x_63_);
if (v___x_64_ == 0)
{
v___y_27_ = v_positions_7_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
else
{
lean_object* v___x_65_; 
lean_inc_ref(v_p_1_);
lean_inc_ref(v_e_5_);
v___x_65_ = l_Lean_Meta_isExprDefEq(v_e_5_, v_p_1_, v_a_8_, v_a_9_, v_a_10_, v_a_11_);
if (lean_obj_tag(v___x_65_) == 0)
{
lean_object* v_a_66_; uint8_t v___x_67_; 
v_a_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_a_66_);
lean_dec_ref_known(v___x_65_, 1);
v___x_67_ = lean_unbox(v_a_66_);
lean_dec(v_a_66_);
if (v___x_67_ == 0)
{
v___y_27_ = v_positions_7_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
else
{
lean_object* v___x_68_; lean_object* v_cache_69_; lean_object* v_zetaDeltaFVarIds_70_; lean_object* v_postponed_71_; lean_object* v_diag_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_81_; 
v___x_68_ = lean_st_ref_take(v_a_9_);
v_cache_69_ = lean_ctor_get(v___x_68_, 1);
v_zetaDeltaFVarIds_70_ = lean_ctor_get(v___x_68_, 2);
v_postponed_71_ = lean_ctor_get(v___x_68_, 3);
v_diag_72_ = lean_ctor_get(v___x_68_, 4);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_68_);
if (v_isSharedCheck_81_ == 0)
{
lean_object* v_unused_82_; 
v_unused_82_ = lean_ctor_get(v___x_68_, 0);
lean_dec(v_unused_82_);
v___x_74_ = v___x_68_;
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_diag_72_);
lean_inc(v_postponed_71_);
lean_inc(v_zetaDeltaFVarIds_70_);
lean_inc(v_cache_69_);
lean_dec(v___x_68_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___x_77_; 
lean_inc_ref(v_mctx_2_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v_mctx_2_);
v___x_77_ = v___x_74_;
goto v_reusejp_76_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_mctx_2_);
lean_ctor_set(v_reuseFailAlloc_80_, 1, v_cache_69_);
lean_ctor_set(v_reuseFailAlloc_80_, 2, v_zetaDeltaFVarIds_70_);
lean_ctor_set(v_reuseFailAlloc_80_, 3, v_postponed_71_);
lean_ctor_set(v_reuseFailAlloc_80_, 4, v_diag_72_);
v___x_77_ = v_reuseFailAlloc_80_;
goto v_reusejp_76_;
}
v_reusejp_76_:
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = lean_st_ref_set(v_a_9_, v___x_77_);
lean_inc(v_pos_6_);
v___x_79_ = lean_array_push(v_positions_7_, v_pos_6_);
v___y_27_ = v___x_79_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
}
}
}
else
{
lean_object* v_a_83_; lean_object* v___x_85_; uint8_t v_isShared_86_; uint8_t v_isSharedCheck_90_; 
lean_dec_ref(v_positions_7_);
lean_dec(v_pos_6_);
lean_dec_ref(v_e_5_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
v_a_83_ = lean_ctor_get(v___x_65_, 0);
v_isSharedCheck_90_ = !lean_is_exclusive(v___x_65_);
if (v_isSharedCheck_90_ == 0)
{
v___x_85_ = v___x_65_;
v_isShared_86_ = v_isSharedCheck_90_;
goto v_resetjp_84_;
}
else
{
lean_inc(v_a_83_);
lean_dec(v___x_65_);
v___x_85_ = lean_box(0);
v_isShared_86_ = v_isSharedCheck_90_;
goto v_resetjp_84_;
}
v_resetjp_84_:
{
lean_object* v___x_88_; 
if (v_isShared_86_ == 0)
{
v___x_88_ = v___x_85_;
goto v_reusejp_87_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v_a_83_);
v___x_88_ = v_reuseFailAlloc_89_;
goto v_reusejp_87_;
}
v_reusejp_87_:
{
return v___x_88_;
}
}
}
}
}
else
{
v___y_27_ = v_positions_7_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
}
}
else
{
v___y_27_ = v_positions_7_;
v___y_28_ = v_a_8_;
v___y_29_ = v_a_9_;
v___y_30_ = v_a_10_;
v___y_31_ = v_a_11_;
goto v___jp_26_;
}
v___jp_13_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = l_Lean_SubExpr_Pos_pushBindingDomain(v_pos_6_);
lean_inc_ref(v_mctx_2_);
lean_inc_ref(v_p_1_);
v___x_22_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_1_, v_mctx_2_, v_pHeadIdx_3_, v_pNumArgs_4_, v_binderType_14_, v___x_21_, v___y_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
if (lean_obj_tag(v___x_22_) == 0)
{
lean_object* v_a_23_; lean_object* v___x_24_; 
v_a_23_ = lean_ctor_get(v___x_22_, 0);
lean_inc(v_a_23_);
lean_dec_ref_known(v___x_22_, 1);
v___x_24_ = l_Lean_SubExpr_Pos_pushBindingBody(v_pos_6_);
lean_dec(v_pos_6_);
v_e_5_ = v_body_15_;
v_pos_6_ = v___x_24_;
v_positions_7_ = v_a_23_;
v_a_8_ = v___y_17_;
v_a_9_ = v___y_18_;
v_a_10_ = v___y_19_;
v_a_11_ = v___y_20_;
goto _start;
}
else
{
lean_dec_ref(v_body_15_);
lean_dec(v_pos_6_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
return v___x_22_;
}
}
v___jp_26_:
{
switch(lean_obj_tag(v_e_5_))
{
case 5:
{
lean_object* v_fn_32_; lean_object* v_arg_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v_fn_32_ = lean_ctor_get(v_e_5_, 0);
lean_inc_ref(v_fn_32_);
v_arg_33_ = lean_ctor_get(v_e_5_, 1);
lean_inc_ref(v_arg_33_);
lean_dec_ref_known(v_e_5_, 2);
v___x_34_ = l_Lean_SubExpr_Pos_pushAppFn(v_pos_6_);
lean_inc_ref(v_mctx_2_);
lean_inc_ref(v_p_1_);
v___x_35_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_1_, v_mctx_2_, v_pHeadIdx_3_, v_pNumArgs_4_, v_fn_32_, v___x_34_, v___y_27_, v___y_28_, v___y_29_, v___y_30_, v___y_31_);
if (lean_obj_tag(v___x_35_) == 0)
{
lean_object* v_a_36_; lean_object* v___x_37_; 
v_a_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_a_36_);
lean_dec_ref_known(v___x_35_, 1);
v___x_37_ = l_Lean_SubExpr_Pos_pushAppArg(v_pos_6_);
lean_dec(v_pos_6_);
v_e_5_ = v_arg_33_;
v_pos_6_ = v___x_37_;
v_positions_7_ = v_a_36_;
v_a_8_ = v___y_28_;
v_a_9_ = v___y_29_;
v_a_10_ = v___y_30_;
v_a_11_ = v___y_31_;
goto _start;
}
else
{
lean_dec_ref(v_arg_33_);
lean_dec(v_pos_6_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
return v___x_35_;
}
}
case 10:
{
lean_object* v_expr_39_; 
v_expr_39_ = lean_ctor_get(v_e_5_, 1);
lean_inc_ref(v_expr_39_);
lean_dec_ref_known(v_e_5_, 2);
v_e_5_ = v_expr_39_;
v_positions_7_ = v___y_27_;
v_a_8_ = v___y_28_;
v_a_9_ = v___y_29_;
v_a_10_ = v___y_30_;
v_a_11_ = v___y_31_;
goto _start;
}
case 11:
{
lean_object* v_struct_41_; lean_object* v___x_42_; 
v_struct_41_ = lean_ctor_get(v_e_5_, 2);
lean_inc_ref(v_struct_41_);
lean_dec_ref_known(v_e_5_, 3);
v___x_42_ = l_Lean_SubExpr_Pos_pushProj(v_pos_6_);
lean_dec(v_pos_6_);
v_e_5_ = v_struct_41_;
v_pos_6_ = v___x_42_;
v_positions_7_ = v___y_27_;
v_a_8_ = v___y_28_;
v_a_9_ = v___y_29_;
v_a_10_ = v___y_30_;
v_a_11_ = v___y_31_;
goto _start;
}
case 8:
{
lean_object* v_type_44_; lean_object* v_value_45_; lean_object* v_body_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v_type_44_ = lean_ctor_get(v_e_5_, 1);
lean_inc_ref(v_type_44_);
v_value_45_ = lean_ctor_get(v_e_5_, 2);
lean_inc_ref(v_value_45_);
v_body_46_ = lean_ctor_get(v_e_5_, 3);
lean_inc_ref(v_body_46_);
lean_dec_ref_known(v_e_5_, 4);
v___x_47_ = l_Lean_SubExpr_Pos_pushLetVarType(v_pos_6_);
lean_inc_ref(v_mctx_2_);
lean_inc_ref(v_p_1_);
v___x_48_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_1_, v_mctx_2_, v_pHeadIdx_3_, v_pNumArgs_4_, v_type_44_, v___x_47_, v___y_27_, v___y_28_, v___y_29_, v___y_30_, v___y_31_);
if (lean_obj_tag(v___x_48_) == 0)
{
lean_object* v_a_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v_a_49_ = lean_ctor_get(v___x_48_, 0);
lean_inc(v_a_49_);
lean_dec_ref_known(v___x_48_, 1);
v___x_50_ = l_Lean_SubExpr_Pos_pushLetValue(v_pos_6_);
lean_inc_ref(v_mctx_2_);
lean_inc_ref(v_p_1_);
v___x_51_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_1_, v_mctx_2_, v_pHeadIdx_3_, v_pNumArgs_4_, v_value_45_, v___x_50_, v_a_49_, v___y_28_, v___y_29_, v___y_30_, v___y_31_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v_a_52_; lean_object* v___x_53_; 
v_a_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc(v_a_52_);
lean_dec_ref_known(v___x_51_, 1);
v___x_53_ = l_Lean_SubExpr_Pos_pushLetBody(v_pos_6_);
lean_dec(v_pos_6_);
v_e_5_ = v_body_46_;
v_pos_6_ = v___x_53_;
v_positions_7_ = v_a_52_;
v_a_8_ = v___y_28_;
v_a_9_ = v___y_29_;
v_a_10_ = v___y_30_;
v_a_11_ = v___y_31_;
goto _start;
}
else
{
lean_dec_ref(v_body_46_);
lean_dec(v_pos_6_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
return v___x_51_;
}
}
else
{
lean_dec_ref(v_body_46_);
lean_dec_ref(v_value_45_);
lean_dec(v_pos_6_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
return v___x_48_;
}
}
case 6:
{
lean_object* v_binderType_55_; lean_object* v_body_56_; 
v_binderType_55_ = lean_ctor_get(v_e_5_, 1);
lean_inc_ref(v_binderType_55_);
v_body_56_ = lean_ctor_get(v_e_5_, 2);
lean_inc_ref(v_body_56_);
lean_dec_ref_known(v_e_5_, 3);
v_binderType_14_ = v_binderType_55_;
v_body_15_ = v_body_56_;
v___y_16_ = v___y_27_;
v___y_17_ = v___y_28_;
v___y_18_ = v___y_29_;
v___y_19_ = v___y_30_;
v___y_20_ = v___y_31_;
goto v___jp_13_;
}
case 7:
{
lean_object* v_binderType_57_; lean_object* v_body_58_; 
v_binderType_57_ = lean_ctor_get(v_e_5_, 1);
lean_inc_ref(v_binderType_57_);
v_body_58_ = lean_ctor_get(v_e_5_, 2);
lean_inc_ref(v_body_58_);
lean_dec_ref_known(v_e_5_, 3);
v_binderType_14_ = v_binderType_57_;
v_body_15_ = v_body_58_;
v___y_16_ = v___y_27_;
v___y_17_ = v___y_28_;
v___y_18_ = v___y_29_;
v___y_19_ = v___y_30_;
v___y_20_ = v___y_31_;
goto v___jp_13_;
}
default: 
{
lean_object* v___x_59_; 
lean_dec(v_pos_6_);
lean_dec_ref(v_e_5_);
lean_dec_ref(v_mctx_2_);
lean_dec_ref(v_p_1_);
v___x_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_59_, 0, v___y_27_);
return v___x_59_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions_visit___boxed(lean_object* v_p_91_, lean_object* v_mctx_92_, lean_object* v_pHeadIdx_93_, lean_object* v_pNumArgs_94_, lean_object* v_e_95_, lean_object* v_pos_96_, lean_object* v_positions_97_, lean_object* v_a_98_, lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_91_, v_mctx_92_, v_pHeadIdx_93_, v_pNumArgs_94_, v_e_95_, v_pos_96_, v_positions_97_, v_a_98_, v_a_99_, v_a_100_, v_a_101_);
lean_dec(v_a_101_);
lean_dec_ref(v_a_100_);
lean_dec(v_a_99_);
lean_dec_ref(v_a_98_);
lean_dec(v_pNumArgs_94_);
lean_dec(v_pHeadIdx_93_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions(lean_object* v_p_106_, lean_object* v_e_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_113_; lean_object* v_mctx_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_113_ = lean_st_ref_get(v_a_109_);
v_mctx_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc_ref(v_mctx_114_);
lean_dec(v___x_113_);
lean_inc_ref(v_p_106_);
v___x_115_ = l_Lean_Expr_toHeadIndex(v_p_106_);
v___x_116_ = l_Lean_Expr_headNumArgs(v_p_106_);
v___x_117_ = l_Lean_SubExpr_Pos_root;
v___x_118_ = ((lean_object*)(lp_mathlib_Lean_Meta_kabstractPositions___closed__0));
v___x_119_ = lp_mathlib_Lean_Meta_kabstractPositions_visit(v_p_106_, v_mctx_114_, v___x_115_, v___x_116_, v_e_107_, v___x_117_, v___x_118_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
lean_dec(v___x_116_);
lean_dec(v___x_115_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractPositions___boxed(lean_object* v_p_120_, lean_object* v_e_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Lean_Meta_kabstractPositions(v_p_120_, v_e_121_, v_a_122_, v_a_123_, v_a_124_, v_a_125_);
lean_dec(v_a_125_);
lean_dec_ref(v_a_124_);
lean_dec(v_a_123_);
lean_dec_ref(v_a_122_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2(lean_object* v_msg_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___f_135_; lean_object* v___x_756__overap_136_; lean_object* v___x_137_; 
v___f_135_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___closed__0));
v___x_756__overap_136_ = lean_panic_fn_borrowed(v___f_135_, v_msg_129_);
lean_inc(v___y_133_);
lean_inc_ref(v___y_132_);
lean_inc(v___y_131_);
lean_inc_ref(v___y_130_);
v___x_137_ = lean_apply_5(v___x_756__overap_136_, v___y_130_, v___y_131_, v___y_132_, v___y_133_, lean_box(0));
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2___boxed(lean_object* v_msg_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2(v_msg_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
lean_dec(v___y_140_);
lean_dec_ref(v___y_139_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6(lean_object* v_xs_145_, lean_object* v_v_146_, lean_object* v_i_147_){
_start:
{
lean_object* v___x_148_; uint8_t v___x_149_; 
v___x_148_ = lean_array_get_size(v_xs_145_);
v___x_149_ = lean_nat_dec_lt(v_i_147_, v___x_148_);
if (v___x_149_ == 0)
{
lean_object* v___x_150_; 
lean_dec(v_i_147_);
v___x_150_ = lean_box(0);
return v___x_150_;
}
else
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = lean_array_fget_borrowed(v_xs_145_, v_i_147_);
v___x_152_ = lean_nat_dec_eq(v___x_151_, v_v_146_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = lean_unsigned_to_nat(1u);
v___x_154_ = lean_nat_add(v_i_147_, v___x_153_);
lean_dec(v_i_147_);
v_i_147_ = v___x_154_;
goto _start;
}
else
{
lean_object* v___x_156_; 
v___x_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_156_, 0, v_i_147_);
return v___x_156_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6___boxed(lean_object* v_xs_157_, lean_object* v_v_158_, lean_object* v_i_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6(v_xs_157_, v_v_158_, v_i_159_);
lean_dec(v_v_158_);
lean_dec_ref(v_xs_157_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3(lean_object* v_xs_161_, lean_object* v_v_162_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = lean_unsigned_to_nat(0u);
v___x_164_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3_spec__6(v_xs_161_, v_v_162_, v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3___boxed(lean_object* v_xs_165_, lean_object* v_v_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3(v_xs_165_, v_v_166_);
lean_dec(v_v_166_);
lean_dec_ref(v_xs_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1(lean_object* v_xs_168_, lean_object* v_v_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1_spec__3(v_xs_168_, v_v_169_);
if (lean_obj_tag(v___x_170_) == 0)
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
else
{
lean_object* v_val_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
v_val_172_ = lean_ctor_get(v___x_170_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_170_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_val_172_);
lean_dec(v___x_170_);
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
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_val_172_);
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
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1___boxed(lean_object* v_xs_180_, lean_object* v_v_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1(v_xs_180_, v_v_181_);
lean_dec(v_v_181_);
lean_dec_ref(v_xs_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4(lean_object* v_msgData_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v___x_189_; lean_object* v_env_190_; lean_object* v___x_191_; lean_object* v_mctx_192_; lean_object* v_lctx_193_; lean_object* v_options_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_189_ = lean_st_ref_get(v___y_187_);
v_env_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc_ref(v_env_190_);
lean_dec(v___x_189_);
v___x_191_ = lean_st_ref_get(v___y_185_);
v_mctx_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc_ref(v_mctx_192_);
lean_dec(v___x_191_);
v_lctx_193_ = lean_ctor_get(v___y_184_, 2);
v_options_194_ = lean_ctor_get(v___y_186_, 2);
lean_inc_ref(v_options_194_);
lean_inc_ref(v_lctx_193_);
v___x_195_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_195_, 0, v_env_190_);
lean_ctor_set(v___x_195_, 1, v_mctx_192_);
lean_ctor_set(v___x_195_, 2, v_lctx_193_);
lean_ctor_set(v___x_195_, 3, v_options_194_);
v___x_196_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, v_msgData_183_);
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_msgData_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4(v_msgData_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
lean_dec(v___y_200_);
lean_dec_ref(v___y_199_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(lean_object* v_msg_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v_ref_211_; lean_object* v___x_212_; lean_object* v_a_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_221_; 
v_ref_211_ = lean_ctor_get(v___y_208_, 5);
v___x_212_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2_spec__4(v_msg_205_, v___y_206_, v___y_207_, v___y_208_, v___y_209_);
v_a_213_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_221_ == 0)
{
v___x_215_ = v___x_212_;
v_isShared_216_ = v_isSharedCheck_221_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_a_213_);
lean_dec(v___x_212_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_221_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v___x_217_; lean_object* v___x_219_; 
lean_inc(v_ref_211_);
v___x_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_217_, 0, v_ref_211_);
lean_ctor_set(v___x_217_, 1, v_a_213_);
if (v_isShared_216_ == 0)
{
lean_ctor_set_tag(v___x_215_, 1);
lean_ctor_set(v___x_215_, 0, v___x_217_);
v___x_219_ = v___x_215_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v___x_217_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_msg_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v_msg_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec(v___y_224_);
lean_dec_ref(v___y_223_);
return v_res_228_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__0));
v___x_231_ = l_Lean_stringToMessageData(v___x_230_);
return v___x_231_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_233_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__2));
v___x_234_ = l_Lean_stringToMessageData(v___x_233_);
return v___x_234_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__4));
v___x_237_ = l_Lean_stringToMessageData(v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0(lean_object* v_e_238_, lean_object* v_n_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_){
_start:
{
lean_object* v_e_246_; lean_object* v_c_247_; lean_object* v___x_258_; uint8_t v___x_259_; 
v___x_258_ = lean_unsigned_to_nat(3u);
v___x_259_ = lean_nat_dec_eq(v_n_239_, v___x_258_);
if (v___x_259_ == 0)
{
lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_260_ = lean_unsigned_to_nat(0u);
v___x_261_ = lean_nat_dec_eq(v_n_239_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_262_ = lean_unsigned_to_nat(1u);
v___x_263_ = lean_nat_dec_eq(v_n_239_, v___x_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_264_ = lean_unsigned_to_nat(2u);
v___x_265_ = lean_nat_dec_eq(v_n_239_, v___x_264_);
if (v___x_265_ == 0)
{
if (lean_obj_tag(v_e_238_) == 10)
{
lean_object* v_expr_266_; 
v_expr_266_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_expr_266_);
lean_dec_ref_known(v_e_238_, 2);
v_e_238_ = v_expr_266_;
goto _start;
}
else
{
v_e_246_ = v_e_238_;
v_c_247_ = v_n_239_;
goto v___jp_245_;
}
}
else
{
lean_dec(v_n_239_);
switch(lean_obj_tag(v_e_238_))
{
case 8:
{
lean_object* v_body_268_; lean_object* v___x_269_; 
v_body_268_ = lean_ctor_get(v_e_238_, 3);
lean_inc_ref(v_body_268_);
lean_dec_ref_known(v_e_238_, 4);
v___x_269_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_269_, 0, v_body_268_);
return v___x_269_;
}
case 10:
{
lean_object* v_expr_270_; 
v_expr_270_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_expr_270_);
lean_dec_ref_known(v_e_238_, 2);
v_e_238_ = v_expr_270_;
v_n_239_ = v___x_264_;
goto _start;
}
default: 
{
v_e_246_ = v_e_238_;
v_c_247_ = v___x_264_;
goto v___jp_245_;
}
}
}
}
else
{
lean_dec(v_n_239_);
switch(lean_obj_tag(v_e_238_))
{
case 5:
{
lean_object* v_arg_272_; lean_object* v___x_273_; 
v_arg_272_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_arg_272_);
lean_dec_ref_known(v_e_238_, 2);
v___x_273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_273_, 0, v_arg_272_);
return v___x_273_;
}
case 6:
{
lean_object* v_body_274_; lean_object* v___x_275_; 
v_body_274_ = lean_ctor_get(v_e_238_, 2);
lean_inc_ref(v_body_274_);
lean_dec_ref_known(v_e_238_, 3);
v___x_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_275_, 0, v_body_274_);
return v___x_275_;
}
case 7:
{
lean_object* v_body_276_; lean_object* v___x_277_; 
v_body_276_ = lean_ctor_get(v_e_238_, 2);
lean_inc_ref(v_body_276_);
lean_dec_ref_known(v_e_238_, 3);
v___x_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_277_, 0, v_body_276_);
return v___x_277_;
}
case 8:
{
lean_object* v_value_278_; lean_object* v___x_279_; 
v_value_278_ = lean_ctor_get(v_e_238_, 2);
lean_inc_ref(v_value_278_);
lean_dec_ref_known(v_e_238_, 4);
v___x_279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_279_, 0, v_value_278_);
return v___x_279_;
}
case 10:
{
lean_object* v_expr_280_; 
v_expr_280_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_expr_280_);
lean_dec_ref_known(v_e_238_, 2);
v_e_238_ = v_expr_280_;
v_n_239_ = v___x_262_;
goto _start;
}
default: 
{
v_e_246_ = v_e_238_;
v_c_247_ = v___x_262_;
goto v___jp_245_;
}
}
}
}
else
{
lean_dec(v_n_239_);
switch(lean_obj_tag(v_e_238_))
{
case 5:
{
lean_object* v_fn_282_; lean_object* v___x_283_; 
v_fn_282_ = lean_ctor_get(v_e_238_, 0);
lean_inc_ref(v_fn_282_);
lean_dec_ref_known(v_e_238_, 2);
v___x_283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_283_, 0, v_fn_282_);
return v___x_283_;
}
case 6:
{
lean_object* v_binderType_284_; lean_object* v___x_285_; 
v_binderType_284_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_binderType_284_);
lean_dec_ref_known(v_e_238_, 3);
v___x_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_285_, 0, v_binderType_284_);
return v___x_285_;
}
case 7:
{
lean_object* v_binderType_286_; lean_object* v___x_287_; 
v_binderType_286_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_binderType_286_);
lean_dec_ref_known(v_e_238_, 3);
v___x_287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_287_, 0, v_binderType_286_);
return v___x_287_;
}
case 8:
{
lean_object* v_type_288_; lean_object* v___x_289_; 
v_type_288_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_type_288_);
lean_dec_ref_known(v_e_238_, 4);
v___x_289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_289_, 0, v_type_288_);
return v___x_289_;
}
case 11:
{
lean_object* v_struct_290_; lean_object* v___x_291_; 
v_struct_290_ = lean_ctor_get(v_e_238_, 2);
lean_inc_ref(v_struct_290_);
lean_dec_ref_known(v_e_238_, 3);
v___x_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_291_, 0, v_struct_290_);
return v___x_291_;
}
case 10:
{
lean_object* v_expr_292_; 
v_expr_292_ = lean_ctor_get(v_e_238_, 1);
lean_inc_ref(v_expr_292_);
lean_dec_ref_known(v_e_238_, 2);
v_e_238_ = v_expr_292_;
v_n_239_ = v___x_260_;
goto _start;
}
default: 
{
v_e_246_ = v_e_238_;
v_c_247_ = v___x_260_;
goto v___jp_245_;
}
}
}
}
else
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
lean_dec(v_n_239_);
v___x_294_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__5);
v___x_295_ = l_Lean_MessageData_ofExpr(v_e_238_);
v___x_296_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_294_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
v___x_297_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v___x_296_, v___y_240_, v___y_241_, v___y_242_, v___y_243_);
return v___x_297_;
}
v___jp_245_:
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_248_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__1);
v___x_249_ = l_Nat_reprFast(v_c_247_);
v___x_250_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
v___x_251_ = l_Lean_MessageData_ofFormat(v___x_250_);
v___x_252_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_248_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
v___x_253_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3);
v___x_254_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_252_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = l_Lean_MessageData_ofExpr(v_e_246_);
v___x_256_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_254_);
lean_ctor_set(v___x_256_, 1, v___x_255_);
v___x_257_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v___x_256_, v___y_240_, v___y_241_, v___y_242_, v___y_243_);
return v___x_257_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___boxed(lean_object* v_e_298_, lean_object* v_n_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0(v_e_298_, v_n_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1(lean_object* v_f_306_, lean_object* v_init_307_, lean_object* v_p_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
uint8_t v___x_314_; 
v___x_314_ = l_Lean_SubExpr_Pos_isRoot(v_p_308_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_315_ = l_Lean_SubExpr_Pos_tail(v_p_308_);
lean_inc_ref(v_f_306_);
v___x_316_ = lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1(v_f_306_, v_init_307_, v___x_315_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
lean_dec(v___x_315_);
if (lean_obj_tag(v___x_316_) == 0)
{
lean_object* v_a_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v_a_317_ = lean_ctor_get(v___x_316_, 0);
lean_inc(v_a_317_);
lean_dec_ref_known(v___x_316_, 1);
v___x_318_ = l_Lean_SubExpr_Pos_head(v_p_308_);
lean_inc(v___y_312_);
lean_inc_ref(v___y_311_);
lean_inc(v___y_310_);
lean_inc_ref(v___y_309_);
v___x_319_ = lean_apply_7(v_f_306_, v_a_317_, v___x_318_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, lean_box(0));
return v___x_319_;
}
else
{
lean_dec_ref(v_f_306_);
return v___x_316_;
}
}
else
{
lean_object* v___x_320_; 
lean_dec_ref(v_f_306_);
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v_init_307_);
return v___x_320_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1___boxed(lean_object* v_f_321_, lean_object* v_init_322_, lean_object* v_p_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1(v_f_321_, v_init_322_, v_p_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v_p_323_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0(lean_object* v_p_331_, lean_object* v_root_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = ((lean_object*)(lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___closed__0));
v___x_339_ = lp_mathlib_Lean_SubExpr_Pos_foldlM___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__1(v___x_338_, v_root_332_, v_p_331_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0___boxed(lean_object* v_p_340_, lean_object* v_root_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0(v_p_340_, v_root_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
lean_dec(v_p_340_);
return v_res_347_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_351_ = ((lean_object*)(lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__2));
v___x_352_ = lean_unsigned_to_nat(39u);
v___x_353_ = lean_unsigned_to_nat(77u);
v___x_354_ = ((lean_object*)(lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__1));
v___x_355_ = ((lean_object*)(lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__0));
v___x_356_ = l_mkPanicMessageWithDecl(v___x_355_, v___x_354_, v___x_353_, v___x_352_, v___x_351_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr(lean_object* v_e_357_, lean_object* v_pos_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_){
_start:
{
lean_object* v___x_364_; 
lean_inc_ref(v_e_357_);
v___x_364_ = lp_mathlib_Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0(v_pos_358_, v_e_357_, v_a_359_, v_a_360_, v_a_361_, v_a_362_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_411_; 
v_a_365_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_411_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_411_ == 0)
{
v___x_367_ = v___x_364_;
v_isShared_368_ = v_isSharedCheck_411_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_364_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_411_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
uint8_t v___x_369_; 
v___x_369_ = l_Lean_Expr_hasLooseBVars(v_a_365_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; 
lean_del_object(v___x_367_);
lean_inc(v_a_365_);
v___x_370_ = lp_mathlib_Lean_Meta_kabstractPositions(v_a_365_, v_e_357_, v_a_359_, v_a_360_, v_a_361_, v_a_362_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_398_; 
v_a_371_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_398_ == 0)
{
v___x_373_ = v___x_370_;
v_isShared_374_ = v_isSharedCheck_398_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_370_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_398_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___y_376_; lean_object* v___x_382_; 
v___x_382_ = lp_mathlib_Array_idxOf_x3f___at___00Lean_Meta_viewKAbstractSubExpr_spec__1(v_a_371_, v_pos_358_);
if (lean_obj_tag(v___x_382_) == 1)
{
lean_object* v_val_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_395_; 
v_val_383_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_395_ == 0)
{
v___x_385_ = v___x_382_;
v_isShared_386_ = v_isSharedCheck_395_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_val_383_);
lean_dec(v___x_382_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_395_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; lean_object* v___x_388_; uint8_t v___x_389_; 
v___x_387_ = lean_array_get_size(v_a_371_);
lean_dec(v_a_371_);
v___x_388_ = lean_unsigned_to_nat(1u);
v___x_389_ = lean_nat_dec_eq(v___x_387_, v___x_388_);
if (v___x_389_ == 0)
{
lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_390_ = lean_nat_add(v_val_383_, v___x_388_);
lean_dec(v_val_383_);
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 0, v___x_390_);
v___x_392_ = v___x_385_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
v___y_376_ = v___x_392_;
goto v___jp_375_;
}
}
else
{
lean_object* v___x_394_; 
lean_del_object(v___x_385_);
lean_dec(v_val_383_);
v___x_394_ = lean_box(0);
v___y_376_ = v___x_394_;
goto v___jp_375_;
}
}
}
else
{
lean_object* v___x_396_; lean_object* v___x_397_; 
lean_dec(v___x_382_);
lean_del_object(v___x_373_);
lean_dec(v_a_371_);
lean_dec(v_a_365_);
v___x_396_ = lean_obj_once(&lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3, &lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3_once, _init_lp_mathlib_Lean_Meta_viewKAbstractSubExpr___closed__3);
v___x_397_ = lp_mathlib_panic___at___00Lean_Meta_viewKAbstractSubExpr_spec__2(v___x_396_, v_a_359_, v_a_360_, v_a_361_, v_a_362_);
return v___x_397_;
}
v___jp_375_:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_380_; 
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v_a_365_);
lean_ctor_set(v___x_377_, 1, v___y_376_);
v___x_378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_378_);
v___x_380_ = v___x_373_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v___x_378_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_dec(v_a_365_);
v_a_399_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_370_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_370_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
else
{
lean_object* v___x_407_; lean_object* v___x_409_; 
lean_dec(v_a_365_);
lean_dec_ref(v_e_357_);
v___x_407_ = lean_box(0);
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 0, v___x_407_);
v___x_409_ = v___x_367_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v___x_407_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
else
{
lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_419_; 
lean_dec_ref(v_e_357_);
v_a_412_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_419_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_419_ == 0)
{
v___x_414_ = v___x_364_;
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_364_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_417_; 
if (v_isShared_415_ == 0)
{
v___x_417_ = v___x_414_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v_a_412_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___boxed(lean_object* v_e_420_, lean_object* v_pos_421_, lean_object* v_a_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_, lean_object* v_a_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_Lean_Meta_viewKAbstractSubExpr(v_e_420_, v_pos_421_, v_a_422_, v_a_423_, v_a_424_, v_a_425_);
lean_dec(v_a_425_);
lean_dec_ref(v_a_424_);
lean_dec(v_a_423_);
lean_dec_ref(v_a_422_);
lean_dec(v_pos_421_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2(lean_object* v_00_u03b1_428_, lean_object* v_msg_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v_msg_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b1_436_, lean_object* v_msg_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2(v_00_u03b1_436_, v_msg_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
lean_dec(v___y_441_);
lean_dec_ref(v___y_440_);
lean_dec(v___y_439_);
lean_dec_ref(v___y_438_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0(lean_object* v_fvar_444_, lean_object* v_x_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_451_, 0, v_fvar_444_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0___boxed(lean_object* v_fvar_452_, lean_object* v_x_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0(v_fvar_452_, v_x_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_);
lean_dec(v___y_457_);
lean_dec_ref(v___y_456_);
lean_dec(v___y_455_);
lean_dec_ref(v___y_454_);
lean_dec_ref(v_x_453_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__4(lean_object* v_msg_460_){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; 
v___x_461_ = l_Lean_instInhabitedExpr;
v___x_462_ = lean_panic_fn_borrowed(v___x_461_, v_msg_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0(lean_object* v_k_463_, lean_object* v_b_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; 
lean_inc(v___y_468_);
lean_inc_ref(v___y_467_);
lean_inc(v___y_466_);
lean_inc_ref(v___y_465_);
v___x_470_ = lean_apply_6(v_k_463_, v_b_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_, lean_box(0));
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0___boxed(lean_object* v_k_471_, lean_object* v_b_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0(v_k_471_, v_b_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(lean_object* v_name_479_, lean_object* v_type_480_, lean_object* v_val_481_, lean_object* v_k_482_, uint8_t v_nondep_483_, uint8_t v_kind_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v___f_490_; lean_object* v___x_491_; 
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_490_, 0, v_k_482_);
v___x_491_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_479_, v_type_480_, v_val_481_, v___f_490_, v_nondep_483_, v_kind_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
if (lean_obj_tag(v___x_491_) == 0)
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
v_a_492_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_491_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_491_);
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
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_507_; 
v_a_500_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_507_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_507_ == 0)
{
v___x_502_ = v___x_491_;
v_isShared_503_ = v_isSharedCheck_507_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_a_500_);
lean_dec(v___x_491_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg___boxed(lean_object* v_name_508_, lean_object* v_type_509_, lean_object* v_val_510_, lean_object* v_k_511_, lean_object* v_nondep_512_, lean_object* v_kind_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_){
_start:
{
uint8_t v_nondep_boxed_519_; uint8_t v_kind_boxed_520_; lean_object* v_res_521_; 
v_nondep_boxed_519_ = lean_unbox(v_nondep_512_);
v_kind_boxed_520_ = lean_unbox(v_kind_513_);
v_res_521_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_508_, v_type_509_, v_val_510_, v_k_511_, v_nondep_boxed_519_, v_kind_boxed_520_, v___y_514_, v___y_515_, v___y_516_, v___y_517_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
lean_dec(v___y_515_);
lean_dec_ref(v___y_514_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0(lean_object* v_k_522_, uint8_t v_usedLetOnly_523_, lean_object* v_x_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
lean_object* v___x_530_; 
lean_inc(v___y_528_);
lean_inc_ref(v___y_527_);
lean_inc(v___y_526_);
lean_inc_ref(v___y_525_);
lean_inc_ref(v_x_524_);
v___x_530_ = lean_apply_6(v_k_522_, v_x_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_, lean_box(0));
if (lean_obj_tag(v___x_530_) == 0)
{
lean_object* v_a_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; uint8_t v___x_535_; uint8_t v___x_536_; lean_object* v___x_537_; 
v_a_531_ = lean_ctor_get(v___x_530_, 0);
lean_inc(v_a_531_);
lean_dec_ref_known(v___x_530_, 1);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_mk_empty_array_with_capacity(v___x_532_);
v___x_534_ = lean_array_push(v___x_533_, v_x_524_);
v___x_535_ = 0;
v___x_536_ = 1;
v___x_537_ = l_Lean_Meta_mkLetFVars(v___x_534_, v_a_531_, v_usedLetOnly_523_, v___x_535_, v___x_536_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec_ref(v___x_534_);
return v___x_537_;
}
else
{
lean_dec_ref(v_x_524_);
return v___x_530_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0___boxed(lean_object* v_k_538_, lean_object* v_usedLetOnly_539_, lean_object* v_x_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
uint8_t v_usedLetOnly_boxed_546_; lean_object* v_res_547_; 
v_usedLetOnly_boxed_546_ = lean_unbox(v_usedLetOnly_539_);
v_res_547_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0(v_k_538_, v_usedLetOnly_boxed_546_, v_x_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5(lean_object* v_name_548_, lean_object* v_type_549_, lean_object* v_val_550_, lean_object* v_k_551_, uint8_t v_nondep_552_, uint8_t v_kind_553_, uint8_t v_usedLetOnly_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
lean_object* v___x_560_; lean_object* v___f_561_; lean_object* v___x_562_; 
v___x_560_ = lean_box(v_usedLetOnly_554_);
v___f_561_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___lam__0___boxed), 8, 2);
lean_closure_set(v___f_561_, 0, v_k_551_);
lean_closure_set(v___f_561_, 1, v___x_560_);
v___x_562_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_548_, v_type_549_, v_val_550_, v___f_561_, v_nondep_552_, v_kind_553_, v___y_555_, v___y_556_, v___y_557_, v___y_558_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_name_563_, lean_object* v_type_564_, lean_object* v_val_565_, lean_object* v_k_566_, lean_object* v_nondep_567_, lean_object* v_kind_568_, lean_object* v_usedLetOnly_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_){
_start:
{
uint8_t v_nondep_boxed_575_; uint8_t v_kind_boxed_576_; uint8_t v_usedLetOnly_boxed_577_; lean_object* v_res_578_; 
v_nondep_boxed_575_ = lean_unbox(v_nondep_567_);
v_kind_boxed_576_ = lean_unbox(v_kind_568_);
v_usedLetOnly_boxed_577_ = lean_unbox(v_usedLetOnly_569_);
v_res_578_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5(v_name_563_, v_type_564_, v_val_565_, v_k_566_, v_nondep_boxed_575_, v_kind_boxed_576_, v_usedLetOnly_boxed_577_, v___y_570_, v___y_571_, v___y_572_, v___y_573_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_572_);
lean_dec(v___y_571_);
lean_dec_ref(v___y_570_);
return v_res_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0(lean_object* v_body_579_, lean_object* v_g_580_, lean_object* v_x_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_587_ = lean_expr_instantiate1(v_body_579_, v_x_581_);
lean_inc(v___y_585_);
lean_inc_ref(v___y_584_);
lean_inc(v___y_583_);
lean_inc_ref(v___y_582_);
v___x_588_ = lean_apply_6(v_g_580_, v___x_587_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, lean_box(0));
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v_body_589_, lean_object* v_g_590_, lean_object* v_x_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0(v_body_589_, v_g_590_, v_x_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec_ref(v_x_591_);
lean_dec_ref(v_body_589_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0(lean_object* v___x_598_, lean_object* v_body_599_, lean_object* v_g_600_, lean_object* v_b_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
v___x_607_ = lean_mk_empty_array_with_capacity(v___x_598_);
v___x_608_ = lean_array_push(v___x_607_, v_b_601_);
v___x_609_ = lean_expr_instantiate_rev(v_body_599_, v___x_608_);
lean_inc(v___y_605_);
lean_inc_ref(v___y_604_);
lean_inc(v___y_603_);
lean_inc_ref(v___y_602_);
v___x_610_ = lean_apply_6(v_g_600_, v___x_609_, v___y_602_, v___y_603_, v___y_604_, v___y_605_, lean_box(0));
if (lean_obj_tag(v___x_610_) == 0)
{
lean_object* v_a_611_; uint8_t v___x_612_; uint8_t v___x_613_; uint8_t v___x_614_; lean_object* v___x_615_; 
v_a_611_ = lean_ctor_get(v___x_610_, 0);
lean_inc(v_a_611_);
lean_dec_ref_known(v___x_610_, 1);
v___x_612_ = 0;
v___x_613_ = 1;
v___x_614_ = 1;
v___x_615_ = l_Lean_Meta_mkForallFVars(v___x_608_, v_a_611_, v___x_612_, v___x_613_, v___x_613_, v___x_614_, v___y_602_, v___y_603_, v___y_604_, v___y_605_);
lean_dec_ref(v___x_608_);
return v___x_615_;
}
else
{
lean_dec_ref(v___x_608_);
return v___x_610_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0___boxed(lean_object* v___x_616_, lean_object* v_body_617_, lean_object* v_g_618_, lean_object* v_b_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0(v___x_616_, v_body_617_, v_g_618_, v_b_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
lean_dec(v___y_621_);
lean_dec_ref(v___y_620_);
lean_dec_ref(v_body_617_);
lean_dec(v___x_616_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9(lean_object* v_body_626_, lean_object* v_g_627_, lean_object* v_name_628_, uint8_t v_bi_629_, lean_object* v_type_630_, uint8_t v_kind_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_){
_start:
{
lean_object* v___x_637_; lean_object* v___f_638_; lean_object* v___x_639_; 
v___x_637_ = lean_unsigned_to_nat(1u);
v___f_638_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___lam__0___boxed), 9, 3);
lean_closure_set(v___f_638_, 0, v___x_637_);
lean_closure_set(v___f_638_, 1, v_body_626_);
lean_closure_set(v___f_638_, 2, v_g_627_);
v___x_639_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_628_, v_bi_629_, v_type_630_, v___f_638_, v_kind_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_);
if (lean_obj_tag(v___x_639_) == 0)
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
v_a_640_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_639_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_639_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
else
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
v_a_648_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_639_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_639_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9___boxed(lean_object* v_body_656_, lean_object* v_g_657_, lean_object* v_name_658_, lean_object* v_bi_659_, lean_object* v_type_660_, lean_object* v_kind_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_){
_start:
{
uint8_t v_bi_boxed_667_; uint8_t v_kind_boxed_668_; lean_object* v_res_669_; 
v_bi_boxed_667_ = lean_unbox(v_bi_659_);
v_kind_boxed_668_ = lean_unbox(v_kind_661_);
v_res_669_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9(v_body_656_, v_g_657_, v_name_658_, v_bi_boxed_667_, v_type_660_, v_kind_boxed_668_, v___y_662_, v___y_663_, v___y_664_, v___y_665_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0(lean_object* v___x_670_, lean_object* v_body_671_, lean_object* v_g_672_, lean_object* v_b_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_){
_start:
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_679_ = lean_mk_empty_array_with_capacity(v___x_670_);
v___x_680_ = lean_array_push(v___x_679_, v_b_673_);
v___x_681_ = lean_expr_instantiate_rev(v_body_671_, v___x_680_);
lean_inc(v___y_677_);
lean_inc_ref(v___y_676_);
lean_inc(v___y_675_);
lean_inc_ref(v___y_674_);
v___x_682_ = lean_apply_6(v_g_672_, v___x_681_, v___y_674_, v___y_675_, v___y_676_, v___y_677_, lean_box(0));
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v_a_683_; uint8_t v___x_684_; uint8_t v___x_685_; uint8_t v___x_686_; lean_object* v___x_687_; 
v_a_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc(v_a_683_);
lean_dec_ref_known(v___x_682_, 1);
v___x_684_ = 0;
v___x_685_ = 1;
v___x_686_ = 1;
v___x_687_ = l_Lean_Meta_mkLambdaFVars(v___x_680_, v_a_683_, v___x_684_, v___x_685_, v___x_684_, v___x_685_, v___x_686_, v___y_674_, v___y_675_, v___y_676_, v___y_677_);
lean_dec_ref(v___x_680_);
return v___x_687_;
}
else
{
lean_dec_ref(v___x_680_);
return v___x_682_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0___boxed(lean_object* v___x_688_, lean_object* v_body_689_, lean_object* v_g_690_, lean_object* v_b_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_){
_start:
{
lean_object* v_res_697_; 
v_res_697_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0(v___x_688_, v_body_689_, v_g_690_, v_b_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
lean_dec(v___y_695_);
lean_dec_ref(v___y_694_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
lean_dec_ref(v_body_689_);
lean_dec(v___x_688_);
return v_res_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8(lean_object* v_body_698_, lean_object* v_g_699_, lean_object* v_name_700_, uint8_t v_bi_701_, lean_object* v_type_702_, uint8_t v_kind_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v___x_709_; lean_object* v___f_710_; lean_object* v___x_711_; 
v___x_709_ = lean_unsigned_to_nat(1u);
v___f_710_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___lam__0___boxed), 9, 3);
lean_closure_set(v___f_710_, 0, v___x_709_);
lean_closure_set(v___f_710_, 1, v_body_698_);
lean_closure_set(v___f_710_, 2, v_g_699_);
v___x_711_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_700_, v_bi_701_, v_type_702_, v___f_710_, v_kind_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_object* v_a_712_; lean_object* v___x_714_; uint8_t v_isShared_715_; uint8_t v_isSharedCheck_719_; 
v_a_712_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_719_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_719_ == 0)
{
v___x_714_ = v___x_711_;
v_isShared_715_ = v_isSharedCheck_719_;
goto v_resetjp_713_;
}
else
{
lean_inc(v_a_712_);
lean_dec(v___x_711_);
v___x_714_ = lean_box(0);
v_isShared_715_ = v_isSharedCheck_719_;
goto v_resetjp_713_;
}
v_resetjp_713_:
{
lean_object* v___x_717_; 
if (v_isShared_715_ == 0)
{
v___x_717_ = v___x_714_;
goto v_reusejp_716_;
}
else
{
lean_object* v_reuseFailAlloc_718_; 
v_reuseFailAlloc_718_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_718_, 0, v_a_712_);
v___x_717_ = v_reuseFailAlloc_718_;
goto v_reusejp_716_;
}
v_reusejp_716_:
{
return v___x_717_;
}
}
}
else
{
lean_object* v_a_720_; lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_727_; 
v_a_720_ = lean_ctor_get(v___x_711_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v___x_711_);
if (v_isSharedCheck_727_ == 0)
{
v___x_722_ = v___x_711_;
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
else
{
lean_inc(v_a_720_);
lean_dec(v___x_711_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v___x_725_; 
if (v_isShared_723_ == 0)
{
v___x_725_ = v___x_722_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_a_720_);
v___x_725_ = v_reuseFailAlloc_726_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
return v___x_725_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8___boxed(lean_object* v_body_728_, lean_object* v_g_729_, lean_object* v_name_730_, lean_object* v_bi_731_, lean_object* v_type_732_, lean_object* v_kind_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
uint8_t v_bi_boxed_739_; uint8_t v_kind_boxed_740_; lean_object* v_res_741_; 
v_bi_boxed_739_ = lean_unbox(v_bi_731_);
v_kind_boxed_740_ = lean_unbox(v_kind_733_);
v_res_741_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8(v_body_728_, v_g_729_, v_name_730_, v_bi_boxed_739_, v_type_732_, v_kind_boxed_740_, v___y_734_, v___y_735_, v___y_736_, v___y_737_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
return v_res_741_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
v___x_745_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__2));
v___x_746_ = lean_unsigned_to_nat(17u);
v___x_747_ = lean_unsigned_to_nat(1885u);
v___x_748_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__1));
v___x_749_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__0));
v___x_750_ = l_mkPanicMessageWithDecl(v___x_749_, v___x_748_, v___x_747_, v___x_746_, v___x_745_);
return v___x_750_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5(void){
_start:
{
lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_752_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__4));
v___x_753_ = l_Lean_stringToMessageData(v___x_752_);
return v___x_753_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7(void){
_start:
{
lean_object* v___x_755_; lean_object* v___x_756_; 
v___x_755_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__6));
v___x_756_ = l_Lean_stringToMessageData(v___x_755_);
return v___x_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1(lean_object* v_g_757_, lean_object* v_n_758_, lean_object* v_e_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_){
_start:
{
lean_object* v_n_766_; lean_object* v_a_767_; lean_object* v_c_797_; lean_object* v_e_798_; lean_object* v___x_809_; uint8_t v___x_810_; 
v___x_809_ = lean_unsigned_to_nat(0u);
v___x_810_ = lean_nat_dec_eq(v_n_758_, v___x_809_);
if (v___x_810_ == 0)
{
lean_object* v___x_811_; uint8_t v___x_812_; 
v___x_811_ = lean_unsigned_to_nat(1u);
v___x_812_ = lean_nat_dec_eq(v_n_758_, v___x_811_);
if (v___x_812_ == 0)
{
lean_object* v___x_813_; uint8_t v___x_814_; 
v___x_813_ = lean_unsigned_to_nat(2u);
v___x_814_ = lean_nat_dec_eq(v_n_758_, v___x_813_);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; uint8_t v___x_816_; 
v___x_815_ = lean_unsigned_to_nat(3u);
v___x_816_ = lean_nat_dec_eq(v_n_758_, v___x_815_);
if (v___x_816_ == 0)
{
if (lean_obj_tag(v_e_759_) == 10)
{
lean_object* v_expr_817_; 
v_expr_817_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_expr_817_);
v_n_766_ = v_n_758_;
v_a_767_ = v_expr_817_;
goto v___jp_765_;
}
else
{
lean_dec_ref(v_g_757_);
v_c_797_ = v_n_758_;
v_e_798_ = v_e_759_;
goto v___jp_796_;
}
}
else
{
lean_dec(v_n_758_);
if (lean_obj_tag(v_e_759_) == 10)
{
lean_object* v_expr_818_; 
v_expr_818_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_expr_818_);
v_n_766_ = v___x_815_;
v_a_767_ = v_expr_818_;
goto v___jp_765_;
}
else
{
lean_object* v___x_819_; lean_object* v___x_820_; 
lean_dec_ref(v_e_759_);
lean_dec_ref(v_g_757_);
v___x_819_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__7);
v___x_820_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v___x_819_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
return v___x_820_;
}
}
}
else
{
lean_dec(v_n_758_);
switch(lean_obj_tag(v_e_759_))
{
case 8:
{
lean_object* v_declName_821_; lean_object* v_type_822_; lean_object* v_value_823_; lean_object* v_body_824_; uint8_t v_nondep_825_; lean_object* v___f_826_; uint8_t v___x_827_; lean_object* v___x_828_; 
v_declName_821_ = lean_ctor_get(v_e_759_, 0);
lean_inc(v_declName_821_);
v_type_822_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_type_822_);
v_value_823_ = lean_ctor_get(v_e_759_, 2);
lean_inc_ref(v_value_823_);
v_body_824_ = lean_ctor_get(v_e_759_, 3);
lean_inc_ref(v_body_824_);
v_nondep_825_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_759_, 4);
v___f_826_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___lam__0___boxed), 8, 2);
lean_closure_set(v___f_826_, 0, v_body_824_);
lean_closure_set(v___f_826_, 1, v_g_757_);
v___x_827_ = 0;
v___x_828_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5(v_declName_821_, v_type_822_, v_value_823_, v___f_826_, v_nondep_825_, v___x_827_, v___x_812_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
return v___x_828_;
}
case 10:
{
lean_object* v_expr_829_; 
v_expr_829_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_expr_829_);
v_n_766_ = v___x_813_;
v_a_767_ = v_expr_829_;
goto v___jp_765_;
}
default: 
{
lean_dec_ref(v_g_757_);
v_c_797_ = v___x_813_;
v_e_798_ = v_e_759_;
goto v___jp_796_;
}
}
}
}
else
{
lean_dec(v_n_758_);
switch(lean_obj_tag(v_e_759_))
{
case 5:
{
lean_object* v_fn_830_; lean_object* v_arg_831_; lean_object* v___x_832_; 
v_fn_830_ = lean_ctor_get(v_e_759_, 0);
v_arg_831_ = lean_ctor_get(v_e_759_, 1);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_arg_831_);
v___x_832_ = lean_apply_6(v_g_757_, v_arg_831_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_832_) == 0)
{
lean_object* v_a_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_851_; 
v_a_833_ = lean_ctor_get(v___x_832_, 0);
v_isSharedCheck_851_ = !lean_is_exclusive(v___x_832_);
if (v_isSharedCheck_851_ == 0)
{
v___x_835_ = v___x_832_;
v_isShared_836_ = v_isSharedCheck_851_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_a_833_);
lean_dec(v___x_832_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_851_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
uint8_t v___y_838_; size_t v___x_846_; uint8_t v___x_847_; 
v___x_846_ = lean_ptr_addr(v_fn_830_);
v___x_847_ = lean_usize_dec_eq(v___x_846_, v___x_846_);
if (v___x_847_ == 0)
{
v___y_838_ = v___x_847_;
goto v___jp_837_;
}
else
{
size_t v___x_848_; size_t v___x_849_; uint8_t v___x_850_; 
v___x_848_ = lean_ptr_addr(v_arg_831_);
v___x_849_ = lean_ptr_addr(v_a_833_);
v___x_850_ = lean_usize_dec_eq(v___x_848_, v___x_849_);
v___y_838_ = v___x_850_;
goto v___jp_837_;
}
v___jp_837_:
{
if (v___y_838_ == 0)
{
lean_object* v___x_839_; lean_object* v___x_841_; 
lean_inc_ref(v_fn_830_);
lean_dec_ref_known(v_e_759_, 2);
v___x_839_ = l_Lean_Expr_app___override(v_fn_830_, v_a_833_);
if (v_isShared_836_ == 0)
{
lean_ctor_set(v___x_835_, 0, v___x_839_);
v___x_841_ = v___x_835_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v___x_839_);
v___x_841_ = v_reuseFailAlloc_842_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
return v___x_841_;
}
}
else
{
lean_object* v___x_844_; 
lean_dec(v_a_833_);
if (v_isShared_836_ == 0)
{
lean_ctor_set(v___x_835_, 0, v_e_759_);
v___x_844_ = v___x_835_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_e_759_);
v___x_844_ = v_reuseFailAlloc_845_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
return v___x_844_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 2);
return v___x_832_;
}
}
case 6:
{
lean_object* v_binderName_852_; lean_object* v_binderType_853_; lean_object* v_body_854_; uint8_t v_binderInfo_855_; uint8_t v___x_856_; lean_object* v___x_857_; 
v_binderName_852_ = lean_ctor_get(v_e_759_, 0);
lean_inc(v_binderName_852_);
v_binderType_853_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_binderType_853_);
v_body_854_ = lean_ctor_get(v_e_759_, 2);
lean_inc_ref(v_body_854_);
v_binderInfo_855_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_759_, 3);
v___x_856_ = 0;
v___x_857_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__8(v_body_854_, v_g_757_, v_binderName_852_, v_binderInfo_855_, v_binderType_853_, v___x_856_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
return v___x_857_;
}
case 7:
{
lean_object* v_binderName_858_; lean_object* v_binderType_859_; lean_object* v_body_860_; uint8_t v_binderInfo_861_; uint8_t v___x_862_; lean_object* v___x_863_; 
v_binderName_858_ = lean_ctor_get(v_e_759_, 0);
lean_inc(v_binderName_858_);
v_binderType_859_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_binderType_859_);
v_body_860_ = lean_ctor_get(v_e_759_, 2);
lean_inc_ref(v_body_860_);
v_binderInfo_861_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_759_, 3);
v___x_862_ = 0;
v___x_863_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__9(v_body_860_, v_g_757_, v_binderName_858_, v_binderInfo_861_, v_binderType_859_, v___x_862_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
return v___x_863_;
}
case 8:
{
lean_object* v_declName_864_; lean_object* v_type_865_; lean_object* v_value_866_; lean_object* v_body_867_; uint8_t v_nondep_868_; lean_object* v___x_869_; 
v_declName_864_ = lean_ctor_get(v_e_759_, 0);
v_type_865_ = lean_ctor_get(v_e_759_, 1);
v_value_866_ = lean_ctor_get(v_e_759_, 2);
v_body_867_ = lean_ctor_get(v_e_759_, 3);
v_nondep_868_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*4 + 8);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_value_866_);
v___x_869_ = lean_apply_6(v_g_757_, v_value_866_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_869_) == 0)
{
lean_object* v_a_870_; lean_object* v___x_872_; uint8_t v_isShared_873_; uint8_t v_isSharedCheck_894_; 
v_a_870_ = lean_ctor_get(v___x_869_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_869_);
if (v_isSharedCheck_894_ == 0)
{
v___x_872_ = v___x_869_;
v_isShared_873_ = v_isSharedCheck_894_;
goto v_resetjp_871_;
}
else
{
lean_inc(v_a_870_);
lean_dec(v___x_869_);
v___x_872_ = lean_box(0);
v_isShared_873_ = v_isSharedCheck_894_;
goto v_resetjp_871_;
}
v_resetjp_871_:
{
uint8_t v___y_875_; size_t v___x_889_; uint8_t v___x_890_; 
v___x_889_ = lean_ptr_addr(v_type_865_);
v___x_890_ = lean_usize_dec_eq(v___x_889_, v___x_889_);
if (v___x_890_ == 0)
{
v___y_875_ = v___x_890_;
goto v___jp_874_;
}
else
{
size_t v___x_891_; size_t v___x_892_; uint8_t v___x_893_; 
v___x_891_ = lean_ptr_addr(v_value_866_);
v___x_892_ = lean_ptr_addr(v_a_870_);
v___x_893_ = lean_usize_dec_eq(v___x_891_, v___x_892_);
v___y_875_ = v___x_893_;
goto v___jp_874_;
}
v___jp_874_:
{
if (v___y_875_ == 0)
{
lean_object* v___x_876_; lean_object* v___x_878_; 
lean_inc_ref(v_body_867_);
lean_inc_ref(v_type_865_);
lean_inc(v_declName_864_);
lean_dec_ref_known(v_e_759_, 4);
v___x_876_ = l_Lean_Expr_letE___override(v_declName_864_, v_type_865_, v_a_870_, v_body_867_, v_nondep_868_);
if (v_isShared_873_ == 0)
{
lean_ctor_set(v___x_872_, 0, v___x_876_);
v___x_878_ = v___x_872_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_879_, 0, v___x_876_);
v___x_878_ = v_reuseFailAlloc_879_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
return v___x_878_;
}
}
else
{
size_t v___x_880_; uint8_t v___x_881_; 
v___x_880_ = lean_ptr_addr(v_body_867_);
v___x_881_ = lean_usize_dec_eq(v___x_880_, v___x_880_);
if (v___x_881_ == 0)
{
lean_object* v___x_882_; lean_object* v___x_884_; 
lean_inc_ref(v_body_867_);
lean_inc_ref(v_type_865_);
lean_inc(v_declName_864_);
lean_dec_ref_known(v_e_759_, 4);
v___x_882_ = l_Lean_Expr_letE___override(v_declName_864_, v_type_865_, v_a_870_, v_body_867_, v_nondep_868_);
if (v_isShared_873_ == 0)
{
lean_ctor_set(v___x_872_, 0, v___x_882_);
v___x_884_ = v___x_872_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v___x_882_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
else
{
lean_object* v___x_887_; 
lean_dec(v_a_870_);
if (v_isShared_873_ == 0)
{
lean_ctor_set(v___x_872_, 0, v_e_759_);
v___x_887_ = v___x_872_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_888_; 
v_reuseFailAlloc_888_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_888_, 0, v_e_759_);
v___x_887_ = v_reuseFailAlloc_888_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
return v___x_887_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 4);
return v___x_869_;
}
}
case 10:
{
lean_object* v_expr_895_; 
v_expr_895_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_expr_895_);
v_n_766_ = v___x_811_;
v_a_767_ = v_expr_895_;
goto v___jp_765_;
}
default: 
{
lean_dec_ref(v_g_757_);
v_c_797_ = v___x_811_;
v_e_798_ = v_e_759_;
goto v___jp_796_;
}
}
}
}
else
{
lean_dec(v_n_758_);
switch(lean_obj_tag(v_e_759_))
{
case 5:
{
lean_object* v_fn_896_; lean_object* v_arg_897_; lean_object* v___x_898_; 
v_fn_896_ = lean_ctor_get(v_e_759_, 0);
v_arg_897_ = lean_ctor_get(v_e_759_, 1);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_fn_896_);
v___x_898_ = lean_apply_6(v_g_757_, v_fn_896_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_898_) == 0)
{
lean_object* v_a_899_; lean_object* v___x_901_; uint8_t v_isShared_902_; uint8_t v_isSharedCheck_917_; 
v_a_899_ = lean_ctor_get(v___x_898_, 0);
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_898_);
if (v_isSharedCheck_917_ == 0)
{
v___x_901_ = v___x_898_;
v_isShared_902_ = v_isSharedCheck_917_;
goto v_resetjp_900_;
}
else
{
lean_inc(v_a_899_);
lean_dec(v___x_898_);
v___x_901_ = lean_box(0);
v_isShared_902_ = v_isSharedCheck_917_;
goto v_resetjp_900_;
}
v_resetjp_900_:
{
uint8_t v___y_904_; size_t v___x_912_; size_t v___x_913_; uint8_t v___x_914_; 
v___x_912_ = lean_ptr_addr(v_fn_896_);
v___x_913_ = lean_ptr_addr(v_a_899_);
v___x_914_ = lean_usize_dec_eq(v___x_912_, v___x_913_);
if (v___x_914_ == 0)
{
v___y_904_ = v___x_914_;
goto v___jp_903_;
}
else
{
size_t v___x_915_; uint8_t v___x_916_; 
v___x_915_ = lean_ptr_addr(v_arg_897_);
v___x_916_ = lean_usize_dec_eq(v___x_915_, v___x_915_);
v___y_904_ = v___x_916_;
goto v___jp_903_;
}
v___jp_903_:
{
if (v___y_904_ == 0)
{
lean_object* v___x_905_; lean_object* v___x_907_; 
lean_inc_ref(v_arg_897_);
lean_dec_ref_known(v_e_759_, 2);
v___x_905_ = l_Lean_Expr_app___override(v_a_899_, v_arg_897_);
if (v_isShared_902_ == 0)
{
lean_ctor_set(v___x_901_, 0, v___x_905_);
v___x_907_ = v___x_901_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v___x_905_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
else
{
lean_object* v___x_910_; 
lean_dec(v_a_899_);
if (v_isShared_902_ == 0)
{
lean_ctor_set(v___x_901_, 0, v_e_759_);
v___x_910_ = v___x_901_;
goto v_reusejp_909_;
}
else
{
lean_object* v_reuseFailAlloc_911_; 
v_reuseFailAlloc_911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_911_, 0, v_e_759_);
v___x_910_ = v_reuseFailAlloc_911_;
goto v_reusejp_909_;
}
v_reusejp_909_:
{
return v___x_910_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 2);
return v___x_898_;
}
}
case 6:
{
lean_object* v_binderName_918_; lean_object* v_binderType_919_; lean_object* v_body_920_; uint8_t v_binderInfo_921_; lean_object* v___x_922_; 
v_binderName_918_ = lean_ctor_get(v_e_759_, 0);
v_binderType_919_ = lean_ctor_get(v_e_759_, 1);
v_body_920_ = lean_ctor_get(v_e_759_, 2);
v_binderInfo_921_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*3 + 8);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_binderType_919_);
v___x_922_ = lean_apply_6(v_g_757_, v_binderType_919_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_922_) == 0)
{
lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_946_; 
v_a_923_ = lean_ctor_get(v___x_922_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_946_ == 0)
{
v___x_925_ = v___x_922_;
v_isShared_926_ = v_isSharedCheck_946_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_922_);
v___x_925_ = lean_box(0);
v_isShared_926_ = v_isSharedCheck_946_;
goto v_resetjp_924_;
}
v_resetjp_924_:
{
uint8_t v___y_928_; size_t v___x_941_; size_t v___x_942_; uint8_t v___x_943_; 
v___x_941_ = lean_ptr_addr(v_binderType_919_);
v___x_942_ = lean_ptr_addr(v_a_923_);
v___x_943_ = lean_usize_dec_eq(v___x_941_, v___x_942_);
if (v___x_943_ == 0)
{
v___y_928_ = v___x_943_;
goto v___jp_927_;
}
else
{
size_t v___x_944_; uint8_t v___x_945_; 
v___x_944_ = lean_ptr_addr(v_body_920_);
v___x_945_ = lean_usize_dec_eq(v___x_944_, v___x_944_);
v___y_928_ = v___x_945_;
goto v___jp_927_;
}
v___jp_927_:
{
if (v___y_928_ == 0)
{
lean_object* v___x_929_; lean_object* v___x_931_; 
lean_inc_ref(v_body_920_);
lean_inc(v_binderName_918_);
lean_dec_ref_known(v_e_759_, 3);
v___x_929_ = l_Lean_Expr_lam___override(v_binderName_918_, v_a_923_, v_body_920_, v_binderInfo_921_);
if (v_isShared_926_ == 0)
{
lean_ctor_set(v___x_925_, 0, v___x_929_);
v___x_931_ = v___x_925_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v___x_929_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
else
{
uint8_t v___x_933_; 
v___x_933_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_921_, v_binderInfo_921_);
if (v___x_933_ == 0)
{
lean_object* v___x_934_; lean_object* v___x_936_; 
lean_inc_ref(v_body_920_);
lean_inc(v_binderName_918_);
lean_dec_ref_known(v_e_759_, 3);
v___x_934_ = l_Lean_Expr_lam___override(v_binderName_918_, v_a_923_, v_body_920_, v_binderInfo_921_);
if (v_isShared_926_ == 0)
{
lean_ctor_set(v___x_925_, 0, v___x_934_);
v___x_936_ = v___x_925_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v___x_934_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
else
{
lean_object* v___x_939_; 
lean_dec(v_a_923_);
if (v_isShared_926_ == 0)
{
lean_ctor_set(v___x_925_, 0, v_e_759_);
v___x_939_ = v___x_925_;
goto v_reusejp_938_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_e_759_);
v___x_939_ = v_reuseFailAlloc_940_;
goto v_reusejp_938_;
}
v_reusejp_938_:
{
return v___x_939_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 3);
return v___x_922_;
}
}
case 7:
{
lean_object* v_binderName_947_; lean_object* v_binderType_948_; lean_object* v_body_949_; uint8_t v_binderInfo_950_; lean_object* v___x_951_; 
v_binderName_947_ = lean_ctor_get(v_e_759_, 0);
v_binderType_948_ = lean_ctor_get(v_e_759_, 1);
v_body_949_ = lean_ctor_get(v_e_759_, 2);
v_binderInfo_950_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*3 + 8);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_binderType_948_);
v___x_951_ = lean_apply_6(v_g_757_, v_binderType_948_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_951_) == 0)
{
lean_object* v_a_952_; lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_975_; 
v_a_952_ = lean_ctor_get(v___x_951_, 0);
v_isSharedCheck_975_ = !lean_is_exclusive(v___x_951_);
if (v_isSharedCheck_975_ == 0)
{
v___x_954_ = v___x_951_;
v_isShared_955_ = v_isSharedCheck_975_;
goto v_resetjp_953_;
}
else
{
lean_inc(v_a_952_);
lean_dec(v___x_951_);
v___x_954_ = lean_box(0);
v_isShared_955_ = v_isSharedCheck_975_;
goto v_resetjp_953_;
}
v_resetjp_953_:
{
uint8_t v___y_957_; size_t v___x_970_; size_t v___x_971_; uint8_t v___x_972_; 
v___x_970_ = lean_ptr_addr(v_binderType_948_);
v___x_971_ = lean_ptr_addr(v_a_952_);
v___x_972_ = lean_usize_dec_eq(v___x_970_, v___x_971_);
if (v___x_972_ == 0)
{
v___y_957_ = v___x_972_;
goto v___jp_956_;
}
else
{
size_t v___x_973_; uint8_t v___x_974_; 
v___x_973_ = lean_ptr_addr(v_body_949_);
v___x_974_ = lean_usize_dec_eq(v___x_973_, v___x_973_);
v___y_957_ = v___x_974_;
goto v___jp_956_;
}
v___jp_956_:
{
if (v___y_957_ == 0)
{
lean_object* v___x_958_; lean_object* v___x_960_; 
lean_inc_ref(v_body_949_);
lean_inc(v_binderName_947_);
lean_dec_ref_known(v_e_759_, 3);
v___x_958_ = l_Lean_Expr_forallE___override(v_binderName_947_, v_a_952_, v_body_949_, v_binderInfo_950_);
if (v_isShared_955_ == 0)
{
lean_ctor_set(v___x_954_, 0, v___x_958_);
v___x_960_ = v___x_954_;
goto v_reusejp_959_;
}
else
{
lean_object* v_reuseFailAlloc_961_; 
v_reuseFailAlloc_961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_961_, 0, v___x_958_);
v___x_960_ = v_reuseFailAlloc_961_;
goto v_reusejp_959_;
}
v_reusejp_959_:
{
return v___x_960_;
}
}
else
{
uint8_t v___x_962_; 
v___x_962_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_950_, v_binderInfo_950_);
if (v___x_962_ == 0)
{
lean_object* v___x_963_; lean_object* v___x_965_; 
lean_inc_ref(v_body_949_);
lean_inc(v_binderName_947_);
lean_dec_ref_known(v_e_759_, 3);
v___x_963_ = l_Lean_Expr_forallE___override(v_binderName_947_, v_a_952_, v_body_949_, v_binderInfo_950_);
if (v_isShared_955_ == 0)
{
lean_ctor_set(v___x_954_, 0, v___x_963_);
v___x_965_ = v___x_954_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v___x_963_);
v___x_965_ = v_reuseFailAlloc_966_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
return v___x_965_;
}
}
else
{
lean_object* v___x_968_; 
lean_dec(v_a_952_);
if (v_isShared_955_ == 0)
{
lean_ctor_set(v___x_954_, 0, v_e_759_);
v___x_968_ = v___x_954_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v_e_759_);
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
else
{
lean_dec_ref_known(v_e_759_, 3);
return v___x_951_;
}
}
case 8:
{
lean_object* v_declName_976_; lean_object* v_type_977_; lean_object* v_value_978_; lean_object* v_body_979_; uint8_t v_nondep_980_; lean_object* v___x_981_; 
v_declName_976_ = lean_ctor_get(v_e_759_, 0);
v_type_977_ = lean_ctor_get(v_e_759_, 1);
v_value_978_ = lean_ctor_get(v_e_759_, 2);
v_body_979_ = lean_ctor_get(v_e_759_, 3);
v_nondep_980_ = lean_ctor_get_uint8(v_e_759_, sizeof(void*)*4 + 8);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_type_977_);
v___x_981_ = lean_apply_6(v_g_757_, v_type_977_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_981_) == 0)
{
lean_object* v_a_982_; lean_object* v___x_984_; uint8_t v_isShared_985_; uint8_t v_isSharedCheck_1006_; 
v_a_982_ = lean_ctor_get(v___x_981_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_981_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_984_ = v___x_981_;
v_isShared_985_ = v_isSharedCheck_1006_;
goto v_resetjp_983_;
}
else
{
lean_inc(v_a_982_);
lean_dec(v___x_981_);
v___x_984_ = lean_box(0);
v_isShared_985_ = v_isSharedCheck_1006_;
goto v_resetjp_983_;
}
v_resetjp_983_:
{
uint8_t v___y_987_; size_t v___x_1001_; size_t v___x_1002_; uint8_t v___x_1003_; 
v___x_1001_ = lean_ptr_addr(v_type_977_);
v___x_1002_ = lean_ptr_addr(v_a_982_);
v___x_1003_ = lean_usize_dec_eq(v___x_1001_, v___x_1002_);
if (v___x_1003_ == 0)
{
v___y_987_ = v___x_1003_;
goto v___jp_986_;
}
else
{
size_t v___x_1004_; uint8_t v___x_1005_; 
v___x_1004_ = lean_ptr_addr(v_value_978_);
v___x_1005_ = lean_usize_dec_eq(v___x_1004_, v___x_1004_);
v___y_987_ = v___x_1005_;
goto v___jp_986_;
}
v___jp_986_:
{
if (v___y_987_ == 0)
{
lean_object* v___x_988_; lean_object* v___x_990_; 
lean_inc_ref(v_body_979_);
lean_inc_ref(v_value_978_);
lean_inc(v_declName_976_);
lean_dec_ref_known(v_e_759_, 4);
v___x_988_ = l_Lean_Expr_letE___override(v_declName_976_, v_a_982_, v_value_978_, v_body_979_, v_nondep_980_);
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 0, v___x_988_);
v___x_990_ = v___x_984_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v___x_988_);
v___x_990_ = v_reuseFailAlloc_991_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
return v___x_990_;
}
}
else
{
size_t v___x_992_; uint8_t v___x_993_; 
v___x_992_ = lean_ptr_addr(v_body_979_);
v___x_993_ = lean_usize_dec_eq(v___x_992_, v___x_992_);
if (v___x_993_ == 0)
{
lean_object* v___x_994_; lean_object* v___x_996_; 
lean_inc_ref(v_body_979_);
lean_inc_ref(v_value_978_);
lean_inc(v_declName_976_);
lean_dec_ref_known(v_e_759_, 4);
v___x_994_ = l_Lean_Expr_letE___override(v_declName_976_, v_a_982_, v_value_978_, v_body_979_, v_nondep_980_);
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 0, v___x_994_);
v___x_996_ = v___x_984_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v___x_994_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
else
{
lean_object* v___x_999_; 
lean_dec(v_a_982_);
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 0, v_e_759_);
v___x_999_ = v___x_984_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v_e_759_);
v___x_999_ = v_reuseFailAlloc_1000_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
return v___x_999_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 4);
return v___x_981_;
}
}
case 11:
{
lean_object* v_typeName_1007_; lean_object* v_idx_1008_; lean_object* v_struct_1009_; lean_object* v___x_1010_; 
v_typeName_1007_ = lean_ctor_get(v_e_759_, 0);
v_idx_1008_ = lean_ctor_get(v_e_759_, 1);
v_struct_1009_ = lean_ctor_get(v_e_759_, 2);
lean_inc(v___y_763_);
lean_inc_ref(v___y_762_);
lean_inc(v___y_761_);
lean_inc_ref(v___y_760_);
lean_inc_ref(v_struct_1009_);
v___x_1010_ = lean_apply_6(v_g_757_, v_struct_1009_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, lean_box(0));
if (lean_obj_tag(v___x_1010_) == 0)
{
lean_object* v_a_1011_; lean_object* v___x_1013_; uint8_t v_isShared_1014_; uint8_t v_isSharedCheck_1025_; 
v_a_1011_ = lean_ctor_get(v___x_1010_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_1010_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1013_ = v___x_1010_;
v_isShared_1014_ = v_isSharedCheck_1025_;
goto v_resetjp_1012_;
}
else
{
lean_inc(v_a_1011_);
lean_dec(v___x_1010_);
v___x_1013_ = lean_box(0);
v_isShared_1014_ = v_isSharedCheck_1025_;
goto v_resetjp_1012_;
}
v_resetjp_1012_:
{
size_t v___x_1015_; size_t v___x_1016_; uint8_t v___x_1017_; 
v___x_1015_ = lean_ptr_addr(v_struct_1009_);
v___x_1016_ = lean_ptr_addr(v_a_1011_);
v___x_1017_ = lean_usize_dec_eq(v___x_1015_, v___x_1016_);
if (v___x_1017_ == 0)
{
lean_object* v___x_1018_; lean_object* v___x_1020_; 
lean_inc(v_idx_1008_);
lean_inc(v_typeName_1007_);
lean_dec_ref_known(v_e_759_, 3);
v___x_1018_ = l_Lean_Expr_proj___override(v_typeName_1007_, v_idx_1008_, v_a_1011_);
if (v_isShared_1014_ == 0)
{
lean_ctor_set(v___x_1013_, 0, v___x_1018_);
v___x_1020_ = v___x_1013_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1021_; 
v_reuseFailAlloc_1021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1021_, 0, v___x_1018_);
v___x_1020_ = v_reuseFailAlloc_1021_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
return v___x_1020_;
}
}
else
{
lean_object* v___x_1023_; 
lean_dec(v_a_1011_);
if (v_isShared_1014_ == 0)
{
lean_ctor_set(v___x_1013_, 0, v_e_759_);
v___x_1023_ = v___x_1013_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v_e_759_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
}
}
else
{
lean_dec_ref_known(v_e_759_, 3);
return v___x_1010_;
}
}
case 10:
{
lean_object* v_expr_1026_; 
v_expr_1026_ = lean_ctor_get(v_e_759_, 1);
lean_inc_ref(v_expr_1026_);
v_n_766_ = v___x_809_;
v_a_767_ = v_expr_1026_;
goto v___jp_765_;
}
default: 
{
lean_dec_ref(v_g_757_);
v_c_797_ = v___x_809_;
v_e_798_ = v_e_759_;
goto v___jp_796_;
}
}
}
v___jp_765_:
{
lean_object* v___x_768_; 
v___x_768_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1(v_g_757_, v_n_766_, v_a_767_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
if (lean_obj_tag(v___x_768_) == 0)
{
if (lean_obj_tag(v_e_759_) == 10)
{
lean_object* v_a_769_; lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_785_; 
v_a_769_ = lean_ctor_get(v___x_768_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_785_ == 0)
{
v___x_771_ = v___x_768_;
v_isShared_772_ = v_isSharedCheck_785_;
goto v_resetjp_770_;
}
else
{
lean_inc(v_a_769_);
lean_dec(v___x_768_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_785_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v_data_773_; lean_object* v_expr_774_; size_t v___x_775_; size_t v___x_776_; uint8_t v___x_777_; 
v_data_773_ = lean_ctor_get(v_e_759_, 0);
v_expr_774_ = lean_ctor_get(v_e_759_, 1);
v___x_775_ = lean_ptr_addr(v_expr_774_);
v___x_776_ = lean_ptr_addr(v_a_769_);
v___x_777_ = lean_usize_dec_eq(v___x_775_, v___x_776_);
if (v___x_777_ == 0)
{
lean_object* v___x_778_; lean_object* v___x_780_; 
lean_inc(v_data_773_);
lean_dec_ref_known(v_e_759_, 2);
v___x_778_ = l_Lean_Expr_mdata___override(v_data_773_, v_a_769_);
if (v_isShared_772_ == 0)
{
lean_ctor_set(v___x_771_, 0, v___x_778_);
v___x_780_ = v___x_771_;
goto v_reusejp_779_;
}
else
{
lean_object* v_reuseFailAlloc_781_; 
v_reuseFailAlloc_781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_781_, 0, v___x_778_);
v___x_780_ = v_reuseFailAlloc_781_;
goto v_reusejp_779_;
}
v_reusejp_779_:
{
return v___x_780_;
}
}
else
{
lean_object* v___x_783_; 
lean_dec(v_a_769_);
if (v_isShared_772_ == 0)
{
lean_ctor_set(v___x_771_, 0, v_e_759_);
v___x_783_ = v___x_771_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v_e_759_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
else
{
lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_794_; 
lean_dec_ref(v_e_759_);
v_isSharedCheck_794_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_794_ == 0)
{
lean_object* v_unused_795_; 
v_unused_795_ = lean_ctor_get(v___x_768_, 0);
lean_dec(v_unused_795_);
v___x_787_ = v___x_768_;
v_isShared_788_ = v_isSharedCheck_794_;
goto v_resetjp_786_;
}
else
{
lean_dec(v___x_768_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_794_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_792_; 
v___x_789_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__3);
v___x_790_ = lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__4(v___x_789_);
if (v_isShared_788_ == 0)
{
lean_ctor_set(v___x_787_, 0, v___x_790_);
v___x_792_ = v___x_787_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v___x_790_);
v___x_792_ = v_reuseFailAlloc_793_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
return v___x_792_;
}
}
}
}
else
{
lean_dec_ref(v_e_759_);
return v___x_768_;
}
}
v___jp_796_:
{
lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_799_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___closed__5);
v___x_800_ = l_Nat_reprFast(v_c_797_);
v___x_801_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
v___x_802_ = l_Lean_MessageData_ofFormat(v___x_801_);
v___x_803_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_803_, 0, v___x_799_);
lean_ctor_set(v___x_803_, 1, v___x_802_);
v___x_804_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0___closed__3);
v___x_805_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_805_, 0, v___x_803_);
lean_ctor_set(v___x_805_, 1, v___x_804_);
v___x_806_ = l_Lean_MessageData_ofExpr(v_e_798_);
v___x_807_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_807_, 0, v___x_805_);
lean_ctor_set(v___x_807_, 1, v___x_806_);
v___x_808_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Core_viewCoordRaw___at___00Lean_Core_viewSubexpr___at___00Lean_Meta_viewKAbstractSubExpr_spec__0_spec__0_spec__2___redArg(v___x_807_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
return v___x_808_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1___boxed(lean_object* v_g_1027_, lean_object* v_n_1028_, lean_object* v_e_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1(v_g_1027_, v_n_1028_, v_e_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_);
lean_dec(v___y_1033_);
lean_dec_ref(v___y_1032_);
lean_dec(v___y_1031_);
lean_dec_ref(v___y_1030_);
return v_res_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0___boxed(lean_object* v_g_1036_, lean_object* v_x_1037_, lean_object* v_x_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_){
_start:
{
lean_object* v_res_1044_; 
v_res_1044_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0(v_g_1036_, v_x_1037_, v_x_1038_, v___y_1039_, v___y_1040_, v___y_1041_, v___y_1042_);
lean_dec(v___y_1042_);
lean_dec_ref(v___y_1041_);
lean_dec(v___y_1040_);
lean_dec_ref(v___y_1039_);
return v_res_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0(lean_object* v_g_1045_, lean_object* v_x_1046_, lean_object* v_x_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_){
_start:
{
if (lean_obj_tag(v_x_1046_) == 0)
{
lean_object* v___x_1053_; 
lean_inc(v___y_1051_);
lean_inc_ref(v___y_1050_);
lean_inc(v___y_1049_);
lean_inc_ref(v___y_1048_);
v___x_1053_ = lean_apply_6(v_g_1045_, v_x_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_, lean_box(0));
return v___x_1053_;
}
else
{
lean_object* v_head_1054_; lean_object* v_tail_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v_head_1054_ = lean_ctor_get(v_x_1046_, 0);
lean_inc(v_head_1054_);
v_tail_1055_ = lean_ctor_get(v_x_1046_, 1);
lean_inc(v_tail_1055_);
lean_dec_ref_known(v_x_1046_, 2);
v___x_1056_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0___boxed), 8, 2);
lean_closure_set(v___x_1056_, 0, v_g_1045_);
lean_closure_set(v___x_1056_, 1, v_tail_1055_);
v___x_1057_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1(v___x_1056_, v_head_1054_, v_x_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_);
return v___x_1057_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0(lean_object* v_replace_1058_, lean_object* v_p_1059_, lean_object* v_root_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1066_ = l_Lean_SubExpr_Pos_toArray(v_p_1059_);
v___x_1067_ = lean_array_to_list(v___x_1066_);
v___x_1068_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0(v_replace_1058_, v___x_1067_, v_root_1060_, v___y_1061_, v___y_1062_, v___y_1063_, v___y_1064_);
return v___x_1068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0___boxed(lean_object* v_replace_1069_, lean_object* v_p_1070_, lean_object* v_root_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_){
_start:
{
lean_object* v_res_1077_; 
v_res_1077_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0(v_replace_1069_, v_p_1070_, v_root_1071_, v___y_1072_, v___y_1073_, v___y_1074_, v___y_1075_);
lean_dec(v___y_1075_);
lean_dec_ref(v___y_1074_);
lean_dec(v___y_1073_);
lean_dec_ref(v___y_1072_);
lean_dec(v_p_1070_);
return v_res_1077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1(lean_object* v_pos_1078_, lean_object* v_e_1079_, lean_object* v_fvar_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_){
_start:
{
lean_object* v___f_1086_; lean_object* v___x_1087_; 
v___f_1086_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1086_, 0, v_fvar_1080_);
v___x_1087_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0(v___f_1086_, v_pos_1078_, v_e_1079_, v___y_1081_, v___y_1082_, v___y_1083_, v___y_1084_);
if (lean_obj_tag(v___x_1087_) == 0)
{
lean_object* v_a_1088_; lean_object* v___x_1089_; 
v_a_1088_ = lean_ctor_get(v___x_1087_, 0);
lean_inc(v_a_1088_);
lean_dec_ref_known(v___x_1087_, 1);
v___x_1089_ = l_Lean_Meta_isTypeCorrect(v_a_1088_, v___y_1081_, v___y_1082_, v___y_1083_, v___y_1084_);
return v___x_1089_;
}
else
{
lean_object* v_a_1090_; lean_object* v___x_1092_; uint8_t v_isShared_1093_; uint8_t v_isSharedCheck_1097_; 
v_a_1090_ = lean_ctor_get(v___x_1087_, 0);
v_isSharedCheck_1097_ = !lean_is_exclusive(v___x_1087_);
if (v_isSharedCheck_1097_ == 0)
{
v___x_1092_ = v___x_1087_;
v_isShared_1093_ = v_isSharedCheck_1097_;
goto v_resetjp_1091_;
}
else
{
lean_inc(v_a_1090_);
lean_dec(v___x_1087_);
v___x_1092_ = lean_box(0);
v_isShared_1093_ = v_isSharedCheck_1097_;
goto v_resetjp_1091_;
}
v_resetjp_1091_:
{
lean_object* v___x_1095_; 
if (v_isShared_1093_ == 0)
{
v___x_1095_ = v___x_1092_;
goto v_reusejp_1094_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v_a_1090_);
v___x_1095_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1094_;
}
v_reusejp_1094_:
{
return v___x_1095_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1___boxed(lean_object* v_pos_1098_, lean_object* v_e_1099_, lean_object* v_fvar_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
lean_object* v_res_1106_; 
v_res_1106_ = lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1(v_pos_1098_, v_e_1099_, v_fvar_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec(v___y_1102_);
lean_dec_ref(v___y_1101_);
lean_dec(v_pos_1098_);
return v_res_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg(lean_object* v_name_1107_, uint8_t v_bi_1108_, lean_object* v_type_1109_, lean_object* v_k_1110_, uint8_t v_kind_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_){
_start:
{
lean_object* v___f_1117_; lean_object* v___x_1118_; 
v___f_1117_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1117_, 0, v_k_1110_);
v___x_1118_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1107_, v_bi_1108_, v_type_1109_, v___f_1117_, v_kind_1111_, v___y_1112_, v___y_1113_, v___y_1114_, v___y_1115_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1126_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1121_ = v___x_1118_;
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1118_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1124_; 
if (v_isShared_1122_ == 0)
{
v___x_1124_ = v___x_1121_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v_a_1119_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
else
{
lean_object* v_a_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1134_; 
v_a_1127_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1134_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1129_ = v___x_1118_;
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_a_1127_);
lean_dec(v___x_1118_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1132_; 
if (v_isShared_1130_ == 0)
{
v___x_1132_ = v___x_1129_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1133_; 
v_reuseFailAlloc_1133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1133_, 0, v_a_1127_);
v___x_1132_ = v_reuseFailAlloc_1133_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
return v___x_1132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg___boxed(lean_object* v_name_1135_, lean_object* v_bi_1136_, lean_object* v_type_1137_, lean_object* v_k_1138_, lean_object* v_kind_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_){
_start:
{
uint8_t v_bi_boxed_1145_; uint8_t v_kind_boxed_1146_; lean_object* v_res_1147_; 
v_bi_boxed_1145_ = lean_unbox(v_bi_1136_);
v_kind_boxed_1146_ = lean_unbox(v_kind_1139_);
v_res_1147_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg(v_name_1135_, v_bi_boxed_1145_, v_type_1137_, v_k_1138_, v_kind_boxed_1146_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg(lean_object* v_name_1148_, lean_object* v_type_1149_, lean_object* v_k_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_){
_start:
{
uint8_t v___x_1156_; uint8_t v___x_1157_; lean_object* v___x_1158_; 
v___x_1156_ = 0;
v___x_1157_ = 0;
v___x_1158_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg(v_name_1148_, v___x_1156_, v_type_1149_, v_k_1150_, v___x_1157_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg___boxed(lean_object* v_name_1159_, lean_object* v_type_1160_, lean_object* v_k_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg(v_name_1159_, v_type_1160_, v_k_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect(lean_object* v_e_1171_, lean_object* v_subExpr_1172_, lean_object* v_pos_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_){
_start:
{
lean_object* v___x_1179_; 
lean_inc(v_a_1177_);
lean_inc_ref(v_a_1176_);
lean_inc(v_a_1175_);
lean_inc_ref(v_a_1174_);
v___x_1179_ = lean_infer_type(v_subExpr_1172_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_);
if (lean_obj_tag(v___x_1179_) == 0)
{
lean_object* v_a_1180_; lean_object* v___f_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; 
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
lean_inc(v_a_1180_);
lean_dec_ref_known(v___x_1179_, 1);
v___f_1181_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___lam__1___boxed), 8, 2);
lean_closure_set(v___f_1181_, 0, v_pos_1173_);
lean_closure_set(v___f_1181_, 1, v_e_1171_);
v___x_1182_ = ((lean_object*)(lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___closed__1));
v___x_1183_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg(v___x_1182_, v_a_1180_, v___f_1181_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_);
return v___x_1183_;
}
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
lean_dec(v_pos_1173_);
lean_dec_ref(v_e_1171_);
v_a_1184_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_1179_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1179_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1189_; 
if (v_isShared_1187_ == 0)
{
v___x_1189_ = v___x_1186_;
goto v_reusejp_1188_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_a_1184_);
v___x_1189_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1188_;
}
v_reusejp_1188_:
{
return v___x_1189_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___boxed(lean_object* v_e_1192_, lean_object* v_subExpr_1193_, lean_object* v_pos_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_, lean_object* v_a_1199_){
_start:
{
lean_object* v_res_1200_; 
v_res_1200_ = lp_mathlib_Lean_Meta_kabstractIsTypeCorrect(v_e_1192_, v_subExpr_1193_, v_pos_1194_, v_a_1195_, v_a_1196_, v_a_1197_, v_a_1198_);
lean_dec(v_a_1198_);
lean_dec_ref(v_a_1197_);
lean_dec(v_a_1196_);
lean_dec_ref(v_a_1195_);
return v_res_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2(lean_object* v_00_u03b1_1201_, lean_object* v_name_1202_, uint8_t v_bi_1203_, lean_object* v_type_1204_, lean_object* v_k_1205_, uint8_t v_kind_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_){
_start:
{
lean_object* v___x_1212_; 
v___x_1212_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___redArg(v_name_1202_, v_bi_1203_, v_type_1204_, v_k_1205_, v_kind_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2___boxed(lean_object* v_00_u03b1_1213_, lean_object* v_name_1214_, lean_object* v_bi_1215_, lean_object* v_type_1216_, lean_object* v_k_1217_, lean_object* v_kind_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_){
_start:
{
uint8_t v_bi_boxed_1224_; uint8_t v_kind_boxed_1225_; lean_object* v_res_1226_; 
v_bi_boxed_1224_ = lean_unbox(v_bi_1215_);
v_kind_boxed_1225_ = lean_unbox(v_kind_1218_);
v_res_1226_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1_spec__2(v_00_u03b1_1213_, v_name_1214_, v_bi_boxed_1224_, v_type_1216_, v_k_1217_, v_kind_boxed_1225_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_);
lean_dec(v___y_1222_);
lean_dec_ref(v___y_1221_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
return v_res_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1(lean_object* v_00_u03b1_1227_, lean_object* v_name_1228_, lean_object* v_type_1229_, lean_object* v_k_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_){
_start:
{
lean_object* v___x_1236_; 
v___x_1236_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___redArg(v_name_1228_, v_type_1229_, v_k_1230_, v___y_1231_, v___y_1232_, v___y_1233_, v___y_1234_);
return v___x_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1___boxed(lean_object* v_00_u03b1_1237_, lean_object* v_name_1238_, lean_object* v_type_1239_, lean_object* v_k_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_){
_start:
{
lean_object* v_res_1246_; 
v_res_1246_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Meta_kabstractIsTypeCorrect_spec__1(v_00_u03b1_1237_, v_name_1238_, v_type_1239_, v_k_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec(v___y_1242_);
lean_dec_ref(v___y_1241_);
return v_res_1246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6(lean_object* v_00_u03b1_1247_, lean_object* v_name_1248_, lean_object* v_type_1249_, lean_object* v_val_1250_, lean_object* v_k_1251_, uint8_t v_nondep_1252_, uint8_t v_kind_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_){
_start:
{
lean_object* v___x_1259_; 
v___x_1259_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___redArg(v_name_1248_, v_type_1249_, v_val_1250_, v_k_1251_, v_nondep_1252_, v_kind_1253_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
return v___x_1259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6___boxed(lean_object* v_00_u03b1_1260_, lean_object* v_name_1261_, lean_object* v_type_1262_, lean_object* v_val_1263_, lean_object* v_k_1264_, lean_object* v_nondep_1265_, lean_object* v_kind_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_){
_start:
{
uint8_t v_nondep_boxed_1272_; uint8_t v_kind_boxed_1273_; lean_object* v_res_1274_; 
v_nondep_boxed_1272_ = lean_unbox(v_nondep_1265_);
v_kind_boxed_1273_ = lean_unbox(v_kind_1266_);
v_res_1274_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00Lean_Meta_kabstractIsTypeCorrect_spec__0_spec__0_spec__1_spec__5_spec__6(v_00_u03b1_1260_, v_name_1261_, v_type_1262_, v_val_1263_, v_k_1264_, v_nondep_boxed_1272_, v_kind_boxed_1273_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
lean_dec(v___y_1268_);
lean_dec_ref(v___y_1267_);
return v_res_1274_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_HeadIndex(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Check(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_HeadIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_HeadIndex(uint8_t builtin);
lean_object* initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* initialize_Lean_Meta_Check(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_HeadIndex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(builtin);
}
#ifdef __cplusplus
}
#endif
