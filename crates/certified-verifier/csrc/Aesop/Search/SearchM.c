// Lean compiler output
// Module: Aesop.Search.SearchM
// Imports: public import Init public meta import Init public import Aesop.Search.Queue.Class public import Aesop.Tree.TreeM
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
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkInitialTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Queue_init_x27___redArg(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedContext_default;
lean_object* l_ReaderT_read___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNormSimpContext;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonad___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__0;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonad___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__1;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__2 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonad___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__3 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonad___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__4 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonad___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonad___closed__5 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonad___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadRef___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__0 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadRef___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadRef___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__1 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadRef___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadRef___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__2 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadRef___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadRef___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__3 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadRef___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__4;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__5;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__6;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__7;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__8;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instMonadRef___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instMonadRef___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5;
static const lean_string_object lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_SearchM_instInhabited___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instInhabited___closed__0 = (const lean_object*)&lp_aesop_Aesop_SearchM_instInhabited___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadStateState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___closed__0 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadStateState___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___closed__1 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadStateState___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___closed__2 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_SearchM_instMonadStateState___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__0_value),((lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__1_value),((lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___closed__3 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadStateState___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_SearchM_instMonadLiftTreeM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___closed__0 = (const lean_object*)&lp_aesop_Aesop_SearchM_instMonadLiftTreeM___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_SearchM_run___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_SearchM_run___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_SearchM_run___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_SearchM_run___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__3;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_SearchM_run___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__5_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__12 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_SearchM_run___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__12_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__13 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_SearchM_run___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__13_value),((lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__14_value;
static const lean_string_object lp_aesop_Aesop_SearchM_run___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "aesop: internal error: root mvar cluster does not contain exactly one goal."};
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__15 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__15_value;
static lean_once_cell_t lp_aesop_Aesop_SearchM_run___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__16;
static const lean_closure_object lp_aesop_Aesop_SearchM_run___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_SearchM_run___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_SearchM_run___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; uint8_t v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_3_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__0));
v___x_4_ = lean_box(0);
v___x_5_ = 0;
v___x_6_ = l_Lean_Meta_Simp_instInhabitedContext_default;
v___x_7_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v___x_7_, 0, v___x_6_);
lean_ctor_set(v___x_7_, 1, v___x_4_);
lean_ctor_set(v___x_7_, 2, v___x_3_);
lean_ctor_set_uint8(v___x_7_, sizeof(void*)*3, v___x_5_);
lean_ctor_set_uint8(v___x_7_, sizeof(void*)*3 + 1, v___x_5_);
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1, &lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default___closed__1);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNormSimpContext(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_aesop_Aesop_instInhabitedNormSimpContext_default;
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; uint8_t v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_unsigned_to_nat(0u);
v___x_12_ = 0;
v___x_13_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_13_, 0, v___x_11_);
lean_ctor_set(v___x_13_, 1, v_inst_10_);
lean_ctor_set_uint8(v___x_13_, sizeof(void*)*2, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default(lean_object* v_Q_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(v_inst_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState_default___boxed(lean_object* v_Q_18_, lean_object* v_inst_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_aesop_Aesop_SearchM_instInhabitedState_default(v_Q_18_, v_inst_19_, v_inst_20_);
lean_dec_ref(v_inst_19_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___redArg(lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(v_inst_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState(lean_object* v_a_24_, lean_object* v_inst_25_, lean_object* v_a_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_aesop_Aesop_SearchM_instInhabitedState_default___redArg(v_inst_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabitedState___boxed(lean_object* v_a_28_, lean_object* v_inst_29_, lean_object* v_a_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_aesop_Aesop_SearchM_instInhabitedState(v_a_28_, v_inst_29_, v_a_30_);
lean_dec_ref(v_a_30_);
return v_res_31_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__0(void){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = l_instMonadEIO(lean_box(0));
return v___x_32_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonad___closed__1(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonad___closed__0, &lp_aesop_Aesop_SearchM_instMonad___closed__0_once, _init_lp_aesop_Aesop_SearchM_instMonad___closed__0);
v___x_34_ = l_StateRefT_x27_instMonad___redArg(v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object* v_Q_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; lean_object* v_toApplicative_42_; lean_object* v_toFunctor_43_; lean_object* v_toSeq_44_; lean_object* v_toSeqLeft_45_; lean_object* v_toSeqRight_46_; lean_object* v___f_47_; lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___f_52_; lean_object* v___f_53_; lean_object* v___f_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v_toApplicative_58_; lean_object* v___x_60_; uint8_t v_isShared_61_; uint8_t v_isSharedCheck_89_; 
v___x_41_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonad___closed__1, &lp_aesop_Aesop_SearchM_instMonad___closed__1_once, _init_lp_aesop_Aesop_SearchM_instMonad___closed__1);
v_toApplicative_42_ = lean_ctor_get(v___x_41_, 0);
v_toFunctor_43_ = lean_ctor_get(v_toApplicative_42_, 0);
v_toSeq_44_ = lean_ctor_get(v_toApplicative_42_, 2);
v_toSeqLeft_45_ = lean_ctor_get(v_toApplicative_42_, 3);
v_toSeqRight_46_ = lean_ctor_get(v_toApplicative_42_, 4);
v___f_47_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__2));
v___f_48_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_43_, 2);
v___f_49_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_49_, 0, v_toFunctor_43_);
v___f_50_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_50_, 0, v_toFunctor_43_);
v___x_51_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_51_, 0, v___f_49_);
lean_ctor_set(v___x_51_, 1, v___f_50_);
lean_inc(v_toSeqRight_46_);
v___f_52_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_52_, 0, v_toSeqRight_46_);
lean_inc(v_toSeqLeft_45_);
v___f_53_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_53_, 0, v_toSeqLeft_45_);
lean_inc(v_toSeq_44_);
v___f_54_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_54_, 0, v_toSeq_44_);
v___x_55_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_55_, 0, v___x_51_);
lean_ctor_set(v___x_55_, 1, v___f_47_);
lean_ctor_set(v___x_55_, 2, v___f_54_);
lean_ctor_set(v___x_55_, 3, v___f_53_);
lean_ctor_set(v___x_55_, 4, v___f_52_);
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v___f_48_);
v___x_57_ = l_StateRefT_x27_instMonad___redArg(v___x_56_);
v_toApplicative_58_ = lean_ctor_get(v___x_57_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_57_);
if (v_isSharedCheck_89_ == 0)
{
lean_object* v_unused_90_; 
v_unused_90_ = lean_ctor_get(v___x_57_, 1);
lean_dec(v_unused_90_);
v___x_60_ = v___x_57_;
v_isShared_61_ = v_isSharedCheck_89_;
goto v_resetjp_59_;
}
else
{
lean_inc(v_toApplicative_58_);
lean_dec(v___x_57_);
v___x_60_ = lean_box(0);
v_isShared_61_ = v_isSharedCheck_89_;
goto v_resetjp_59_;
}
v_resetjp_59_:
{
lean_object* v_toFunctor_62_; lean_object* v_toSeq_63_; lean_object* v_toSeqLeft_64_; lean_object* v_toSeqRight_65_; lean_object* v___x_67_; uint8_t v_isShared_68_; uint8_t v_isSharedCheck_87_; 
v_toFunctor_62_ = lean_ctor_get(v_toApplicative_58_, 0);
v_toSeq_63_ = lean_ctor_get(v_toApplicative_58_, 2);
v_toSeqLeft_64_ = lean_ctor_get(v_toApplicative_58_, 3);
v_toSeqRight_65_ = lean_ctor_get(v_toApplicative_58_, 4);
v_isSharedCheck_87_ = !lean_is_exclusive(v_toApplicative_58_);
if (v_isSharedCheck_87_ == 0)
{
lean_object* v_unused_88_; 
v_unused_88_ = lean_ctor_get(v_toApplicative_58_, 1);
lean_dec(v_unused_88_);
v___x_67_ = v_toApplicative_58_;
v_isShared_68_ = v_isSharedCheck_87_;
goto v_resetjp_66_;
}
else
{
lean_inc(v_toSeqRight_65_);
lean_inc(v_toSeqLeft_64_);
lean_inc(v_toSeq_63_);
lean_inc(v_toFunctor_62_);
lean_dec(v_toApplicative_58_);
v___x_67_ = lean_box(0);
v_isShared_68_ = v_isSharedCheck_87_;
goto v_resetjp_66_;
}
v_resetjp_66_:
{
lean_object* v___f_69_; lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___f_72_; lean_object* v___x_73_; lean_object* v___f_74_; lean_object* v___f_75_; lean_object* v___f_76_; lean_object* v___x_78_; 
v___f_69_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__4));
v___f_70_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_62_);
v___f_71_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_71_, 0, v_toFunctor_62_);
v___f_72_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_72_, 0, v_toFunctor_62_);
v___x_73_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_73_, 0, v___f_71_);
lean_ctor_set(v___x_73_, 1, v___f_72_);
v___f_74_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_74_, 0, v_toSeqRight_65_);
v___f_75_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_75_, 0, v_toSeqLeft_64_);
v___f_76_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_76_, 0, v_toSeq_63_);
if (v_isShared_68_ == 0)
{
lean_ctor_set(v___x_67_, 4, v___f_74_);
lean_ctor_set(v___x_67_, 3, v___f_75_);
lean_ctor_set(v___x_67_, 2, v___f_76_);
lean_ctor_set(v___x_67_, 1, v___f_69_);
lean_ctor_set(v___x_67_, 0, v___x_73_);
v___x_78_ = v___x_67_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v___x_73_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v___f_69_);
lean_ctor_set(v_reuseFailAlloc_86_, 2, v___f_76_);
lean_ctor_set(v_reuseFailAlloc_86_, 3, v___f_75_);
lean_ctor_set(v_reuseFailAlloc_86_, 4, v___f_74_);
v___x_78_ = v_reuseFailAlloc_86_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
lean_object* v___x_80_; 
if (v_isShared_61_ == 0)
{
lean_ctor_set(v___x_60_, 1, v___f_70_);
lean_ctor_set(v___x_60_, 0, v___x_78_);
v___x_80_ = v___x_60_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___x_78_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v___f_70_);
v___x_80_ = v_reuseFailAlloc_85_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_81_ = l_StateRefT_x27_instMonad___redArg(v___x_80_);
v___x_82_ = l_StateRefT_x27_instMonad___redArg(v___x_81_);
v___x_83_ = l_StateRefT_x27_instMonad___redArg(v___x_82_);
v___x_84_ = l_ReaderT_instMonad___redArg(v___x_83_);
return v___x_84_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonad___boxed(lean_object* v_Q_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_aesop_Aesop_SearchM_instMonad(v_Q_91_, v_inst_92_);
lean_dec_ref(v_inst_92_);
return v_res_93_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__4(void){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_98_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_99_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__1));
v___x_100_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__0));
v___x_101_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_100_, v___x_99_, v___x_98_);
return v___x_101_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__5(void){
_start:
{
lean_object* v___x_102_; lean_object* v___f_103_; lean_object* v___f_104_; lean_object* v___x_105_; 
v___x_102_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__4, &lp_aesop_Aesop_SearchM_instMonadRef___closed__4_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__4);
v___f_103_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__3));
v___f_104_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__2));
v___x_105_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_104_, v___f_103_, v___x_102_);
return v___x_105_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__6(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__5, &lp_aesop_Aesop_SearchM_instMonadRef___closed__5_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__5);
v___x_107_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__1));
v___x_108_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__0));
v___x_109_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_108_, v___x_107_, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__7(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_110_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__6, &lp_aesop_Aesop_SearchM_instMonadRef___closed__6_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__6);
v___x_111_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__1));
v___x_112_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__0));
v___x_113_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_112_, v___x_111_, v___x_110_);
return v___x_113_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__8(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__7, &lp_aesop_Aesop_SearchM_instMonadRef___closed__7_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__7);
v___x_115_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__1));
v___x_116_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__0));
v___x_117_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_116_, v___x_115_, v___x_114_);
return v___x_117_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__9(void){
_start:
{
lean_object* v___x_118_; lean_object* v___f_119_; lean_object* v___f_120_; lean_object* v___x_121_; 
v___x_118_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__8, &lp_aesop_Aesop_SearchM_instMonadRef___closed__8_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__8);
v___f_119_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__3));
v___f_120_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__2));
v___x_121_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_120_, v___f_119_, v___x_118_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object* v_Q_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v___x_124_; lean_object* v_toMonadRef_125_; 
v___x_124_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__9, &lp_aesop_Aesop_SearchM_instMonadRef___closed__9_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__9);
v_toMonadRef_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc_ref(v_toMonadRef_125_);
return v_toMonadRef_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadRef___boxed(lean_object* v_Q_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_aesop_Aesop_SearchM_instMonadRef(v_Q_126_, v_inst_127_);
lean_dec_ref(v_inst_127_);
return v_res_128_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0(void){
_start:
{
lean_object* v___x_129_; lean_object* v___f_130_; 
v___x_129_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_130_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_130_, 0, v___x_129_);
return v___f_130_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1(void){
_start:
{
lean_object* v___x_131_; lean_object* v___f_132_; 
v___x_131_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_132_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_132_, 0, v___x_131_);
return v___f_132_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2(void){
_start:
{
lean_object* v___f_133_; lean_object* v___f_134_; lean_object* v___x_135_; 
v___f_133_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__1);
v___f_134_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__0);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___f_134_);
lean_ctor_set(v___x_135_, 1, v___f_133_);
return v___x_135_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3(void){
_start:
{
lean_object* v___x_136_; lean_object* v___f_137_; 
v___x_136_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2);
v___f_137_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_137_, 0, v___x_136_);
return v___f_137_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4(void){
_start:
{
lean_object* v___x_138_; lean_object* v___f_139_; 
v___x_138_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__2);
v___f_139_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_139_, 0, v___x_138_);
return v___f_139_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5(void){
_start:
{
lean_object* v___f_140_; lean_object* v___f_141_; lean_object* v___x_142_; 
v___f_140_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__4);
v___f_141_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__3);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___f_141_);
lean_ctor_set(v___x_142_, 1, v___f_140_);
return v___x_142_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__6));
v___x_145_ = l_Lean_stringToMessageData(v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0(lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v___x_155_; lean_object* v_toApplicative_156_; lean_object* v_toFunctor_157_; lean_object* v_toSeq_158_; lean_object* v_toSeqLeft_159_; lean_object* v_toSeqRight_160_; lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___f_163_; lean_object* v___f_164_; lean_object* v___x_165_; lean_object* v___f_166_; lean_object* v___f_167_; lean_object* v___f_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v_toApplicative_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_208_; 
v___x_155_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonad___closed__1, &lp_aesop_Aesop_SearchM_instMonad___closed__1_once, _init_lp_aesop_Aesop_SearchM_instMonad___closed__1);
v_toApplicative_156_ = lean_ctor_get(v___x_155_, 0);
v_toFunctor_157_ = lean_ctor_get(v_toApplicative_156_, 0);
v_toSeq_158_ = lean_ctor_get(v_toApplicative_156_, 2);
v_toSeqLeft_159_ = lean_ctor_get(v_toApplicative_156_, 3);
v_toSeqRight_160_ = lean_ctor_get(v_toApplicative_156_, 4);
v___f_161_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__2));
v___f_162_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_157_, 2);
v___f_163_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_163_, 0, v_toFunctor_157_);
v___f_164_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_164_, 0, v_toFunctor_157_);
v___x_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_165_, 0, v___f_163_);
lean_ctor_set(v___x_165_, 1, v___f_164_);
lean_inc(v_toSeqRight_160_);
v___f_166_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_166_, 0, v_toSeqRight_160_);
lean_inc(v_toSeqLeft_159_);
v___f_167_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_167_, 0, v_toSeqLeft_159_);
lean_inc(v_toSeq_158_);
v___f_168_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_168_, 0, v_toSeq_158_);
v___x_169_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_169_, 0, v___x_165_);
lean_ctor_set(v___x_169_, 1, v___f_161_);
lean_ctor_set(v___x_169_, 2, v___f_168_);
lean_ctor_set(v___x_169_, 3, v___f_167_);
lean_ctor_set(v___x_169_, 4, v___f_166_);
v___x_170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___f_162_);
v___x_171_ = l_StateRefT_x27_instMonad___redArg(v___x_170_);
v_toApplicative_172_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_208_ == 0)
{
lean_object* v_unused_209_; 
v_unused_209_ = lean_ctor_get(v___x_171_, 1);
lean_dec(v_unused_209_);
v___x_174_ = v___x_171_;
v_isShared_175_ = v_isSharedCheck_208_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_toApplicative_172_);
lean_dec(v___x_171_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_208_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v_toFunctor_176_; lean_object* v_toSeq_177_; lean_object* v_toSeqLeft_178_; lean_object* v_toSeqRight_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_206_; 
v_toFunctor_176_ = lean_ctor_get(v_toApplicative_172_, 0);
v_toSeq_177_ = lean_ctor_get(v_toApplicative_172_, 2);
v_toSeqLeft_178_ = lean_ctor_get(v_toApplicative_172_, 3);
v_toSeqRight_179_ = lean_ctor_get(v_toApplicative_172_, 4);
v_isSharedCheck_206_ = !lean_is_exclusive(v_toApplicative_172_);
if (v_isSharedCheck_206_ == 0)
{
lean_object* v_unused_207_; 
v_unused_207_ = lean_ctor_get(v_toApplicative_172_, 1);
lean_dec(v_unused_207_);
v___x_181_ = v_toApplicative_172_;
v_isShared_182_ = v_isSharedCheck_206_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_toSeqRight_179_);
lean_inc(v_toSeqLeft_178_);
lean_inc(v_toSeq_177_);
lean_inc(v_toFunctor_176_);
lean_dec(v_toApplicative_172_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_206_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___f_183_; lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___x_187_; lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_192_; 
v___f_183_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__4));
v___f_184_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_176_);
v___f_185_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_185_, 0, v_toFunctor_176_);
v___f_186_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_186_, 0, v_toFunctor_176_);
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v___f_185_);
lean_ctor_set(v___x_187_, 1, v___f_186_);
v___f_188_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_188_, 0, v_toSeqRight_179_);
v___f_189_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_189_, 0, v_toSeqLeft_178_);
v___f_190_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_190_, 0, v_toSeq_177_);
if (v_isShared_182_ == 0)
{
lean_ctor_set(v___x_181_, 4, v___f_188_);
lean_ctor_set(v___x_181_, 3, v___f_189_);
lean_ctor_set(v___x_181_, 2, v___f_190_);
lean_ctor_set(v___x_181_, 1, v___f_183_);
lean_ctor_set(v___x_181_, 0, v___x_187_);
v___x_192_ = v___x_181_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v___x_187_);
lean_ctor_set(v_reuseFailAlloc_205_, 1, v___f_183_);
lean_ctor_set(v_reuseFailAlloc_205_, 2, v___f_190_);
lean_ctor_set(v_reuseFailAlloc_205_, 3, v___f_189_);
lean_ctor_set(v_reuseFailAlloc_205_, 4, v___f_188_);
v___x_192_ = v_reuseFailAlloc_205_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
lean_object* v___x_194_; 
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 1, v___f_184_);
lean_ctor_set(v___x_174_, 0, v___x_192_);
v___x_194_ = v___x_174_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v___x_192_);
lean_ctor_set(v_reuseFailAlloc_204_, 1, v___f_184_);
v___x_194_ = v_reuseFailAlloc_204_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v_toMonadRef_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_366__overap_202_; lean_object* v___x_203_; 
v___x_195_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5);
v___x_196_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__5, &lp_aesop_Aesop_SearchM_instMonadRef___closed__5_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__5);
v_toMonadRef_197_ = lean_ctor_get(v___x_196_, 0);
v___x_198_ = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(v___x_194_);
v___x_199_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_198_, v___x_194_);
lean_inc_ref(v_toMonadRef_197_);
v___x_200_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_200_, 0, v___x_195_);
lean_ctor_set(v___x_200_, 1, v_toMonadRef_197_);
lean_ctor_set(v___x_200_, 2, v___x_199_);
v___x_201_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__7);
v___x_366__overap_202_ = l_Lean_throwError___redArg(v___x_194_, v___x_200_, v___x_201_);
lean_inc(v___y_153_);
lean_inc_ref(v___y_152_);
lean_inc(v___y_151_);
lean_inc_ref(v___y_150_);
v___x_203_ = lean_apply_5(v___x_366__overap_202_, v___y_150_, v___y_151_, v___y_152_, v___y_153_, lean_box(0));
return v___x_203_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___lam__0___boxed(lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_aesop_Aesop_SearchM_instInhabited___lam__0(v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec(v___y_212_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited(lean_object* v_Q_221_, lean_object* v_inst_222_, lean_object* v_00_u03b1_223_){
_start:
{
lean_object* v___f_224_; 
v___f_224_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instInhabited___closed__0));
return v___f_224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instInhabited___boxed(lean_object* v_Q_225_, lean_object* v_inst_226_, lean_object* v_00_u03b1_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_aesop_Aesop_SearchM_instInhabited(v_Q_225_, v_inst_226_, v_00_u03b1_227_);
lean_dec_ref(v_inst_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_st_ref_get(v___y_230_);
v___x_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__0___boxed(lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_aesop_Aesop_SearchM_instMonadStateState___lam__0(v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec(v___y_242_);
lean_dec(v___y_241_);
lean_dec_ref(v___y_240_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(lean_object* v_s_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = lean_st_ref_set(v___y_252_, v_s_250_);
v___x_261_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__1___boxed(lean_object* v_s_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_aesop_Aesop_SearchM_instMonadStateState___lam__1(v_s_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_);
lean_dec(v___y_270_);
lean_dec_ref(v___y_269_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v___y_266_);
lean_dec(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(lean_object* v_00_u03b1_273_, lean_object* v_f_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v_fst_286_; lean_object* v_snd_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_284_ = lean_st_ref_take(v___y_276_);
v___x_285_ = lean_apply_1(v_f_274_, v___x_284_);
v_fst_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_fst_286_);
v_snd_287_ = lean_ctor_get(v___x_285_, 1);
lean_inc(v_snd_287_);
lean_dec_ref(v___x_285_);
v___x_288_ = lean_st_ref_set(v___y_276_, v_snd_287_);
v___x_289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_289_, 0, v_fst_286_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___lam__2___boxed(lean_object* v_00_u03b1_290_, lean_object* v_f_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_aesop_Aesop_SearchM_instMonadStateState___lam__2(v_00_u03b1_290_, v_f_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
lean_dec(v___y_295_);
lean_dec(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState(lean_object* v_Q_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadStateState___closed__3));
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadStateState___boxed(lean_object* v_Q_312_, lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_aesop_Aesop_SearchM_instMonadStateState(v_Q_312_, v_inst_313_);
lean_dec_ref(v_inst_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext(lean_object* v_Q_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; lean_object* v_toApplicative_318_; lean_object* v_toFunctor_319_; lean_object* v_toSeq_320_; lean_object* v_toSeqLeft_321_; lean_object* v_toSeqRight_322_; lean_object* v___f_323_; lean_object* v___f_324_; lean_object* v___f_325_; lean_object* v___f_326_; lean_object* v___x_327_; lean_object* v___f_328_; lean_object* v___f_329_; lean_object* v___f_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v_toApplicative_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_365_; 
v___x_317_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonad___closed__1, &lp_aesop_Aesop_SearchM_instMonad___closed__1_once, _init_lp_aesop_Aesop_SearchM_instMonad___closed__1);
v_toApplicative_318_ = lean_ctor_get(v___x_317_, 0);
v_toFunctor_319_ = lean_ctor_get(v_toApplicative_318_, 0);
v_toSeq_320_ = lean_ctor_get(v_toApplicative_318_, 2);
v_toSeqLeft_321_ = lean_ctor_get(v_toApplicative_318_, 3);
v_toSeqRight_322_ = lean_ctor_get(v_toApplicative_318_, 4);
v___f_323_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__2));
v___f_324_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_319_, 2);
v___f_325_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_325_, 0, v_toFunctor_319_);
v___f_326_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_326_, 0, v_toFunctor_319_);
v___x_327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_327_, 0, v___f_325_);
lean_ctor_set(v___x_327_, 1, v___f_326_);
lean_inc(v_toSeqRight_322_);
v___f_328_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_328_, 0, v_toSeqRight_322_);
lean_inc(v_toSeqLeft_321_);
v___f_329_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_329_, 0, v_toSeqLeft_321_);
lean_inc(v_toSeq_320_);
v___f_330_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_330_, 0, v_toSeq_320_);
v___x_331_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_331_, 0, v___x_327_);
lean_ctor_set(v___x_331_, 1, v___f_323_);
lean_ctor_set(v___x_331_, 2, v___f_330_);
lean_ctor_set(v___x_331_, 3, v___f_329_);
lean_ctor_set(v___x_331_, 4, v___f_328_);
v___x_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
lean_ctor_set(v___x_332_, 1, v___f_324_);
v___x_333_ = l_StateRefT_x27_instMonad___redArg(v___x_332_);
v_toApplicative_334_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_365_ == 0)
{
lean_object* v_unused_366_; 
v_unused_366_ = lean_ctor_get(v___x_333_, 1);
lean_dec(v_unused_366_);
v___x_336_ = v___x_333_;
v_isShared_337_ = v_isSharedCheck_365_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_toApplicative_334_);
lean_dec(v___x_333_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_365_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v_toFunctor_338_; lean_object* v_toSeq_339_; lean_object* v_toSeqLeft_340_; lean_object* v_toSeqRight_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_363_; 
v_toFunctor_338_ = lean_ctor_get(v_toApplicative_334_, 0);
v_toSeq_339_ = lean_ctor_get(v_toApplicative_334_, 2);
v_toSeqLeft_340_ = lean_ctor_get(v_toApplicative_334_, 3);
v_toSeqRight_341_ = lean_ctor_get(v_toApplicative_334_, 4);
v_isSharedCheck_363_ = !lean_is_exclusive(v_toApplicative_334_);
if (v_isSharedCheck_363_ == 0)
{
lean_object* v_unused_364_; 
v_unused_364_ = lean_ctor_get(v_toApplicative_334_, 1);
lean_dec(v_unused_364_);
v___x_343_ = v_toApplicative_334_;
v_isShared_344_ = v_isSharedCheck_363_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_toSeqRight_341_);
lean_inc(v_toSeqLeft_340_);
lean_inc(v_toSeq_339_);
lean_inc(v_toFunctor_338_);
lean_dec(v_toApplicative_334_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_363_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___f_345_; lean_object* v___f_346_; lean_object* v___f_347_; lean_object* v___f_348_; lean_object* v___x_349_; lean_object* v___f_350_; lean_object* v___f_351_; lean_object* v___f_352_; lean_object* v___x_354_; 
v___f_345_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__4));
v___f_346_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_338_);
v___f_347_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_347_, 0, v_toFunctor_338_);
v___f_348_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_348_, 0, v_toFunctor_338_);
v___x_349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_349_, 0, v___f_347_);
lean_ctor_set(v___x_349_, 1, v___f_348_);
v___f_350_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_350_, 0, v_toSeqRight_341_);
v___f_351_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_351_, 0, v_toSeqLeft_340_);
v___f_352_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_352_, 0, v_toSeq_339_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 4, v___f_350_);
lean_ctor_set(v___x_343_, 3, v___f_351_);
lean_ctor_set(v___x_343_, 2, v___f_352_);
lean_ctor_set(v___x_343_, 1, v___f_345_);
lean_ctor_set(v___x_343_, 0, v___x_349_);
v___x_354_ = v___x_343_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v___x_349_);
lean_ctor_set(v_reuseFailAlloc_362_, 1, v___f_345_);
lean_ctor_set(v_reuseFailAlloc_362_, 2, v___f_352_);
lean_ctor_set(v_reuseFailAlloc_362_, 3, v___f_351_);
lean_ctor_set(v_reuseFailAlloc_362_, 4, v___f_350_);
v___x_354_ = v_reuseFailAlloc_362_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
lean_object* v___x_356_; 
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 1, v___f_346_);
lean_ctor_set(v___x_336_, 0, v___x_354_);
v___x_356_ = v___x_336_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v___x_354_);
lean_ctor_set(v_reuseFailAlloc_361_, 1, v___f_346_);
v___x_356_ = v_reuseFailAlloc_361_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_357_ = l_StateRefT_x27_instMonad___redArg(v___x_356_);
v___x_358_ = l_StateRefT_x27_instMonad___redArg(v___x_357_);
v___x_359_ = l_StateRefT_x27_instMonad___redArg(v___x_358_);
v___x_360_ = lean_alloc_closure((void*)(l_ReaderT_read___boxed), 4, 3);
lean_closure_set(v___x_360_, 0, lean_box(0));
lean_closure_set(v___x_360_, 1, lean_box(0));
lean_closure_set(v___x_360_, 2, v___x_359_);
return v___x_360_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadReaderContext___boxed(lean_object* v_Q_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_aesop_Aesop_SearchM_instMonadReaderContext(v_Q_367_, v_inst_368_);
lean_dec_ref(v_inst_368_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(lean_object* v_00_u03b1_370_, lean_object* v_x_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v___x_381_; lean_object* v_iteration_382_; lean_object* v_ruleSet_383_; lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_381_ = lean_st_ref_get(v___y_373_);
v_iteration_382_ = lean_ctor_get(v___x_381_, 0);
lean_inc(v_iteration_382_);
lean_dec(v___x_381_);
v_ruleSet_383_ = lean_ctor_get(v___y_372_, 0);
lean_inc_ref(v_ruleSet_383_);
v___x_384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_384_, 0, v_iteration_382_);
lean_ctor_set(v___x_384_, 1, v_ruleSet_383_);
lean_inc(v___y_379_);
lean_inc_ref(v___y_378_);
lean_inc(v___y_377_);
lean_inc_ref(v___y_376_);
lean_inc(v___y_375_);
lean_inc(v___y_374_);
v___x_385_ = lean_apply_8(v_x_371_, v___x_384_, v___y_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, lean_box(0));
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object* v_00_u03b1_386_, lean_object* v_x_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0(v_00_u03b1_386_, v_x_387_, v___y_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
lean_dec(v___y_395_);
lean_dec_ref(v___y_394_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
lean_dec(v___y_391_);
lean_dec(v___y_390_);
lean_dec(v___y_389_);
lean_dec_ref(v___y_388_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM(lean_object* v_Q_399_, lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; 
v___f_401_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___closed__0));
return v___f_401_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___boxed(lean_object* v_Q_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_aesop_Aesop_SearchM_instMonadLiftTreeM(v_Q_402_, v_inst_403_);
lean_dec_ref(v_inst_403_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg(lean_object* v_ctx_405_, lean_object* v_00_u03c3_406_, lean_object* v_tree_407_, lean_object* v_x_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_415_ = lean_st_mk_ref(v_tree_407_);
v___x_416_ = lean_st_mk_ref(v_00_u03c3_406_);
lean_inc(v_a_413_);
lean_inc_ref(v_a_412_);
lean_inc(v_a_411_);
lean_inc_ref(v_a_410_);
lean_inc(v_a_409_);
lean_inc(v___x_415_);
lean_inc(v___x_416_);
v___x_417_ = lean_apply_9(v_x_408_, v_ctx_405_, v___x_416_, v___x_415_, v_a_409_, v_a_410_, v_a_411_, v_a_412_, v_a_413_, lean_box(0));
if (lean_obj_tag(v___x_417_) == 0)
{
lean_object* v_a_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_429_; 
v_a_418_ = lean_ctor_get(v___x_417_, 0);
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_429_ == 0)
{
v___x_420_ = v___x_417_;
v_isShared_421_ = v_isSharedCheck_429_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_a_418_);
lean_dec(v___x_417_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_429_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_427_; 
v___x_422_ = lean_st_ref_get(v___x_416_);
lean_dec(v___x_416_);
v___x_423_ = lean_st_ref_get(v___x_415_);
lean_dec(v___x_415_);
v___x_424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_422_);
lean_ctor_set(v___x_424_, 1, v___x_423_);
v___x_425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_425_, 0, v_a_418_);
lean_ctor_set(v___x_425_, 1, v___x_424_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 0, v___x_425_);
v___x_427_ = v___x_420_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_425_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_dec(v___x_416_);
lean_dec(v___x_415_);
v_a_430_ = lean_ctor_get(v___x_417_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___x_417_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_417_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_a_430_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___redArg___boxed(lean_object* v_ctx_438_, lean_object* v_00_u03c3_439_, lean_object* v_tree_440_, lean_object* v_x_441_, lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_aesop_Aesop_SearchM_run_x27___redArg(v_ctx_438_, v_00_u03c3_439_, v_tree_440_, v_x_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
lean_dec(v_a_446_);
lean_dec_ref(v_a_445_);
lean_dec(v_a_444_);
lean_dec_ref(v_a_443_);
lean_dec(v_a_442_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27(lean_object* v_Q_449_, lean_object* v_inst_450_, lean_object* v_00_u03b1_451_, lean_object* v_ctx_452_, lean_object* v_00_u03c3_453_, lean_object* v_tree_454_, lean_object* v_x_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_aesop_Aesop_SearchM_run_x27___redArg(v_ctx_452_, v_00_u03c3_453_, v_tree_454_, v_x_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run_x27___boxed(lean_object* v_Q_463_, lean_object* v_inst_464_, lean_object* v_00_u03b1_465_, lean_object* v_ctx_466_, lean_object* v_00_u03c3_467_, lean_object* v_tree_468_, lean_object* v_x_469_, lean_object* v_a_470_, lean_object* v_a_471_, lean_object* v_a_472_, lean_object* v_a_473_, lean_object* v_a_474_, lean_object* v_a_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_aesop_Aesop_SearchM_run_x27(v_Q_463_, v_inst_464_, v_00_u03b1_465_, v_ctx_466_, v_00_u03c3_467_, v_tree_468_, v_x_469_, v_a_470_, v_a_471_, v_a_472_, v_a_473_, v_a_474_);
lean_dec(v_a_474_);
lean_dec_ref(v_a_473_);
lean_dec(v_a_472_);
lean_dec_ref(v_a_471_);
lean_dec(v_a_470_);
lean_dec_ref(v_inst_464_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0(lean_object* v_x_477_){
_start:
{
lean_object* v_snd_478_; 
v_snd_478_ = lean_ctor_get(v_x_477_, 1);
lean_inc(v_snd_478_);
return v_snd_478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__0___boxed(lean_object* v_x_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_aesop_Aesop_SearchM_run___redArg___lam__0(v_x_479_);
lean_dec_ref(v_x_479_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1(lean_object* v_x_481_){
_start:
{
lean_object* v_snd_482_; 
v_snd_482_ = lean_ctor_get(v_x_481_, 1);
lean_inc(v_snd_482_);
return v_snd_482_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___lam__1___boxed(lean_object* v_x_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_aesop_Aesop_SearchM_run___redArg___lam__1(v_x_483_);
lean_dec_ref(v_x_483_);
return v_res_484_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__0(void){
_start:
{
lean_object* v___x_485_; lean_object* v___f_486_; 
v___x_485_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5);
v___f_486_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_486_, 0, v___x_485_);
return v___f_486_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__1(void){
_start:
{
lean_object* v___x_487_; lean_object* v___f_488_; 
v___x_487_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5, &lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5_once, _init_lp_aesop_Aesop_SearchM_instInhabited___lam__0___closed__5);
v___f_488_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_488_, 0, v___x_487_);
return v___f_488_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__2(void){
_start:
{
lean_object* v___f_489_; lean_object* v___f_490_; lean_object* v___x_491_; 
v___f_489_ = lean_obj_once(&lp_aesop_Aesop_SearchM_run___redArg___closed__1, &lp_aesop_Aesop_SearchM_run___redArg___closed__1_once, _init_lp_aesop_Aesop_SearchM_run___redArg___closed__1);
v___f_490_ = lean_obj_once(&lp_aesop_Aesop_SearchM_run___redArg___closed__0, &lp_aesop_Aesop_SearchM_run___redArg___closed__0_once, _init_lp_aesop_Aesop_SearchM_run___redArg___closed__0);
v___x_491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_491_, 0, v___f_490_);
lean_ctor_set(v___x_491_, 1, v___f_489_);
return v___x_491_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__3(void){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___f_494_; 
v___x_492_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonadRef___closed__1));
v___x_493_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_494_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_494_, 0, v___x_493_);
lean_closure_set(v___f_494_, 1, v___x_492_);
return v___f_494_;
}
}
static lean_object* _init_lp_aesop_Aesop_SearchM_run___redArg___closed__16(void){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_516_ = ((lean_object*)(lp_aesop_Aesop_SearchM_run___redArg___closed__15));
v___x_517_ = l_Lean_stringToMessageData(v___x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object* v_inst_519_, lean_object* v_ruleSet_520_, lean_object* v_options_521_, lean_object* v_simpConfig_522_, lean_object* v_simpConfigStx_x3f_523_, lean_object* v_goal_524_, lean_object* v_x_525_, lean_object* v_a_526_, lean_object* v_a_527_, lean_object* v_a_528_, lean_object* v_a_529_, lean_object* v_a_530_){
_start:
{
lean_object* v___x_532_; lean_object* v_toApplicative_533_; lean_object* v_toFunctor_534_; lean_object* v_toSeq_535_; lean_object* v_toSeqLeft_536_; lean_object* v_toSeqRight_537_; lean_object* v___f_538_; lean_object* v___f_539_; lean_object* v___f_540_; lean_object* v___f_541_; lean_object* v___x_542_; lean_object* v___f_543_; lean_object* v___f_544_; lean_object* v___f_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v_toApplicative_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_649_; 
v___x_532_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonad___closed__1, &lp_aesop_Aesop_SearchM_instMonad___closed__1_once, _init_lp_aesop_Aesop_SearchM_instMonad___closed__1);
v_toApplicative_533_ = lean_ctor_get(v___x_532_, 0);
v_toFunctor_534_ = lean_ctor_get(v_toApplicative_533_, 0);
v_toSeq_535_ = lean_ctor_get(v_toApplicative_533_, 2);
v_toSeqLeft_536_ = lean_ctor_get(v_toApplicative_533_, 3);
v_toSeqRight_537_ = lean_ctor_get(v_toApplicative_533_, 4);
v___f_538_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__2));
v___f_539_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__3));
lean_inc_ref_n(v_toFunctor_534_, 2);
v___f_540_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_540_, 0, v_toFunctor_534_);
v___f_541_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_541_, 0, v_toFunctor_534_);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v___f_540_);
lean_ctor_set(v___x_542_, 1, v___f_541_);
lean_inc(v_toSeqRight_537_);
v___f_543_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_543_, 0, v_toSeqRight_537_);
lean_inc(v_toSeqLeft_536_);
v___f_544_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_544_, 0, v_toSeqLeft_536_);
lean_inc(v_toSeq_535_);
v___f_545_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_545_, 0, v_toSeq_535_);
v___x_546_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_546_, 0, v___x_542_);
lean_ctor_set(v___x_546_, 1, v___f_538_);
lean_ctor_set(v___x_546_, 2, v___f_545_);
lean_ctor_set(v___x_546_, 3, v___f_544_);
lean_ctor_set(v___x_546_, 4, v___f_543_);
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v___x_546_);
lean_ctor_set(v___x_547_, 1, v___f_539_);
v___x_548_ = l_StateRefT_x27_instMonad___redArg(v___x_547_);
v_toApplicative_549_ = lean_ctor_get(v___x_548_, 0);
v_isSharedCheck_649_ = !lean_is_exclusive(v___x_548_);
if (v_isSharedCheck_649_ == 0)
{
lean_object* v_unused_650_; 
v_unused_650_ = lean_ctor_get(v___x_548_, 1);
lean_dec(v_unused_650_);
v___x_551_ = v___x_548_;
v_isShared_552_ = v_isSharedCheck_649_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_toApplicative_549_);
lean_dec(v___x_548_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_649_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v_toFunctor_553_; lean_object* v_toSeq_554_; lean_object* v_toSeqLeft_555_; lean_object* v_toSeqRight_556_; lean_object* v___x_558_; uint8_t v_isShared_559_; uint8_t v_isSharedCheck_647_; 
v_toFunctor_553_ = lean_ctor_get(v_toApplicative_549_, 0);
v_toSeq_554_ = lean_ctor_get(v_toApplicative_549_, 2);
v_toSeqLeft_555_ = lean_ctor_get(v_toApplicative_549_, 3);
v_toSeqRight_556_ = lean_ctor_get(v_toApplicative_549_, 4);
v_isSharedCheck_647_ = !lean_is_exclusive(v_toApplicative_549_);
if (v_isSharedCheck_647_ == 0)
{
lean_object* v_unused_648_; 
v_unused_648_ = lean_ctor_get(v_toApplicative_549_, 1);
lean_dec(v_unused_648_);
v___x_558_ = v_toApplicative_549_;
v_isShared_559_ = v_isSharedCheck_647_;
goto v_resetjp_557_;
}
else
{
lean_inc(v_toSeqRight_556_);
lean_inc(v_toSeqLeft_555_);
lean_inc(v_toSeq_554_);
lean_inc(v_toFunctor_553_);
lean_dec(v_toApplicative_549_);
v___x_558_ = lean_box(0);
v_isShared_559_ = v_isSharedCheck_647_;
goto v_resetjp_557_;
}
v_resetjp_557_:
{
lean_object* v___f_560_; lean_object* v___f_561_; lean_object* v___f_562_; lean_object* v___f_563_; lean_object* v___x_564_; lean_object* v___f_565_; lean_object* v___f_566_; lean_object* v___f_567_; lean_object* v___x_569_; 
v___f_560_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__4));
v___f_561_ = ((lean_object*)(lp_aesop_Aesop_SearchM_instMonad___closed__5));
lean_inc_ref(v_toFunctor_553_);
v___f_562_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_562_, 0, v_toFunctor_553_);
v___f_563_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_563_, 0, v_toFunctor_553_);
v___x_564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_564_, 0, v___f_562_);
lean_ctor_set(v___x_564_, 1, v___f_563_);
v___f_565_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_565_, 0, v_toSeqRight_556_);
v___f_566_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_566_, 0, v_toSeqLeft_555_);
v___f_567_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_567_, 0, v_toSeq_554_);
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 4, v___f_565_);
lean_ctor_set(v___x_558_, 3, v___f_566_);
lean_ctor_set(v___x_558_, 2, v___f_567_);
lean_ctor_set(v___x_558_, 1, v___f_560_);
lean_ctor_set(v___x_558_, 0, v___x_564_);
v___x_569_ = v___x_558_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v___x_564_);
lean_ctor_set(v_reuseFailAlloc_646_, 1, v___f_560_);
lean_ctor_set(v_reuseFailAlloc_646_, 2, v___f_567_);
lean_ctor_set(v_reuseFailAlloc_646_, 3, v___f_566_);
lean_ctor_set(v_reuseFailAlloc_646_, 4, v___f_565_);
v___x_569_ = v_reuseFailAlloc_646_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
lean_object* v___x_571_; 
if (v_isShared_552_ == 0)
{
lean_ctor_set(v___x_551_, 1, v___f_561_);
lean_ctor_set(v___x_551_, 0, v___x_569_);
v___x_571_ = v___x_551_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v___x_569_);
lean_ctor_set(v_reuseFailAlloc_645_, 1, v___f_561_);
v___x_571_ = v_reuseFailAlloc_645_;
goto v_reusejp_570_;
}
v_reusejp_570_:
{
lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v_toMonadRef_575_; lean_object* v___f_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_572_ = l_StateRefT_x27_instMonad___redArg(v___x_571_);
v___x_573_ = lean_obj_once(&lp_aesop_Aesop_SearchM_run___redArg___closed__2, &lp_aesop_Aesop_SearchM_run___redArg___closed__2_once, _init_lp_aesop_Aesop_SearchM_run___redArg___closed__2);
v___x_574_ = lean_obj_once(&lp_aesop_Aesop_SearchM_instMonadRef___closed__6, &lp_aesop_Aesop_SearchM_instMonadRef___closed__6_once, _init_lp_aesop_Aesop_SearchM_instMonadRef___closed__6);
v_toMonadRef_575_ = lean_ctor_get(v___x_574_, 0);
v___f_576_ = lean_obj_once(&lp_aesop_Aesop_SearchM_run___redArg___closed__3, &lp_aesop_Aesop_SearchM_run___redArg___closed__3_once, _init_lp_aesop_Aesop_SearchM_run___redArg___closed__3);
lean_inc_ref(v___x_572_);
v___x_577_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_576_, v___x_572_);
lean_inc_ref(v_toMonadRef_575_);
v___x_578_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_578_, 0, v___x_573_);
lean_ctor_set(v___x_578_, 1, v_toMonadRef_575_);
lean_ctor_set(v___x_578_, 2, v___x_577_);
lean_inc_ref(v_ruleSet_520_);
v___x_579_ = lp_aesop_Aesop_mkInitialTree(v_goal_524_, v_ruleSet_520_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_579_) == 0)
{
lean_object* v_a_580_; lean_object* v___x_581_; 
v_a_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc(v_a_580_);
lean_dec_ref_known(v___x_579_, 1);
v___x_581_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_530_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v_a_582_; lean_object* v_simpTheoremsArray_583_; lean_object* v_simprocsArray_584_; lean_object* v___f_585_; lean_object* v___x_586_; size_t v_sz_587_; size_t v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v_a_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_a_582_);
lean_dec_ref_known(v___x_581_, 1);
v_simpTheoremsArray_583_ = lean_ctor_get(v_ruleSet_520_, 1);
v_simprocsArray_584_ = lean_ctor_get(v_ruleSet_520_, 2);
v___f_585_ = ((lean_object*)(lp_aesop_Aesop_SearchM_run___redArg___closed__4));
v___x_586_ = ((lean_object*)(lp_aesop_Aesop_SearchM_run___redArg___closed__14));
v_sz_587_ = lean_array_size(v_simpTheoremsArray_583_);
v___x_588_ = ((size_t)0ULL);
lean_inc_ref(v_simpTheoremsArray_583_);
v___x_589_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_586_, v___f_585_, v_sz_587_, v___x_588_, v_simpTheoremsArray_583_);
v___x_590_ = l_Lean_Options_empty;
v___x_591_ = l_Lean_Meta_Simp_mkContext___redArg(v_simpConfig_522_, v___x_589_, v_a_582_, v___x_590_, v_a_527_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_591_) == 0)
{
lean_object* v_a_592_; lean_object* v_toOptions_593_; lean_object* v_root_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v_elimMVarCluster_597_; lean_object* v___x_598_; lean_object* v_goals_599_; lean_object* v___x_600_; lean_object* v___x_601_; uint8_t v___x_602_; 
v_a_592_ = lean_ctor_get(v___x_591_, 0);
lean_inc(v_a_592_);
lean_dec_ref_known(v___x_591_, 1);
v_toOptions_593_ = lean_ctor_get(v_options_521_, 0);
v_root_594_ = lean_ctor_get(v_a_580_, 0);
v___x_595_ = lean_st_ref_get(v_root_594_);
v___x_596_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_597_ = lean_ctor_get(v___x_596_, 5);
lean_inc_ref(v_elimMVarCluster_597_);
v___x_598_ = lean_apply_1(v_elimMVarCluster_597_, v___x_595_);
v_goals_599_ = lean_ctor_get(v___x_598_, 1);
lean_inc_ref(v_goals_599_);
lean_dec_ref(v___x_598_);
v___x_600_ = lean_array_get_size(v_goals_599_);
v___x_601_ = lean_unsigned_to_nat(1u);
v___x_602_ = lean_nat_dec_eq(v___x_600_, v___x_601_);
if (v___x_602_ == 0)
{
lean_object* v___x_603_; lean_object* v___x_1790__overap_604_; lean_object* v___x_605_; 
lean_dec_ref(v_goals_599_);
lean_dec(v_a_592_);
lean_dec(v_a_580_);
lean_dec_ref(v_x_525_);
lean_dec(v_simpConfigStx_x3f_523_);
lean_dec_ref(v_options_521_);
lean_dec_ref(v_ruleSet_520_);
lean_dec_ref(v_inst_519_);
v___x_603_ = lean_obj_once(&lp_aesop_Aesop_SearchM_run___redArg___closed__16, &lp_aesop_Aesop_SearchM_run___redArg___closed__16_once, _init_lp_aesop_Aesop_SearchM_run___redArg___closed__16);
v___x_1790__overap_604_ = l_Lean_throwError___redArg(v___x_572_, v___x_578_, v___x_603_);
lean_inc(v_a_530_);
lean_inc_ref(v_a_529_);
lean_inc(v_a_528_);
lean_inc_ref(v_a_527_);
lean_inc(v_a_526_);
v___x_605_ = lean_apply_6(v___x_1790__overap_604_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, v_a_530_, lean_box(0));
return v___x_605_;
}
else
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; uint8_t v_enableSimp_611_; uint8_t v_useSimpAll_612_; lean_object* v___f_613_; size_t v_sz_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; uint8_t v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
lean_dec_ref_known(v___x_578_, 3);
lean_dec_ref(v___x_572_);
v___x_606_ = lean_unsigned_to_nat(0u);
v___x_607_ = lean_array_fget(v_goals_599_, v___x_606_);
lean_dec_ref(v_goals_599_);
v___x_608_ = lean_mk_empty_array_with_capacity(v___x_601_);
v___x_609_ = lean_array_push(v___x_608_, v___x_607_);
v___x_610_ = lp_aesop_Aesop_Queue_init_x27___redArg(v_inst_519_, v___x_609_);
v_enableSimp_611_ = lean_ctor_get_uint8(v_toOptions_593_, sizeof(void*)*6 + 7);
v_useSimpAll_612_ = lean_ctor_get_uint8(v_toOptions_593_, sizeof(void*)*6 + 8);
v___f_613_ = ((lean_object*)(lp_aesop_Aesop_SearchM_run___redArg___closed__17));
v_sz_614_ = lean_array_size(v_simprocsArray_584_);
lean_inc_ref(v_simprocsArray_584_);
v___x_615_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_586_, v___f_613_, v_sz_614_, v___x_588_, v_simprocsArray_584_);
v___x_616_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v___x_616_, 0, v_a_592_);
lean_ctor_set(v___x_616_, 1, v_simpConfigStx_x3f_523_);
lean_ctor_set(v___x_616_, 2, v___x_615_);
lean_ctor_set_uint8(v___x_616_, sizeof(void*)*3, v_enableSimp_611_);
lean_ctor_set_uint8(v___x_616_, sizeof(void*)*3 + 1, v_useSimpAll_612_);
v___x_617_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_617_, 0, v_ruleSet_520_);
lean_ctor_set(v___x_617_, 1, v___x_616_);
lean_ctor_set(v___x_617_, 2, v_options_521_);
v___x_618_ = 0;
v___x_619_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_619_, 0, v___x_601_);
lean_ctor_set(v___x_619_, 1, v___x_610_);
lean_ctor_set_uint8(v___x_619_, sizeof(void*)*2, v___x_618_);
v___x_620_ = lp_aesop_Aesop_SearchM_run_x27___redArg(v___x_617_, v___x_619_, v_a_580_, v_x_525_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
return v___x_620_;
}
}
else
{
lean_object* v_a_621_; lean_object* v___x_623_; uint8_t v_isShared_624_; uint8_t v_isSharedCheck_628_; 
lean_dec(v_a_580_);
lean_dec_ref_known(v___x_578_, 3);
lean_dec_ref(v___x_572_);
lean_dec_ref(v_x_525_);
lean_dec(v_simpConfigStx_x3f_523_);
lean_dec_ref(v_options_521_);
lean_dec_ref(v_ruleSet_520_);
lean_dec_ref(v_inst_519_);
v_a_621_ = lean_ctor_get(v___x_591_, 0);
v_isSharedCheck_628_ = !lean_is_exclusive(v___x_591_);
if (v_isSharedCheck_628_ == 0)
{
v___x_623_ = v___x_591_;
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
else
{
lean_inc(v_a_621_);
lean_dec(v___x_591_);
v___x_623_ = lean_box(0);
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
v_resetjp_622_:
{
lean_object* v___x_626_; 
if (v_isShared_624_ == 0)
{
v___x_626_ = v___x_623_;
goto v_reusejp_625_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v_a_621_);
v___x_626_ = v_reuseFailAlloc_627_;
goto v_reusejp_625_;
}
v_reusejp_625_:
{
return v___x_626_;
}
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec(v_a_580_);
lean_dec_ref_known(v___x_578_, 3);
lean_dec_ref(v___x_572_);
lean_dec_ref(v_x_525_);
lean_dec(v_simpConfigStx_x3f_523_);
lean_dec_ref(v_simpConfig_522_);
lean_dec_ref(v_options_521_);
lean_dec_ref(v_ruleSet_520_);
lean_dec_ref(v_inst_519_);
v_a_629_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_581_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_581_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
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
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
lean_dec_ref_known(v___x_578_, 3);
lean_dec_ref(v___x_572_);
lean_dec_ref(v_x_525_);
lean_dec(v_simpConfigStx_x3f_523_);
lean_dec_ref(v_simpConfig_522_);
lean_dec_ref(v_options_521_);
lean_dec_ref(v_ruleSet_520_);
lean_dec_ref(v_inst_519_);
v_a_637_ = lean_ctor_get(v___x_579_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_579_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_579_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_642_; 
if (v_isShared_640_ == 0)
{
v___x_642_ = v___x_639_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v_a_637_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___redArg___boxed(lean_object* v_inst_651_, lean_object* v_ruleSet_652_, lean_object* v_options_653_, lean_object* v_simpConfig_654_, lean_object* v_simpConfigStx_x3f_655_, lean_object* v_goal_656_, lean_object* v_x_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_aesop_Aesop_SearchM_run___redArg(v_inst_651_, v_ruleSet_652_, v_options_653_, v_simpConfig_654_, v_simpConfigStx_x3f_655_, v_goal_656_, v_x_657_, v_a_658_, v_a_659_, v_a_660_, v_a_661_, v_a_662_);
lean_dec(v_a_662_);
lean_dec_ref(v_a_661_);
lean_dec(v_a_660_);
lean_dec_ref(v_a_659_);
lean_dec(v_a_658_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run(lean_object* v_Q_665_, lean_object* v_inst_666_, lean_object* v_00_u03b1_667_, lean_object* v_ruleSet_668_, lean_object* v_options_669_, lean_object* v_simpConfig_670_, lean_object* v_simpConfigStx_x3f_671_, lean_object* v_goal_672_, lean_object* v_x_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_, lean_object* v_a_678_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lp_aesop_Aesop_SearchM_run___redArg(v_inst_666_, v_ruleSet_668_, v_options_669_, v_simpConfig_670_, v_simpConfigStx_x3f_671_, v_goal_672_, v_x_673_, v_a_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SearchM_run___boxed(lean_object* v_Q_681_, lean_object* v_inst_682_, lean_object* v_00_u03b1_683_, lean_object* v_ruleSet_684_, lean_object* v_options_685_, lean_object* v_simpConfig_686_, lean_object* v_simpConfigStx_x3f_687_, lean_object* v_goal_688_, lean_object* v_x_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_aesop_Aesop_SearchM_run(v_Q_681_, v_inst_682_, v_00_u03b1_683_, v_ruleSet_684_, v_options_685_, v_simpConfig_686_, v_simpConfigStx_x3f_687_, v_goal_688_, v_x_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_, v_a_694_);
lean_dec(v_a_694_);
lean_dec_ref(v_a_693_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
lean_dec(v_a_690_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg(lean_object* v_a_697_, lean_object* v_a_698_){
_start:
{
lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_700_ = lean_st_ref_get(v_a_697_);
lean_dec(v___x_700_);
v___x_701_ = lean_st_ref_get(v_a_698_);
v___x_702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___redArg___boxed(lean_object* v_a_703_, lean_object* v_a_704_, lean_object* v_a_705_){
_start:
{
lean_object* v_res_706_; 
v_res_706_ = lp_aesop_Aesop_getTree___redArg(v_a_703_, v_a_704_);
lean_dec(v_a_704_);
lean_dec(v_a_703_);
return v_res_706_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree(lean_object* v_Q_707_, lean_object* v_inst_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_, lean_object* v_a_716_){
_start:
{
lean_object* v___x_718_; 
v___x_718_ = lp_aesop_Aesop_getTree___redArg(v_a_710_, v_a_711_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getTree___boxed(lean_object* v_Q_719_, lean_object* v_inst_720_, lean_object* v_a_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_aesop_Aesop_getTree(v_Q_719_, v_inst_720_, v_a_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_);
lean_dec(v_a_728_);
lean_dec_ref(v_a_727_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
lean_dec(v_a_724_);
lean_dec(v_a_723_);
lean_dec(v_a_722_);
lean_dec_ref(v_a_721_);
lean_dec_ref(v_inst_720_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg(lean_object* v_s_731_, lean_object* v_a_732_, lean_object* v_a_733_){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_735_ = lean_st_ref_get(v_a_732_);
lean_dec(v___x_735_);
v___x_736_ = lean_st_ref_take(v_a_733_);
lean_dec(v___x_736_);
v___x_737_ = lean_st_ref_set(v_a_733_, v_s_731_);
v___x_738_ = lean_box(0);
v___x_739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___redArg___boxed(lean_object* v_s_740_, lean_object* v_a_741_, lean_object* v_a_742_, lean_object* v_a_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_aesop_Aesop_setTree___redArg(v_s_740_, v_a_741_, v_a_742_);
lean_dec(v_a_742_);
lean_dec(v_a_741_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree(lean_object* v_Q_745_, lean_object* v_inst_746_, lean_object* v_s_747_, lean_object* v_a_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_){
_start:
{
lean_object* v___x_757_; 
v___x_757_ = lp_aesop_Aesop_setTree___redArg(v_s_747_, v_a_749_, v_a_750_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setTree___boxed(lean_object* v_Q_758_, lean_object* v_inst_759_, lean_object* v_s_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v_a_763_, lean_object* v_a_764_, lean_object* v_a_765_, lean_object* v_a_766_, lean_object* v_a_767_, lean_object* v_a_768_, lean_object* v_a_769_){
_start:
{
lean_object* v_res_770_; 
v_res_770_ = lp_aesop_Aesop_setTree(v_Q_758_, v_inst_759_, v_s_760_, v_a_761_, v_a_762_, v_a_763_, v_a_764_, v_a_765_, v_a_766_, v_a_767_, v_a_768_);
lean_dec(v_a_768_);
lean_dec_ref(v_a_767_);
lean_dec(v_a_766_);
lean_dec_ref(v_a_765_);
lean_dec(v_a_764_);
lean_dec(v_a_763_);
lean_dec(v_a_762_);
lean_dec_ref(v_a_761_);
lean_dec_ref(v_inst_759_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg(lean_object* v_f_771_, lean_object* v_a_772_, lean_object* v_a_773_){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; 
v___x_775_ = lean_st_ref_get(v_a_772_);
lean_dec(v___x_775_);
v___x_776_ = lean_st_ref_take(v_a_773_);
v___x_777_ = lean_apply_1(v_f_771_, v___x_776_);
v___x_778_ = lean_st_ref_set(v_a_773_, v___x_777_);
v___x_779_ = lean_box(0);
v___x_780_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_780_, 0, v___x_779_);
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___redArg___boxed(lean_object* v_f_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_aesop_Aesop_modifyTree___redArg(v_f_781_, v_a_782_, v_a_783_);
lean_dec(v_a_783_);
lean_dec(v_a_782_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree(lean_object* v_Q_786_, lean_object* v_inst_787_, lean_object* v_f_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_, lean_object* v_a_795_, lean_object* v_a_796_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_aesop_Aesop_modifyTree___redArg(v_f_788_, v_a_790_, v_a_791_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyTree___boxed(lean_object* v_Q_799_, lean_object* v_inst_800_, lean_object* v_f_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_){
_start:
{
lean_object* v_res_811_; 
v_res_811_ = lp_aesop_Aesop_modifyTree(v_Q_799_, v_inst_800_, v_f_801_, v_a_802_, v_a_803_, v_a_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_a_805_);
lean_dec(v_a_804_);
lean_dec(v_a_803_);
lean_dec_ref(v_a_802_);
lean_dec_ref(v_inst_800_);
return v_res_811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object* v_a_812_){
_start:
{
lean_object* v___x_814_; lean_object* v_iteration_815_; lean_object* v___x_816_; 
v___x_814_ = lean_st_ref_get(v_a_812_);
v_iteration_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_iteration_815_);
lean_dec(v___x_814_);
v___x_816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_816_, 0, v_iteration_815_);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___redArg___boxed(lean_object* v_a_817_, lean_object* v_a_818_){
_start:
{
lean_object* v_res_819_; 
v_res_819_ = lp_aesop_Aesop_getIteration___redArg(v_a_817_);
lean_dec(v_a_817_);
return v_res_819_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration(lean_object* v_Q_820_, lean_object* v_inst_821_, lean_object* v_a_822_, lean_object* v_a_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v___x_831_; 
v___x_831_ = lp_aesop_Aesop_getIteration___redArg(v_a_823_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getIteration___boxed(lean_object* v_Q_832_, lean_object* v_inst_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_aesop_Aesop_getIteration(v_Q_832_, v_inst_833_, v_a_834_, v_a_835_, v_a_836_, v_a_837_, v_a_838_, v_a_839_, v_a_840_, v_a_841_);
lean_dec(v_a_841_);
lean_dec_ref(v_a_840_);
lean_dec(v_a_839_);
lean_dec_ref(v_a_838_);
lean_dec(v_a_837_);
lean_dec(v_a_836_);
lean_dec(v_a_835_);
lean_dec_ref(v_a_834_);
lean_dec_ref(v_inst_833_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object* v_a_844_){
_start:
{
lean_object* v___x_846_; lean_object* v_iteration_847_; lean_object* v_queue_848_; uint8_t v_maxRuleApplicationDepthReached_849_; lean_object* v___x_851_; uint8_t v_isShared_852_; uint8_t v_isSharedCheck_861_; 
v___x_846_ = lean_st_ref_take(v_a_844_);
v_iteration_847_ = lean_ctor_get(v___x_846_, 0);
v_queue_848_ = lean_ctor_get(v___x_846_, 1);
v_maxRuleApplicationDepthReached_849_ = lean_ctor_get_uint8(v___x_846_, sizeof(void*)*2);
v_isSharedCheck_861_ = !lean_is_exclusive(v___x_846_);
if (v_isSharedCheck_861_ == 0)
{
v___x_851_ = v___x_846_;
v_isShared_852_ = v_isSharedCheck_861_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_queue_848_);
lean_inc(v_iteration_847_);
lean_dec(v___x_846_);
v___x_851_ = lean_box(0);
v_isShared_852_ = v_isSharedCheck_861_;
goto v_resetjp_850_;
}
v_resetjp_850_:
{
lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_856_; 
v___x_853_ = lean_unsigned_to_nat(1u);
v___x_854_ = lean_nat_add(v_iteration_847_, v___x_853_);
lean_dec(v_iteration_847_);
if (v_isShared_852_ == 0)
{
lean_ctor_set(v___x_851_, 0, v___x_854_);
v___x_856_ = v___x_851_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_860_; 
v_reuseFailAlloc_860_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_860_, 0, v___x_854_);
lean_ctor_set(v_reuseFailAlloc_860_, 1, v_queue_848_);
lean_ctor_set_uint8(v_reuseFailAlloc_860_, sizeof(void*)*2, v_maxRuleApplicationDepthReached_849_);
v___x_856_ = v_reuseFailAlloc_860_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_857_ = lean_st_ref_set(v_a_844_, v___x_856_);
v___x_858_ = lean_box(0);
v___x_859_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_859_, 0, v___x_858_);
return v___x_859_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___redArg___boxed(lean_object* v_a_862_, lean_object* v_a_863_){
_start:
{
lean_object* v_res_864_; 
v_res_864_ = lp_aesop_Aesop_incrementIteration___redArg(v_a_862_);
lean_dec(v_a_862_);
return v_res_864_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration(lean_object* v_Q_865_, lean_object* v_inst_866_, lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_a_869_, lean_object* v_a_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lp_aesop_Aesop_incrementIteration___redArg(v_a_868_);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_incrementIteration___boxed(lean_object* v_Q_877_, lean_object* v_inst_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_, lean_object* v_a_882_, lean_object* v_a_883_, lean_object* v_a_884_, lean_object* v_a_885_, lean_object* v_a_886_, lean_object* v_a_887_){
_start:
{
lean_object* v_res_888_; 
v_res_888_ = lp_aesop_Aesop_incrementIteration(v_Q_877_, v_inst_878_, v_a_879_, v_a_880_, v_a_881_, v_a_882_, v_a_883_, v_a_884_, v_a_885_, v_a_886_);
lean_dec(v_a_886_);
lean_dec_ref(v_a_885_);
lean_dec(v_a_884_);
lean_dec_ref(v_a_883_);
lean_dec(v_a_882_);
lean_dec(v_a_881_);
lean_dec(v_a_880_);
lean_dec_ref(v_a_879_);
lean_dec_ref(v_inst_878_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object* v_inst_889_, lean_object* v_a_890_){
_start:
{
lean_object* v___x_892_; lean_object* v_popGoal_893_; lean_object* v___x_894_; lean_object* v_iteration_895_; lean_object* v_queue_896_; uint8_t v_maxRuleApplicationDepthReached_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_909_; 
v___x_892_ = lean_st_ref_get(v_a_890_);
v_popGoal_893_ = lean_ctor_get(v_inst_889_, 2);
lean_inc_ref(v_popGoal_893_);
lean_dec_ref(v_inst_889_);
v___x_894_ = lean_st_ref_get(v_a_890_);
lean_dec(v___x_894_);
v_iteration_895_ = lean_ctor_get(v___x_892_, 0);
v_queue_896_ = lean_ctor_get(v___x_892_, 1);
v_maxRuleApplicationDepthReached_897_ = lean_ctor_get_uint8(v___x_892_, sizeof(void*)*2);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_892_);
if (v_isSharedCheck_909_ == 0)
{
v___x_899_ = v___x_892_;
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_queue_896_);
lean_inc(v_iteration_895_);
lean_dec(v___x_892_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v___x_901_; lean_object* v_fst_902_; lean_object* v_snd_903_; lean_object* v___x_905_; 
v___x_901_ = lean_apply_2(v_popGoal_893_, v_queue_896_, lean_box(0));
v_fst_902_ = lean_ctor_get(v___x_901_, 0);
lean_inc(v_fst_902_);
v_snd_903_ = lean_ctor_get(v___x_901_, 1);
lean_inc(v_snd_903_);
lean_dec_ref(v___x_901_);
if (v_isShared_900_ == 0)
{
lean_ctor_set(v___x_899_, 1, v_snd_903_);
v___x_905_ = v___x_899_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v_iteration_895_);
lean_ctor_set(v_reuseFailAlloc_908_, 1, v_snd_903_);
lean_ctor_set_uint8(v_reuseFailAlloc_908_, sizeof(void*)*2, v_maxRuleApplicationDepthReached_897_);
v___x_905_ = v_reuseFailAlloc_908_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
lean_object* v___x_906_; lean_object* v___x_907_; 
v___x_906_ = lean_st_ref_set(v_a_890_, v___x_905_);
v___x_907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_907_, 0, v_fst_902_);
return v___x_907_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___redArg___boxed(lean_object* v_inst_910_, lean_object* v_a_911_, lean_object* v_a_912_){
_start:
{
lean_object* v_res_913_; 
v_res_913_ = lp_aesop_Aesop_popGoal_x3f___redArg(v_inst_910_, v_a_911_);
lean_dec(v_a_911_);
return v_res_913_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f(lean_object* v_Q_914_, lean_object* v_inst_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_, lean_object* v_a_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_, lean_object* v_a_923_){
_start:
{
lean_object* v___x_925_; 
v___x_925_ = lp_aesop_Aesop_popGoal_x3f___redArg(v_inst_915_, v_a_917_);
return v___x_925_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_popGoal_x3f___boxed(lean_object* v_Q_926_, lean_object* v_inst_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_, lean_object* v_a_931_, lean_object* v_a_932_, lean_object* v_a_933_, lean_object* v_a_934_, lean_object* v_a_935_, lean_object* v_a_936_){
_start:
{
lean_object* v_res_937_; 
v_res_937_ = lp_aesop_Aesop_popGoal_x3f(v_Q_926_, v_inst_927_, v_a_928_, v_a_929_, v_a_930_, v_a_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_);
lean_dec(v_a_935_);
lean_dec_ref(v_a_934_);
lean_dec(v_a_933_);
lean_dec_ref(v_a_932_);
lean_dec(v_a_931_);
lean_dec(v_a_930_);
lean_dec(v_a_929_);
lean_dec_ref(v_a_928_);
return v_res_937_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object* v_inst_938_, lean_object* v_gs_939_, lean_object* v_a_940_){
_start:
{
lean_object* v___x_942_; lean_object* v_addGoals_943_; lean_object* v___x_944_; lean_object* v_iteration_945_; lean_object* v_queue_946_; uint8_t v_maxRuleApplicationDepthReached_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_957_; 
v___x_942_ = lean_st_ref_get(v_a_940_);
v_addGoals_943_ = lean_ctor_get(v_inst_938_, 1);
lean_inc_ref(v_addGoals_943_);
lean_dec_ref(v_inst_938_);
v___x_944_ = lean_st_ref_get(v_a_940_);
lean_dec(v___x_944_);
v_iteration_945_ = lean_ctor_get(v___x_942_, 0);
v_queue_946_ = lean_ctor_get(v___x_942_, 1);
v_maxRuleApplicationDepthReached_947_ = lean_ctor_get_uint8(v___x_942_, sizeof(void*)*2);
v_isSharedCheck_957_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_957_ == 0)
{
v___x_949_ = v___x_942_;
v_isShared_950_ = v_isSharedCheck_957_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_queue_946_);
lean_inc(v_iteration_945_);
lean_dec(v___x_942_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_957_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
lean_object* v___x_951_; lean_object* v___x_953_; 
v___x_951_ = lean_apply_3(v_addGoals_943_, v_queue_946_, v_gs_939_, lean_box(0));
if (v_isShared_950_ == 0)
{
lean_ctor_set(v___x_949_, 1, v___x_951_);
v___x_953_ = v___x_949_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v_iteration_945_);
lean_ctor_set(v_reuseFailAlloc_956_, 1, v___x_951_);
lean_ctor_set_uint8(v_reuseFailAlloc_956_, sizeof(void*)*2, v_maxRuleApplicationDepthReached_947_);
v___x_953_ = v_reuseFailAlloc_956_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
lean_object* v___x_954_; lean_object* v___x_955_; 
v___x_954_ = lean_st_ref_set(v_a_940_, v___x_953_);
v___x_955_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_955_, 0, v___x_954_);
return v___x_955_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___redArg___boxed(lean_object* v_inst_958_, lean_object* v_gs_959_, lean_object* v_a_960_, lean_object* v_a_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_958_, v_gs_959_, v_a_960_);
lean_dec(v_a_960_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals(lean_object* v_Q_963_, lean_object* v_inst_964_, lean_object* v_gs_965_, lean_object* v_a_966_, lean_object* v_a_967_, lean_object* v_a_968_, lean_object* v_a_969_, lean_object* v_a_970_, lean_object* v_a_971_, lean_object* v_a_972_, lean_object* v_a_973_){
_start:
{
lean_object* v___x_975_; 
v___x_975_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_964_, v_gs_965_, v_a_967_);
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enqueueGoals___boxed(lean_object* v_Q_976_, lean_object* v_inst_977_, lean_object* v_gs_978_, lean_object* v_a_979_, lean_object* v_a_980_, lean_object* v_a_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_){
_start:
{
lean_object* v_res_988_; 
v_res_988_ = lp_aesop_Aesop_enqueueGoals(v_Q_976_, v_inst_977_, v_gs_978_, v_a_979_, v_a_980_, v_a_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_, v_a_986_);
lean_dec(v_a_986_);
lean_dec_ref(v_a_985_);
lean_dec(v_a_984_);
lean_dec_ref(v_a_983_);
lean_dec(v_a_982_);
lean_dec(v_a_981_);
lean_dec(v_a_980_);
lean_dec_ref(v_a_979_);
return v_res_988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object* v_a_989_){
_start:
{
lean_object* v___x_991_; lean_object* v_iteration_992_; lean_object* v_queue_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1004_; 
v___x_991_ = lean_st_ref_take(v_a_989_);
v_iteration_992_ = lean_ctor_get(v___x_991_, 0);
v_queue_993_ = lean_ctor_get(v___x_991_, 1);
v_isSharedCheck_1004_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1004_ == 0)
{
v___x_995_ = v___x_991_;
v_isShared_996_ = v_isSharedCheck_1004_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_queue_993_);
lean_inc(v_iteration_992_);
lean_dec(v___x_991_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1004_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
uint8_t v___x_997_; lean_object* v___x_999_; 
v___x_997_ = 1;
if (v_isShared_996_ == 0)
{
v___x_999_ = v___x_995_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_iteration_992_);
lean_ctor_set(v_reuseFailAlloc_1003_, 1, v_queue_993_);
v___x_999_ = v_reuseFailAlloc_1003_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; 
lean_ctor_set_uint8(v___x_999_, sizeof(void*)*2, v___x_997_);
v___x_1000_ = lean_st_ref_set(v_a_989_, v___x_999_);
v___x_1001_ = lean_box(0);
v___x_1002_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
return v___x_1002_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg___boxed(lean_object* v_a_1005_, lean_object* v_a_1006_){
_start:
{
lean_object* v_res_1007_; 
v_res_1007_ = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(v_a_1005_);
lean_dec(v_a_1005_);
return v_res_1007_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached(lean_object* v_Q_1008_, lean_object* v_inst_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_, lean_object* v_a_1017_){
_start:
{
lean_object* v___x_1019_; 
v___x_1019_ = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(v_a_1011_);
return v___x_1019_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___boxed(lean_object* v_Q_1020_, lean_object* v_inst_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v_res_1031_; 
v_res_1031_ = lp_aesop_Aesop_setMaxRuleApplicationDepthReached(v_Q_1020_, v_inst_1021_, v_a_1022_, v_a_1023_, v_a_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
lean_dec(v_a_1029_);
lean_dec_ref(v_a_1028_);
lean_dec(v_a_1027_);
lean_dec_ref(v_a_1026_);
lean_dec(v_a_1025_);
lean_dec(v_a_1024_);
lean_dec(v_a_1023_);
lean_dec_ref(v_a_1022_);
lean_dec_ref(v_inst_1021_);
return v_res_1031_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object* v_a_1032_){
_start:
{
lean_object* v___x_1034_; uint8_t v_maxRuleApplicationDepthReached_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; 
v___x_1034_ = lean_st_ref_get(v_a_1032_);
v_maxRuleApplicationDepthReached_1035_ = lean_ctor_get_uint8(v___x_1034_, sizeof(void*)*2);
lean_dec(v___x_1034_);
v___x_1036_ = lean_box(v_maxRuleApplicationDepthReached_1035_);
v___x_1037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1037_, 0, v___x_1036_);
return v___x_1037_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg___boxed(lean_object* v_a_1038_, lean_object* v_a_1039_){
_start:
{
lean_object* v_res_1040_; 
v_res_1040_ = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(v_a_1038_);
lean_dec(v_a_1038_);
return v_res_1040_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(lean_object* v_Q_1041_, lean_object* v_inst_1042_, lean_object* v_a_1043_, lean_object* v_a_1044_, lean_object* v_a_1045_, lean_object* v_a_1046_, lean_object* v_a_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_){
_start:
{
lean_object* v___x_1052_; 
v___x_1052_ = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(v_a_1044_);
return v___x_1052_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___boxed(lean_object* v_Q_1053_, lean_object* v_inst_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_, lean_object* v_a_1057_, lean_object* v_a_1058_, lean_object* v_a_1059_, lean_object* v_a_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_, lean_object* v_a_1063_){
_start:
{
lean_object* v_res_1064_; 
v_res_1064_ = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached(v_Q_1053_, v_inst_1054_, v_a_1055_, v_a_1056_, v_a_1057_, v_a_1058_, v_a_1059_, v_a_1060_, v_a_1061_, v_a_1062_);
lean_dec(v_a_1062_);
lean_dec_ref(v_a_1061_);
lean_dec(v_a_1060_);
lean_dec_ref(v_a_1059_);
lean_dec(v_a_1058_);
lean_dec(v_a_1057_);
lean_dec(v_a_1056_);
lean_dec_ref(v_a_1055_);
lean_dec_ref(v_inst_1054_);
return v_res_1064_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_SearchM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedNormSimpContext_default = _init_lp_aesop_Aesop_instInhabitedNormSimpContext_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext_default);
lp_aesop_Aesop_instInhabitedNormSimpContext = _init_lp_aesop_Aesop_instInhabitedNormSimpContext();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNormSimpContext);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_SearchM(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_SearchM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_SearchM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_SearchM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_SearchM(builtin);
}
#ifdef __cplusplus
}
#endif
