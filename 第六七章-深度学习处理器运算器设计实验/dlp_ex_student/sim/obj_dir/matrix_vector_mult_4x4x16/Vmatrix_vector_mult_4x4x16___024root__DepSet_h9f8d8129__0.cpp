// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vmatrix_vector_mult_4x4x16.h for the primary calling header

#include "Vmatrix_vector_mult_4x4x16__pch.h"
#include "Vmatrix_vector_mult_4x4x16___024root.h"

VL_INLINE_OPT void Vmatrix_vector_mult_4x4x16___024root___ico_sequent__TOP__0(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[0U] 
        = vlSelf->weights[0U][0U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[1U] 
        = vlSelf->weights[0U][1U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[2U] 
        = vlSelf->weights[0U][2U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[3U] 
        = vlSelf->weights[0U][3U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[0U] 
        = vlSelf->weights[1U][0U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[1U] 
        = vlSelf->weights[1U][1U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[2U] 
        = vlSelf->weights[1U][2U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[3U] 
        = vlSelf->weights[1U][3U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[0U] 
        = vlSelf->weights[2U][0U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[1U] 
        = vlSelf->weights[2U][1U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[2U] 
        = vlSelf->weights[2U][2U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[3U] 
        = vlSelf->weights[2U][3U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[0U] 
        = vlSelf->weights[3U][0U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[1U] 
        = vlSelf->weights[3U][1U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[2U] 
        = vlSelf->weights[3U][2U];
    vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[3U] 
        = vlSelf->weights[3U][3U];
}

void Vmatrix_vector_mult_4x4x16___024root___eval_ico(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vmatrix_vector_mult_4x4x16___024root___ico_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[1U] = 1U;
    }
}

void Vmatrix_vector_mult_4x4x16___024root___eval_triggers__ico(Vmatrix_vector_mult_4x4x16___024root* vlSelf);

bool Vmatrix_vector_mult_4x4x16___024root___eval_phase__ico(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    Vmatrix_vector_mult_4x4x16___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        Vmatrix_vector_mult_4x4x16___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void Vmatrix_vector_mult_4x4x16___024root___eval_act(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vmatrix_vector_mult_4x4x16___024root___nba_sequent__TOP__0(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___nba_sequent__TOP__0\n"); );
    // Init
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v0;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v0 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v5;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v5 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v6;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v6 = 0;
    SData/*15:0*/ __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v7;
    __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v7 = 0;
    // Body
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v0 = 0U;
    __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4 = 0U;
    if (vlSelf->reset) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__i = 4U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v0 = 1U;
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v0 = 1U;
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result = 0U;
    } else if (vlSelf->enable) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__i = 4U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__i = 4U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4 
            = vlSelf->activations[0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v5 
            = vlSelf->activations[1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v6 
            = vlSelf->activations[2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v7 
            = vlSelf->activations[3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4 
            = vlSelf->activations[0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v5 
            = vlSelf->activations[1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v6 
            = vlSelf->activations[2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v7 
            = vlSelf->activations[3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4 
            = vlSelf->activations[0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v5 
            = vlSelf->activations[1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v6 
            = vlSelf->activations[2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v7 
            = vlSelf->activations[3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4 
            = vlSelf->activations[0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v5 
            = vlSelf->activations[1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v6 
            = vlSelf->activations[2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v7 
            = vlSelf->activations[3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights
            [0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v5 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights
            [1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v6 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights
            [2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v7 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights
            [3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights
            [0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v5 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights
            [1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v6 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights
            [2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v7 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights
            [3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights
            [0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v5 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights
            [1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v6 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights
            [2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v7 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights
            [3U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights
            [0U];
        __Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4 = 1U;
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v5 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights
            [1U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v6 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights
            [2U];
        __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v7 
            = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights
            [3U];
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result 
            = (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                             [0U]), 
                           VL_EXTENDS_II(32,16, vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                         [0U])) + (
                                                   VL_MULS_III(32, 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                             [1U]), 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                             [1U])) 
                                                   + 
                                                   (VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                              [2U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                              [2U])) 
                                                    + 
                                                    VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                              [3U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                              [3U])))));
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result 
            = (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                             [0U]), 
                           VL_EXTENDS_II(32,16, vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                         [0U])) + (
                                                   VL_MULS_III(32, 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                             [1U]), 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                             [1U])) 
                                                   + 
                                                   (VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                              [2U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                              [2U])) 
                                                    + 
                                                    VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                              [3U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                              [3U])))));
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result 
            = (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                             [0U]), 
                           VL_EXTENDS_II(32,16, vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                         [0U])) + (
                                                   VL_MULS_III(32, 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                             [1U]), 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                             [1U])) 
                                                   + 
                                                   (VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                              [2U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                              [2U])) 
                                                    + 
                                                    VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                              [3U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                              [3U])))));
        vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result 
            = (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                             [0U]), 
                           VL_EXTENDS_II(32,16, vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                         [0U])) + (
                                                   VL_MULS_III(32, 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                             [1U]), 
                                                               VL_EXTENDS_II(32,16, 
                                                                             vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                             [1U])) 
                                                   + 
                                                   (VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                              [2U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                              [2U])) 
                                                    + 
                                                    VL_MULS_III(32, 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                              [3U]), 
                                                                VL_EXTENDS_II(32,16, 
                                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                              [3U])))));
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg__v7;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v0) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[0U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[1U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[2U] = 0U;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[3U] = 0U;
    }
    if (__Vdlyvset__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4) {
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[0U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v4;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[1U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v5;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[2U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v6;
        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[3U] 
            = __Vdlyvval__matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg__v7;
    }
    vlSelf->results[3U] = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result;
    vlSelf->results[2U] = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result;
    vlSelf->results[1U] = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result;
    vlSelf->results[0U] = vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result;
}

void Vmatrix_vector_mult_4x4x16___024root___eval_nba(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vmatrix_vector_mult_4x4x16___024root___nba_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
}

void Vmatrix_vector_mult_4x4x16___024root___eval_triggers__act(Vmatrix_vector_mult_4x4x16___024root* vlSelf);

bool Vmatrix_vector_mult_4x4x16___024root___eval_phase__act(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vmatrix_vector_mult_4x4x16___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vmatrix_vector_mult_4x4x16___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vmatrix_vector_mult_4x4x16___024root___eval_phase__nba(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vmatrix_vector_mult_4x4x16___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root___dump_triggers__ico(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root___dump_triggers__nba(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root___dump_triggers__act(Vmatrix_vector_mult_4x4x16___024root* vlSelf);
#endif  // VL_DEBUG

void Vmatrix_vector_mult_4x4x16___024root___eval(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            Vmatrix_vector_mult_4x4x16___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/matrix_vector_mult_4x4x16.v", 1, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (Vmatrix_vector_mult_4x4x16___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vmatrix_vector_mult_4x4x16___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/matrix_vector_mult_4x4x16.v", 1, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vmatrix_vector_mult_4x4x16___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/matrix_vector_mult_4x4x16.v", 1, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vmatrix_vector_mult_4x4x16___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vmatrix_vector_mult_4x4x16___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vmatrix_vector_mult_4x4x16___024root___eval_debug_assertions(Vmatrix_vector_mult_4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->reset & 0xfeU))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY((vlSelf->enable & 0xfeU))) {
        Verilated::overWidthError("enable");}
}
#endif  // VL_DEBUG
