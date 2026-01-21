// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vmatrix_mult_4x4x4x16.h for the primary calling header

#include "Vmatrix_mult_4x4x4x16__pch.h"
#include "Vmatrix_mult_4x4x4x16___024root.h"

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_static(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_initial(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_initial\n"); );
    // Body
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_final(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vmatrix_mult_4x4x4x16___024root___eval_phase__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf);

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_settle(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("/mnt/c/CODE/dlp_ex_student/src/matrix_mult_4x4x4x16.v", 1, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vmatrix_mult_4x4x4x16___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___stl_sequent__TOP__0(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___stl_sequent__TOP__0\n"); );
    // Body
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[0U] 
        = vlSelf->activations[0U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[1U] 
        = vlSelf->activations[0U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[2U] 
        = vlSelf->activations[0U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[3U] 
        = vlSelf->activations[0U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[0U] 
        = vlSelf->activations[1U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[1U] 
        = vlSelf->activations[1U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[2U] 
        = vlSelf->activations[1U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[3U] 
        = vlSelf->activations[1U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[0U] 
        = vlSelf->activations[2U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[1U] 
        = vlSelf->activations[2U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[2U] 
        = vlSelf->activations[2U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[3U] 
        = vlSelf->activations[2U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[0U] 
        = vlSelf->activations[3U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[1U] 
        = vlSelf->activations[3U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[2U] 
        = vlSelf->activations[3U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[3U] 
        = vlSelf->activations[3U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[0U] 
        = vlSelf->weights[0U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[1U] 
        = vlSelf->weights[0U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[2U] 
        = vlSelf->weights[0U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[3U] 
        = vlSelf->weights[0U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[0U] 
        = vlSelf->weights[1U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[1U] 
        = vlSelf->weights[1U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[2U] 
        = vlSelf->weights[1U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[3U] 
        = vlSelf->weights[1U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[0U] 
        = vlSelf->weights[2U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[1U] 
        = vlSelf->weights[2U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[2U] 
        = vlSelf->weights[2U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[3U] 
        = vlSelf->weights[2U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[0U] 
        = vlSelf->weights[3U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[1U] 
        = vlSelf->weights[3U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[2U] 
        = vlSelf->weights[3U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[3U] 
        = vlSelf->weights[3U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[0U] 
        = vlSelf->weights[0U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[1U] 
        = vlSelf->weights[0U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[2U] 
        = vlSelf->weights[0U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[3U] 
        = vlSelf->weights[0U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[0U] 
        = vlSelf->weights[1U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[1U] 
        = vlSelf->weights[1U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[2U] 
        = vlSelf->weights[1U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[3U] 
        = vlSelf->weights[1U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[0U] 
        = vlSelf->weights[2U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[1U] 
        = vlSelf->weights[2U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[2U] 
        = vlSelf->weights[2U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[3U] 
        = vlSelf->weights[2U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[0U] 
        = vlSelf->weights[3U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[1U] 
        = vlSelf->weights[3U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[2U] 
        = vlSelf->weights[3U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[3U] 
        = vlSelf->weights[3U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[0U] 
        = vlSelf->weights[0U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[1U] 
        = vlSelf->weights[0U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[2U] 
        = vlSelf->weights[0U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[3U] 
        = vlSelf->weights[0U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[0U] 
        = vlSelf->weights[1U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[1U] 
        = vlSelf->weights[1U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[2U] 
        = vlSelf->weights[1U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[3U] 
        = vlSelf->weights[1U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[0U] 
        = vlSelf->weights[2U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[1U] 
        = vlSelf->weights[2U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[2U] 
        = vlSelf->weights[2U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[3U] 
        = vlSelf->weights[2U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[0U] 
        = vlSelf->weights[3U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[1U] 
        = vlSelf->weights[3U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[2U] 
        = vlSelf->weights[3U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[3U] 
        = vlSelf->weights[3U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[0U] 
        = vlSelf->weights[0U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[1U] 
        = vlSelf->weights[0U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[2U] 
        = vlSelf->weights[0U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[3U] 
        = vlSelf->weights[0U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[0U] 
        = vlSelf->weights[1U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[1U] 
        = vlSelf->weights[1U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[2U] 
        = vlSelf->weights[1U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[3U] 
        = vlSelf->weights[1U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[0U] 
        = vlSelf->weights[2U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[1U] 
        = vlSelf->weights[2U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[2U] 
        = vlSelf->weights[2U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[3U] 
        = vlSelf->weights[2U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[0U] 
        = vlSelf->weights[3U][0U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[1U] 
        = vlSelf->weights[3U][1U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[2U] 
        = vlSelf->weights[3U][2U];
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[3U] 
        = vlSelf->weights[3U][3U];
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[0U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu0__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[1U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu1__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[2U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu2__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[3U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu3__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[0U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu0__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[1U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu1__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[2U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu2__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[3U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu3__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[0U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu0__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[1U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu1__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[2U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu2__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[3U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu3__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[0U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu0__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[1U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu1__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[2U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu2__result;
    vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[3U] 
        = vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu3__result;
    vlSelf->results[0U][0U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results
        [0U];
    vlSelf->results[0U][1U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results
        [1U];
    vlSelf->results[0U][2U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results
        [2U];
    vlSelf->results[0U][3U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results
        [3U];
    vlSelf->results[1U][0U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results
        [0U];
    vlSelf->results[1U][1U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results
        [1U];
    vlSelf->results[1U][2U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results
        [2U];
    vlSelf->results[1U][3U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results
        [3U];
    vlSelf->results[2U][0U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results
        [0U];
    vlSelf->results[2U][1U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results
        [1U];
    vlSelf->results[2U][2U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results
        [2U];
    vlSelf->results[2U][3U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results
        [3U];
    vlSelf->results[3U][0U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results
        [0U];
    vlSelf->results[3U][1U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results
        [1U];
    vlSelf->results[3U][2U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results
        [2U];
    vlSelf->results[3U][3U] = vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results
        [3U];
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_stl(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vmatrix_mult_4x4x4x16___024root___stl_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        vlSelf->__Vm_traceActivity[1U] = 1U;
        vlSelf->__Vm_traceActivity[0U] = 1U;
    }
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_triggers__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf);

VL_ATTR_COLD bool Vmatrix_mult_4x4x4x16___024root___eval_phase__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vmatrix_mult_4x4x4x16___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vmatrix_mult_4x4x4x16___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__ico(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__act(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__nba(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___ctor_var_reset(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->reset = VL_RAND_RESET_I(1);
    vlSelf->enable = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 4; ++__Vi1) {
            vlSelf->activations[__Vi0][__Vi1] = VL_RAND_RESET_I(16);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 4; ++__Vi1) {
            vlSelf->weights[__Vi0][__Vi1] = VL_RAND_RESET_I(16);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 4; ++__Vi1) {
            vlSelf->results[__Vi0][__Vi1] = VL_RAND_RESET_I(32);
        }
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu0__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu1__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu2__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu3__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu0__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu1__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu2__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu3__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu0__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu1__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu2__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu3__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu0__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu1__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu2__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu3__result = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__i = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[__Vi0] = VL_RAND_RESET_I(16);
    }
    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
