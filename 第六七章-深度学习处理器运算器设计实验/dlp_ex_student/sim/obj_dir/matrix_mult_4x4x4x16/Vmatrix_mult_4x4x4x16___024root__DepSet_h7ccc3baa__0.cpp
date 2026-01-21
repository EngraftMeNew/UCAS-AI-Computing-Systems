// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vmatrix_mult_4x4x4x16.h for the primary calling header

#include "Vmatrix_mult_4x4x4x16__pch.h"
#include "Vmatrix_mult_4x4x4x16__Syms.h"
#include "Vmatrix_mult_4x4x4x16___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__ico(Vmatrix_mult_4x4x4x16___024root* vlSelf);
#endif  // VL_DEBUG

void Vmatrix_mult_4x4x4x16___024root___eval_triggers__ico(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_triggers__ico\n"); );
    // Body
    vlSelf->__VicoTriggered.set(0U, (IData)(vlSelf->__VicoFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vmatrix_mult_4x4x4x16___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__act(Vmatrix_mult_4x4x4x16___024root* vlSelf);
#endif  // VL_DEBUG

void Vmatrix_mult_4x4x4x16___024root___eval_triggers__act(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_triggers__act\n"); );
    // Body
    vlSelf->__VactTriggered.set(0U, ((IData)(vlSelf->clk) 
                                     & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__clk__0))));
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vmatrix_mult_4x4x4x16___024root___dump_triggers__act(vlSelf);
    }
#endif
}
