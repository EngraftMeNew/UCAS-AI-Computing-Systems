// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vinner_product_4x16__Syms.h"


void Vinner_product_4x16___024root__trace_chg_0_sub_0(Vinner_product_4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vinner_product_4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_chg_0\n"); );
    // Init
    Vinner_product_4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vinner_product_4x16___024root*>(voidSelf);
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vinner_product_4x16___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vinner_product_4x16___024root__trace_chg_0_sub_0(Vinner_product_4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[1U])) {
        bufp->chgSData(oldp+0,(vlSelf->inner_product_4x16__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+1,(vlSelf->inner_product_4x16__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+2,(vlSelf->inner_product_4x16__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+3,(vlSelf->inner_product_4x16__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+4,(vlSelf->inner_product_4x16__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+5,(vlSelf->inner_product_4x16__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+6,(vlSelf->inner_product_4x16__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+7,(vlSelf->inner_product_4x16__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+8,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                              vlSelf->inner_product_4x16__DOT__activations_reg
                                                              [0U]), 
                                            VL_EXTENDS_II(32,16, 
                                                          vlSelf->inner_product_4x16__DOT__weights_reg
                                                          [0U])) 
                                + (VL_MULS_III(32, 
                                               VL_EXTENDS_II(32,16, 
                                                             vlSelf->inner_product_4x16__DOT__activations_reg
                                                             [1U]), 
                                               VL_EXTENDS_II(32,16, 
                                                             vlSelf->inner_product_4x16__DOT__weights_reg
                                                             [1U])) 
                                   + (VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->inner_product_4x16__DOT__activations_reg
                                                                [2U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->inner_product_4x16__DOT__weights_reg
                                                                [2U])) 
                                      + VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->inner_product_4x16__DOT__activations_reg
                                                                  [3U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->inner_product_4x16__DOT__weights_reg
                                                                  [3U])))))),32);
        bufp->chgIData(oldp+9,(vlSelf->inner_product_4x16__DOT__i),32);
    }
    bufp->chgBit(oldp+10,(vlSelf->clk));
    bufp->chgBit(oldp+11,(vlSelf->reset));
    bufp->chgBit(oldp+12,(vlSelf->enable));
    bufp->chgSData(oldp+13,(vlSelf->activations[0]),16);
    bufp->chgSData(oldp+14,(vlSelf->activations[1]),16);
    bufp->chgSData(oldp+15,(vlSelf->activations[2]),16);
    bufp->chgSData(oldp+16,(vlSelf->activations[3]),16);
    bufp->chgSData(oldp+17,(vlSelf->weights[0]),16);
    bufp->chgSData(oldp+18,(vlSelf->weights[1]),16);
    bufp->chgSData(oldp+19,(vlSelf->weights[2]),16);
    bufp->chgSData(oldp+20,(vlSelf->weights[3]),16);
    bufp->chgIData(oldp+21,(vlSelf->result),32);
}

void Vinner_product_4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_cleanup\n"); );
    // Init
    Vinner_product_4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vinner_product_4x16___024root*>(voidSelf);
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
}
