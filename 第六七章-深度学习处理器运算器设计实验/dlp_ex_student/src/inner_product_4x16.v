module inner_product_4x16 (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire signed [15:0] activations [0:3],
    input wire signed [15:0] weights [0:3],
    output reg signed [31:0] result
);
    // 输入寄存器
    reg signed [15:0] activations_reg [0:3];
    reg signed [15:0] weights_reg [0:3];
    
    // 组合逻辑部分
    wire signed [31:0] dot_product = 
        (activations_reg[0] * weights_reg[0]) +
        (activations_reg[1] * weights_reg[1]) +
        (activations_reg[2] * weights_reg[2]) +
        (activations_reg[3] * weights_reg[3]);
    
    // 寄存器更新逻辑
    integer i;
    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < 4; i = i + 1) begin
                activations_reg[i] <= 16'd0;
                weights_reg[i] <= 16'd0;
            end
            result <= 32'd0;
        end else if (enable) begin
            for (i = 0; i < 4; i = i + 1) begin
                activations_reg[i] <= activations[i];
                weights_reg[i] <= weights[i];
            end
            result <= dot_product;
        end
    end
endmodule
