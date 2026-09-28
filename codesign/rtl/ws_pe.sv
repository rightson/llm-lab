// Original educational PE. Numeric semantics are documented in ../README.md.
// Explicit clock-enable is a proposed extension, not the upstream nanoNPU port ABI.
module ws_pe (
    input logic clk, rst_n, ce, load_w,
    input logic signed [7:0] act_in, weight_in,
    input logic signed [31:0] psum_in,
    output logic signed [7:0] act_out,
    output logic signed [31:0] psum_out
);
    logic signed [7:0] weight;
    logic signed [15:0] product;
    logic signed [31:0] extended_product;
    assign product = act_in * weight;
    assign extended_product = {{16{product[15]}},product};
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight <= '0;
            act_out <= '0;
            psum_out <= '0;
        end else if (ce) begin
            if (load_w) weight <= weight_in;
            else begin
                act_out <= act_in;
                psum_out <= psum_in + extended_product;
            end
        end
    end
endmodule
