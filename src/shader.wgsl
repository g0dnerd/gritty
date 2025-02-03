struct MatrixSize {
  width_a: u32,
  height_a: u32,
  width_b: u32,
};

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> matrix_size: MatrixSize;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;

  if x >= matrix_size.width_b || y >= matrix_size.height_a {
    return;
  }

  var sum: f32 = 0.0;
  for (var k = 0u; k < matrix_size.width_a; k = k + 1u) {
    let a_idx = y * matrix_size.width_a + k;
    let b_idx = k * matrix_size.width_b + x;
    sum = sum + matrixA[a_idx] * matrixB[b_idx];
  }

  let output_idx = y * matrix_size.width_b + x;
  result[output_idx] = sum;
}
