@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  let size = 64u;

  var sum: f32 = 0.0;
  for (var k = 0u; k < size; k = k + 1u) {
    sum = sum + matrixA[row * size + k] * matrixB[k * size + col];
  }

  result[row * size + col] = sum;
}
