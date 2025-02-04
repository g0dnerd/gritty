@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;

@compute @workgroup_size(1024)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  tensor[idx] = max(0.0, tensor[idx]);
}
