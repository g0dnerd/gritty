pub mod compute;

#[cfg(test)]
mod tests {
    use crate::compute::GpuCompute;
    use pollster::block_on;
    use rand::Rng;

    const SIZE: usize = 64;

    // A simple CPU implementation of matrix multiplication for verification
    fn cpu_matrix_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; SIZE * SIZE];
        for i in 0..SIZE {
            for j in 0..SIZE {
                for k in 0..SIZE {
                    result[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];
                }
            }
        }
        result
    }

    // Helper function to generate a random 64x64 matrix
    fn generate_random_matrix() -> Vec<f32> {
        let mut rng = rand::rng();
        (0..SIZE * SIZE)
            .map(|_| rng.random_range(-10.0..10.0)) // Random values between -10 and 10
            .collect()
    }

    #[test]
    fn matrix_mul_1() {
        let gpu_compute = block_on(GpuCompute::new()).unwrap();
        let matrix_a = vec![1.0; SIZE * SIZE];
        let matrix_b = matrix_a.clone();
        let result = block_on(gpu_compute.run(&matrix_a, &matrix_b, SIZE));
        assert_eq!(result, [SIZE as f32; SIZE * SIZE]);
    }

    #[test]
    fn identity_matrix() {
        let gpu_compute = block_on(GpuCompute::new()).unwrap();
        let mut matrix_a = Vec::new();
        let mut identity_matrix = vec![0.0; SIZE * SIZE];

        for (y, v) in identity_matrix.iter_mut().enumerate().take(SIZE * SIZE) {
            matrix_a.push(y as f32);
            if y % (SIZE + 1) == 0 {
                *v = 1.0;
            }
        }

        let expected_result = matrix_a.clone();
        let result = block_on(gpu_compute.run(&matrix_a, &identity_matrix, SIZE));

        assert_eq!(result, expected_result);
    }

    #[test]
    fn random_matrices() {
        let matrix_a = generate_random_matrix();
        let matrix_b = generate_random_matrix();

        let gpu_compute = block_on(GpuCompute::new()).unwrap();

        let result = block_on(gpu_compute.run(&matrix_a, &matrix_b, SIZE));
        let cpu_result = cpu_matrix_multiply(&matrix_a, &matrix_b);

        assert_eq!(result, cpu_result);
    }
}
