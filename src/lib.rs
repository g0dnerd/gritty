pub mod compute;

#[cfg(test)]
mod tests {
    use crate::compute::GpuCompute;
    use pollster::block_on;

    #[test]
    fn matrix_mul_1() {
        let gpu_compute = block_on(GpuCompute::new()).unwrap();
        let matrix_a = vec![1.0; 64 * 64];
        let matrix_b = vec![1.0; 64 * 64];
        let result = block_on(gpu_compute.run(&matrix_a, &matrix_b, 64));
        assert_eq!(result, [64.0; 64 * 64]);
    }

    #[test]
    fn identity_matrix() {
        let gpu_compute = block_on(GpuCompute::new()).unwrap();
        let matrix_a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let identity_matrix = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let expected_result = matrix_a.clone();
        let result = block_on(gpu_compute.run(&matrix_a, &identity_matrix, 4));

        assert_eq!(result, expected_result);
    }
}
