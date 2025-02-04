#![feature(test)]

pub mod compute;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatrixSize {
    width_a: u32,
    height_a: u32,
    width_b: u32,
}

#[derive(PartialEq)]
pub(crate) enum PipelineType {
    Mul,
    Relu,
}

#[cfg(test)]
mod tests {
    use crate::compute::GpuCompute;
    use pollster::block_on;
    use rand::{rng, Rng};

    // CPU implementation of matrix multiplication for verification
    fn cpu_matrix_multiply(a: &[f32], b: &[f32], size: usize) -> Vec<f32> {
        let mut result = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    result[i * size + j] += a[i * size + k] * b[k * size + j];
                }
            }
        }
        result
    }

    fn cpu_relu(input: &mut [f32], size: usize) {
        for e in input.iter_mut().take(size) {
            *e = e.max(0.0);
        }
    }

    // Helper function to generate a random 64x64 matrix
    fn generate_random_matrix(size: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..size * size)
            .map(|_| rng.random_range(-10.0..10.0)) // Random values between -10 and 10
            .collect()
    }

    fn generate_random_tensor(size: usize, rng: &mut impl Rng) -> Vec<f32> {
        (0..size).map(|_| rng.random_range(-10.0..10.0)).collect()
    }

    #[test]
    fn matrix_mul_1() {
        const SIZE: usize = 16;
        let mut gpu_compute = block_on(GpuCompute::new()).unwrap();
        let matrix_a = vec![1.0; SIZE * SIZE];
        let matrix_b = matrix_a.clone();
        let result = block_on(gpu_compute.matrix_mul(&matrix_a, &matrix_b, SIZE));
        assert_eq!(result, [SIZE as f32; SIZE * SIZE]);
    }

    #[test]
    fn identity_matrix() {
        const SIZE: usize = 64;
        let mut gpu_compute = block_on(GpuCompute::new()).unwrap();
        let mut matrix_a = Vec::new();
        let mut identity_matrix = vec![0.0; SIZE * SIZE];

        for (y, v) in identity_matrix.iter_mut().enumerate().take(SIZE * SIZE) {
            matrix_a.push(y as f32);
            if y % (SIZE + 1) == 0 {
                *v = 1.0;
            }
        }

        let expected_result = matrix_a.clone();
        let result = block_on(gpu_compute.matrix_mul(&matrix_a, &identity_matrix, SIZE));

        assert_eq!(result, expected_result);
    }

    #[test]
    fn identity_matrix_small() {
        const SIZE: usize = 4;
        let mut gpu_compute = block_on(GpuCompute::new()).unwrap();
        let mut matrix_a = Vec::new();
        let mut identity_matrix = vec![0.0; SIZE * SIZE];

        for (y, v) in identity_matrix.iter_mut().enumerate().take(SIZE * SIZE) {
            matrix_a.push(y as f32);
            if y % (SIZE + 1) == 0 {
                *v = 1.0;
            }
        }

        let expected_result = matrix_a.clone();
        let result = block_on(gpu_compute.matrix_mul(&matrix_a, &identity_matrix, SIZE));

        assert_eq!(result, expected_result);
    }

    #[test]
    // Tests randomly filled matrices of random sizes
    fn random_matrices() {
        let mut rng = rng();
        let size = rng.random_range(2..64) as usize;
        let matrix_a = generate_random_matrix(size, &mut rng);
        let matrix_b = generate_random_matrix(size, &mut rng);

        let mut gpu_compute = block_on(GpuCompute::new()).unwrap();

        let result = block_on(gpu_compute.matrix_mul(&matrix_a, &matrix_b, size));
        let cpu_result = cpu_matrix_multiply(&matrix_a, &matrix_b, size);

        assert_eq!(result, cpu_result);
    }

    #[test]
    fn test_relu_simple() {
        block_on(async {
            let mut gpu_compute = GpuCompute::new()
                .await
                .expect("Failed to obtain device handle on GPU.");

            let input_data = vec![-2.0, 1.0, 0.0, 1.0, 2.0, 5.0, 3.5];
            let expected_output = vec![0.0, 1.0, 0.0, 1.0, 2.0, 5.0, 3.5];

            let mut gpu_data = input_data.clone();
            gpu_compute.relu(&mut gpu_data);

            assert_eq!(gpu_data, expected_output);
        })
    }

    #[test]
    fn test_relu_iter() {
        for _ in 0..20 {
            block_on(async {
                let mut rng = rng();
                let mut gpu_compute = GpuCompute::new()
                    .await
                    .expect("Failed to obtain device handle on GPU.");

                let size = rng.random_range(2..64);
                let mut random_input = generate_random_tensor(size, &mut rng);
                cpu_relu(&mut random_input, size);
                let expected_output = random_input.clone();

                gpu_compute.relu(&mut random_input);

                assert_eq!(random_input, expected_output);
            })
        }
    }

    mod bench {
        extern crate test;

        use pollster::block_on;
        use rand::rng;
        use std::hint::black_box;
        use test::Bencher;

        use crate::{
            compute::GpuCompute,
            tests::{cpu_matrix_multiply, cpu_relu, generate_random_tensor},
        };

        #[bench]
        fn bench_cpu_mul(b: &mut Bencher) {
            const SIZE: usize = 64;
            let mut matrix_a = Vec::with_capacity(SIZE * SIZE);
            let mut identity_matrix = vec![0.0; SIZE * SIZE];

            for (y, v) in identity_matrix.iter_mut().enumerate().take(SIZE * SIZE) {
                matrix_a.push(y as f32);
                if y % (SIZE + 1) == 0 {
                    *v = 1.0;
                }
            }
            b.iter(|| {
                let _result = black_box(cpu_matrix_multiply(
                    black_box(&matrix_a),
                    black_box(&identity_matrix),
                    black_box(SIZE),
                ));
            });
        }

        #[bench]
        fn bench_gpu_mul(b: &mut Bencher) {
            const SIZE: usize = 64;
            let mut matrix_a = Vec::with_capacity(SIZE * SIZE);
            let mut identity_matrix = vec![0.0; SIZE * SIZE];

            for (y, v) in identity_matrix.iter_mut().enumerate().take(SIZE * SIZE) {
                matrix_a.push(y as f32);
                if y % (SIZE + 1) == 0 {
                    *v = 1.0;
                }
            }

            let mut gpu_compute = block_on(GpuCompute::new()).unwrap();

            b.iter(|| {
                let _result = block_on(gpu_compute.matrix_mul(&matrix_a, &identity_matrix, SIZE));
            });
        }

        #[bench]
        fn bench_cpu_relu(b: &mut Bencher) {
            const SIZE: usize = 1024;
            let mut rng = rng();

            let mut random_input = generate_random_tensor(SIZE, &mut rng);

            b.iter(|| {
                cpu_relu(black_box(&mut random_input), black_box(SIZE));
            });
        }

        #[bench]
        fn bench_gpu_relu(b: &mut Bencher) {
            const SIZE: usize = 1024;
            let mut rng = rng();

            let mut random_input = generate_random_tensor(SIZE, &mut rng);
            let mut gpu_compute = block_on(GpuCompute::new()).unwrap();

            for _ in 0..5 {
                let mut warmup_input = generate_random_tensor(SIZE, &mut rng);
                gpu_compute.relu(&mut warmup_input);
            }

            b.iter(|| gpu_compute.relu(&mut random_input));
        }
    }
}
