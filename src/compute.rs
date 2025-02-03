use anyhow::Context;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::MatrixSize;

/// Creates a simple handle on a GPU compute shader that can do matrix multiplication.
pub struct GpuCompute {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
}

impl GpuCompute {
    pub async fn new() -> anyhow::Result<Self> {
        // Create a wgpu instance
        let instance = wgpu::Instance::default();

        // Request a handle to the physical graphics card
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .context("Tried to request an adapter handle to the GPU.")?;

        // Request a connection to the physical device
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .context("Tried to request device from default GPU instance adapter.")?;

        // Include the .wgsl compute shader and parse it into a module on the GPU.
        let wgsl_shader = include_str!("shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_shader.into()),
        });

        // Create a bind group layout with two read-only input buffers and one read/write output result buffer.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create a pipeline containing the single shader stage.
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline,
        })
    }

    pub async fn run(&self, a: &[f32], b: &[f32], size: usize) -> Vec<f32> {
        let matrix_size = MatrixSize {
            width_a: size as u32,
            height_a: size as u32,
            width_b: size as u32,
        };

        // Create matrix buffers
        let buf_size =
            (matrix_size.height_a * matrix_size.width_b * std::mem::size_of::<f32>() as u32) as u64;

        let matrix_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix A"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let matrix_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix B"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let matrix_size_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix Size Buffer"),
                contents: bytemuck::cast_slice(&[matrix_size]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create a bind group using the devices bind group layout.
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: matrix_size_buf.as_entire_binding(),
                },
            ],
        });

        // Create a command encoder to encode operations for the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        {
            // Start a compute pass
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((size as u32 + 15) / 16, (size as u32 + 15) / 16, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Read back the result
        let result_staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Staging Buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Result Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&result_buf, 0, &result_staging_buf, 0, buf_size);
        self.queue.submit(Some(encoder.finish()));

        let buf_slice = result_staging_buf.slice(..);
        // Get a handle on a new channel's sender and receiver
        let (tx, rx) = std::sync::mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        // Wait for work to complete
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        // Extract data
        let result = buf_slice.get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice(&result).to_vec();
        drop(result);
        result_staging_buf.unmap();

        output
    }
}
