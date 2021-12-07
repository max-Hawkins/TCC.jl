
module TCC

using CUDA
using Test

export TCC_CONFIG

const NR_WARPS = 4

"""
Tensor-core-correlator configuration data
"""
struct TCC_CONFIG
    ptx_file::String

    nr_bits::Int
    nr_channels::Int
    nr_polarizations::Int
    nr_samples_per_channel::Int
    nr_receivers::Int
    nr_receivers_per_block::Int
    nr_baselines::Int
    nr_times_per_block::Int
    nr_blocks_per_dim::Int
    nr_thread_blocks_per_channel::Int

    sample_type::Type
    visibility_type::Type

    correlate_f::CuFunction
    launch_blocks::Tuple{Int64, Int64, Int64}
    launch_threads::Tuple{Int64, Int64, Int64}
end

"""
  correlate(samples_d::CuArray, viz_d::CuArray, config::TCC_CONFIG)

Correlate the given samples using the TCC kernel specified by configuration TCC_CONFIG
"""
function correlate(samples_d::CuArray, viz_d::CuArray, config::TCC_CONFIG)
  CUDA.cudacall(config.correlate_f, 
                (CuPtr{config.visibility_type}, CuPtr{config.sample_type}),
                viz_d, samples_d;
                blocks=config.launch_blocks, threads=config.launch_threads, shmem=1)
end

"""
  TCC_CONFIG(ptx_file, nr_bits, nr_channels, nr_polarizations, nr_samples_per_channel, nr_receivers, nr_receivers_per_block)

Create a tensore-core-correlator configuration struct from the necessary parameters.
"""
function TCC_CONFIG(ptx_file, nr_bits, nr_channels, nr_polarizations, nr_samples_per_channel, nr_receivers, nr_receivers_per_block)
  # Check that PTX file exists
  if !isfile(ptx_file)
    error("PTX file: $ptx_file not found!")
  end

  # Calculate all non-given options
  nr_baselines = Int((nr_receivers) * ((nr_receivers) + 1) / 2)
  nr_times_per_block = Int(128 / (nr_bits))
  nr_blocks_per_dim = floor(Int, (nr_receivers + nr_receivers_per_block - 1) / nr_receivers_per_block);
  nr_thread_blocks_per_channel = floor(Int, nr_receivers_per_block == 64 ? nr_blocks_per_dim * nr_blocks_per_dim : nr_blocks_per_dim * (nr_blocks_per_dim + 1) / 2);

  # Create CUDA Module and find correlate function from precompiled GPU code (.ptx file)
  cu_mod = CUDA.CuModuleFile(ptx_file)
  correlate_f = CUDA.CuFunction(cu_mod, "correlate")

  # Kernel launch parameters
  launch_blocks = (nr_thread_blocks_per_channel, nr_channels, 1)   
  launch_threads = (32, 2, 2)
  println("nr thread blocks per chan:  $nr_thread_blocks_per_channel")
  println("Launch Blocks: $launch_blocks Launch threads: $launch_threads")

  if nr_bits == 4
    error("4-bit samples currently unsupported by Julia-TCC")
  elseif nr_bits == 8
    sample_type     = Complex{Int8}
    visibility_type = Complex{Int32}
  elseif nr_bits == 16
    sample_type      = Complex{Float16}
    visibilitiy_type = Complex{Float32}
  else
    error("Invalid TCC_Config nr_bits of $nr_bits")
  end

  return TCC_CONFIG(ptx_file, nr_bits, nr_channels, nr_polarizations, nr_samples_per_channel, nr_receivers, nr_receivers_per_block,
                    nr_baselines, nr_times_per_block, nr_blocks_per_dim, nr_thread_blocks_per_channel,
                    sample_type, visibility_type, correlate_f, launch_blocks, launch_threads)
end

"""
  create_simple_test_config(ptx_file)

Create a TCC_CONFIG struct for the TCC SimpleTest verification
"""
function create_simple_test_config(ptx_file="/home/mhawkins/tensor-core-correlator/simple_test.ptx")
  nr_bits = 8
  nr_channels = 480
  nr_polarizations = 2
  nr_samples_per_channel = 3072
  nr_receivers = 576
  nr_receivers_per_block = 32
  return TCC_CONFIG(ptx_file, nr_bits, nr_channels, nr_polarizations, nr_samples_per_channel, nr_receivers, nr_receivers_per_block)
end

"""
  create_meerkat_config(ptx_file)

Create a TCC_CONFIG struct for MeerKAT test operation
"""
function create_meerkat_config(ptx_file)
  nr_bits = 8
  nr_channels = 64
  nr_polarizations = 2
  nr_samples_per_channel = 8192
  nr_receivers = 58
  nr_receivers_per_block = 64
  return TCC_CONFIG(ptx_file, nr_bits, nr_channels, nr_polarizations, nr_samples_per_channel, nr_receivers, nr_receivers_per_block)
end

"""
  test(config::TCC_CONFIG)

Run tests to verify the TCC correlate kernel correctly calculates the visibilities
"""
function test(config::TCC_CONFIG)
  println("Running tests on GPU...")
  ### TEST: All zero input => all zero outputs (every visibility is written to)

  # Make equivalent input arrays in col-major ordering in Julia (flipped dims)
  samples = zeros(config.sample_type, (config.nr_times_per_block, config.nr_polarizations, config.nr_receivers, Int(config.nr_samples_per_channel / config.nr_times_per_block), config.nr_channels));
  visibilities = ones(config.visibility_type, (config.nr_polarizations, config.nr_polarizations, config.nr_baselines, config.nr_channels));

 # Create GPU arrays and copy data to them
  samples_d = CuArray(samples);
  visibilities_d = CuArray(visibilities);
  # Run correlation kernel
  correlate(samples_d, visibilities_d, config)
  # Transfer calculated visibilities back to CPU
  visibilities = Array(visibilities_d);

  println("Zero input => zero output test result: \n$(@test all(visibilities .== Complex{Int32}(0,0)))")
  
  ### TEST: Two non-zero input channels for a single cross-polarization => correct output visibilities

  # Set test samples to random values
  samples[:,1,1,:,1] .= rand(config.sample_type, size(samples[:,1,1,:,1]))
  samples[:,1,2,:,1] .= rand(config.sample_type, size(samples[:,1,2,:,1]))
  # Eliminate typemin values in the imaginary component to avoid conjugation errors
  samples[findall(x->x.im==typemin(Int8), samples)] .= config.sample_type(0+0im);
  samples_d = CuArray(samples);

  correlate(samples_d, visibilities_d, config)
  visibilities = Array(visibilities_d);
  # Test for correct test visibility
  expected = sum(config.visibility_type.(samples[:,1,2,:,1]) .* conj.(config.visibility_type.(samples[:,1,1,:,1])))
  println("Two non-zero input channels => correct output visibilitiy test result: \n$(@test visibilities[1, 1, 2, 1] == expected) ")

  # TEST: Random input samples => Calculate visibilties and return for downstream tests

  # Create random sample data
  # base_type = config.sample_type.parameters[1] # Complex{base_type} => basetype
  # rng = MersenneTwister(42)

  # rf_r = floor.(randn(rng, Float32, size(samples)) .* 7)
  # rf_i = floor.(randn(rng, Float32, size(samples)) .* 7)

  # # Zero undesirable data (exceeding bounds or typemin values)
  # rf_r[findall(x->(x<=typemin(Int8)) || (x>typemax(Int8)), rf_r)] .= 0
  # rf_i[findall(x->(x<=typemin(Int8)) || (x>typemax(Int8)), rf_i)] .= 0

  # samples = base_type.(rf_r) + base_type.(rf_i)im

  samples = rand(config.sample_type, size(samples));
  samples[findall(x->x.im==typemin(Int8), samples)] .= config.sample_type(0+0im);

  visibilities .= config.visibility_type(0,0);
  samples_d = CuArray(samples);
  visibilities_d = CuArray(visibilities);

  correlate(samples_d, visibilities_d, config);
  visibilities = Array(visibilities_d);

  return verify(samples, visibilities, config)
end

"""
  verify(samples, viz_truth)

Verify that the visibility data is correct given the sample data by correlating on the CPU

Emulates CorrelatorTest.cc lines 171 - 223
"""
function verify(samples, viz_truth, config)
  println("Running random sample data correlation on CPU to verify. This may take awhile...")

  visibilities = zeros(config.visibility_type, (config.nr_polarizations, config.nr_polarizations, config.nr_baselines, config.nr_channels))
  sum = zeros(config.visibility_type, (config.nr_polarizations, config.nr_polarizations, config.nr_baselines))
  ref = zeros(config.sample_type, size(samples[:,:,:,1,1]))
  sample0 = config.sample_type(0,0)
  sample1 = config.sample_type(0,0)

  nr_errors = 0
  
  # Iterate through channels
  for chan in 1:1#config.nr_channels
    println("Channel: $chan / $(config.nr_channels)")
    sum .= config.visibility_type(0,0)

    for major_time in 1:Int(config.nr_samples_per_channel / config.nr_times_per_block)
      ref = samples[:, :, :, major_time, chan]
      
      baseline = 0
      for recv1 in 1:config.nr_receivers
        for recv0 in 1:config.nr_receivers
          if recv0 > recv1
            break
          end

          baseline += 1
          for minor_time in 1:config.nr_times_per_block
            for pol0 in 1:config.nr_polarizations
              for pol1 in 1:config.nr_polarizations
                sample0 = ref[minor_time, pol0, recv0]
                sample1 = ref[minor_time, pol1, recv1]

                #sum[pol0, pol1, baseline] += config.visibility_type(ref[minor_time, pol1, recv1]) * conj(config.visibility_type(ref[minor_time, pol0, recv0]))

                sum[pol0, pol1, baseline] += config.visibility_type(sample1.re, sample1.im) * conj(config.visibility_type(sample0.re, sample0.im))
                # println("samp0: $sample0 samp1: $sample1  \n out: $(config.visibility_type(sample1.re, sample1.im) * conj(config.visibility_type(sample0.re, sample0.im)))")
                # @inbounds sum[pol0, pol1, baseline] += config.visibility_type(ref[minor_time, pol1, recv1].re, ref[minor_time, pol1, recv1].im) * sample0

              end
            end
          end # End minor_time
        end
      end
    end # End major_time

    println("Mismatched locations: $(findall(sum .!= viz_truth[:,:,:,chan]))")
    if viz_truth[:,:,:,chan] != sum
      println("Visibilities do not match!")
      return viz_truth[:,:,:,chan], sum
    end 
  end # end chan

  return true
end

#  GuppiRaw format for data from Blio is:
#   [Polarizations, Times, Channels, Antennas]

#   TCC wants:
#   [nr_times_per_block, config.nr_polarizations, config.nr_receivers, Int(config.nr_samples_per_channel / config.nr_times_per_block), config.nr_channels]
#   or in more concise language:
#   [Times, Polarizations, Antennas, Int(config.nr_samples_per_channel / config.nr_times_per_block), Channels]
#   It adds an integration dimension 
function raw_to_tcc(data, nr_times_per_block)
 
  size_i = size(data)
  return permutedims(reshape(data, (size_i[1], nr_times_per_block, Int(size_i[2] / nr_times_per_block), size_i[3], size_i[4])), [2, 1, 5, 3, 4])
end

end # Module TCC

### Helpful Snippets ###

# From CorrelatorKernel.cc
#     void CorrelatorKernel::launchAsync(cu::Stream &stream, cu::DeviceMemory &deviceVisibilities, cu::DeviceMemory &deviceSamples)
#     {
#       std::vector<const void *> parameters = { deviceVisibilities.parameter(), deviceSamples.parameter() };
#       stream.launchKernel(function,
#     		      nrThreadBlocksPerChannel, nrChannels, 1,
#     		      32, 2, 2,
#     		      0, parameters);
#     }

# From cu.h:
#     void launchKernel(Function &function, unsigned gridX, unsigned gridY, unsigned gridZ, unsigned blockX, unsigned blockY, unsigned blockZ, unsigned sharedMemBytes, const std::vector<const void *> &parameters)
#     {
#         checkCudaCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, _obj, const_cast<void **>(&parameters[0]), 0));
#     }
# ==> sharedMemBytes = 1
# ==> gridX = NR_THREAD_BLOCKS_PER_CHANNEL
# ==> gridY = NR_CHANNELS
# ==> gridZ = 1


# From CUDAdrv/execution
# "    launch(f::CuFunction; args...; blocks::CuDim=1, threads::CuDim=1,
# cooperative=false, shmem=0, stream=stream())

# Low-level call to launch a CUDA function `f` on the GPU, using `blocks` and `threads` as
# respectively the grid and block configuration. Dynamic shared memory is allocated according
# to `shmem`, and the kernel is launched on stream `stream`.

# Arguments to a kernel should either be bitstype, in which case they will be copied to the
# internal kernel parameter buffer, or a pointer to device memory.
# This is a low-level call, prefer to use [`cudacall`](@ref) instead."
