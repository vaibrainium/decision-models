def get_gpu_info():       
    ''' Gathers GPU information using Numba package
    Returns:
    num_sm: Number of total Streaming Multiprocessors on GPU
    num_cores_per_sm: Number of total SMs on GPU
    '''
    from numba import cuda
    cc_cores_per_SM_dict = {
        (2,0) : 32,  (2,1) : 48,
        (3,0) : 192, (3,5) : 192, (3,7) : 192,
        (5,0) : 128, (5,2) : 128,
        (6,0) : 64,  (6,1) : 128,
        (7,0) : 64,  (7,5) : 64, 
        (8,0) : 64,  (8,6) : 128
        }

    device = cuda.get_current_device()
    num_sm = getattr(device, 'MULTIPROCESSOR_COUNT')
    my_cc = device.compute_capability
    num_cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    total_cores = num_cores_per_sm*num_sm
    print("GPU compute capability: " , my_cc)
    print("GPU total number of Streaming Multiprocessors (SM): " , num_sm)
    print("GPU total number of cores per SMs: " , num_cores_per_sm)
    print("total cores: " , total_cores)
    print('''\n Deciding which execution configuration to use is not easy, and the choice should be driven by performance analysis. However, here are some basic rules to get started:
    - The number of blocks in the grid should be larger than the number of Streaming Multiprocessors on the GPU, typically 2 to 4 times larger.
    - The number of threads per block should be a multiple of 32, typically between 128 and 512. ''')   
    
    return num_sm, num_cores_per_sm



def get_device():
    if numba.cuda.is_available():
        device = "GPU"
        num_sm, num_cores_per_sm = get_gpu_info()
        blockdim = 2*32*(num_cores_per_sm//32)   # num_blocks should be 2x - 4x times number of threads and should be multiplier of 32
        return [device, blockdim]
    else:
        device = "CPU"
        return [device]