import torch


def check_gpu_operations():
    # Check number of GPUs available
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    # Iterate over each GPU
    for gpu_id in range(n_gpus):
        # Set device to current GPU
        device = torch.device(f"cuda:{gpu_id}")
        print(f"\nTesting operations on GPU {gpu_id}")

        try:
            # Create a tensor on the GPU
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            print("Original tensor:", x)

            # Perform a simple calculation
            y = x * x
            print("Tensor after squaring:", y)

            # Transfer tensor back to CPU and print
            y_cpu = y.to("cpu")
            print("Tensor on CPU after operation:", y_cpu)

            print(f"Operations on GPU {gpu_id} succeeded.")

        except Exception as e:
            print(f"An error occurred with GPU {gpu_id}: {e}")


# Run the GPU check
check_gpu_operations()
