import subprocess

def get_free_gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
    return int(output.decode('utf-8').split('\\')[0])

def main():
    required_memory = 2048  # 指定程序所需的显存大小（MB）
    free_memory = get_free_gpu_memory()
    
    if free_memory >= required_memory:
        print("GPU显存足够，运行程序...")
        # 在这里运行你的程序的命令
    else:
        print("GPU显存不足，无法运行程序。")
        print(free_memory)

if __name__ == "__main__":
    main()
