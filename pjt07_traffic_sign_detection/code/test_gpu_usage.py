import torch

print("🧠 PyTorch 버전:", torch.__version__)
print("🧪 PyTorch 빌드 시 사용한 CUDA 버전:", torch.version.cuda)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print("✅ CUDA 사용 가능 (GPU 사용 중)")
    print("🔧 GPU 이름:", props.name)
    print("🔢 Compute Capability:", props.major, ".", props.minor)
    print("🧮 총 메모리:", f"{props.total_memory / 1024**2:.2f} MB")

    print("📦 현재 할당된 메모리:", f"{torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    print("📦 현재 예약된 메모리:", f"{torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    print("📦 최대 할당 메모리 (이 세션 기준):", f"{torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    print("\n📋 메모리 상태 요약:")
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

else:
    print("❌ CUDA 사용 불가. 현재 GPU를 사용할 수 없습니다.")
