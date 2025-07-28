import torch
import sys

print("🍎 macOS MPS Availability Check")
print("=" * 40)
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

if torch.backends.mps.is_available():
    print("✅ MPS backend is available!")
    
    # MPS 장치 생성 테스트
    try:
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        print("✅ MPS device test passed!")
        print(f"   Device: {device}")
        print(f"   Tensor shape: {z.shape}")
        print(f"   Memory allocated: {torch.mps.current_allocated_memory()/1024/1024:.2f} MB")
    except Exception as e:
        print(f"❌ MPS device test failed: {e}")
        
else:
    print("❌ MPS backend is not available")
    print("   Fallback to CPU will be used")

# 추천 설정 출력
print("\n🔧 Recommended settings for iTransformer:")
print("   - Use smaller batch sizes (16-32) for MPS")
print("   - Monitor memory usage with torch.mps.current_allocated_memory()")
print("   - Set device = torch.device('mps') in your code")
