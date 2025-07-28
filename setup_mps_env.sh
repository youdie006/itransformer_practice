#!/bin/bash

# macOS MPS를 위한 iTransformer 환경 설정 스크립트
# PyTorch MPS 백엔드를 사용하여 Apple Silicon GPU 가속 활용

echo "🍎 Setting up iTransformer environment for macOS MPS..."

# conda 환경 생성 (Python 3.9 권장 - MPS 안정성)
echo "📦 Creating conda environment: itransformer_mps"
conda create -n itransformer_mps python=3.9 -y

# 환경 활성화
echo "🔧 Activating environment..."
conda activate itransformer_mps

# macOS MPS 지원 PyTorch 설치 (nightly 또는 2.0+ 버전)
echo "⚡ Installing PyTorch with MPS support..."
conda install pytorch torchvision torchaudio -c pytorch -y

# 기본 패키지 설치
echo "📚 Installing basic packages..."
conda install pandas scikit-learn numpy matplotlib -c conda-forge -y

# 추가 필수 패키지
echo "🔧 Installing additional packages..."
pip install reformer-pytorch==1.4.4
pip install einops  # reformer-pytorch 의존성
pip install seaborn  # 시각화
pip install tqdm     # 진행률 표시

# MPS 가용성 확인 스크립트 생성
echo "✅ Creating MPS availability check script..."
cat > check_mps.py << 'EOF'
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
EOF

echo "🎯 Running MPS availability check..."
python check_mps.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate itransformer_mps"
echo "  cd /Users/manon/CursorProject/iTransformer_practice"
echo "  python iTransformer_Study_Guide.py"
echo ""
echo "💡 Tips for MPS usage:"
echo "  - MPS is optimized for Apple Silicon (M1/M2/M3)"
echo "  - Use smaller batch sizes compared to CUDA"
echo "  - Monitor GPU memory with Activity Monitor"
echo "  - Some operations may fall back to CPU automatically"