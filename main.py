import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# 최종 가중치 파일의 경로
weights_path = '/content/person_data/weights/final_weights.pt'

# Jetson 보드에서는 GPU를 사용할 수 있도록 device를 설정합니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO 모델 초기화
model = YOLO(weights_path, device=device)

# 테스트 이미지의 경로
test_image_path = '/content/person_data/test/images/example.jpg'

# 이미지 불러오기 및 전처리
image = Image.open(test_image_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
input_image = transform(image).unsqueeze(0).to(device)

# 예측 수행
results = model(input_image)

# 결과 출력
results.show()
