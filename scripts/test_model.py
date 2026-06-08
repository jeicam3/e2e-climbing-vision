import torch
import cv2
import torch.nn as nn
from torchvision import transforms, models

MODEL_PATH = 'checkpoints/yolo_climbing_model.pth'
VIDEO_PATH = 'data/test_video.mp4'
CLASS_NAMES = ['LH', 'RH', 'LF', 'RF']

#visualization settings
bar_width = 150
bar_height = 20
start_x = 20
start_y = 300

def load_trained_model(path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(num_ftrs, 4),
        nn.Sigmoid()
    )
    
    #wczytanie wag modelu
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(frame, model):
    frame_t = preprocess(frame).unsqueeze(0)
    
    with torch.no_grad():
        output = model(frame_t)
        probs = output.squeeze().tolist()
        
    return probs

def visualize(frame, probs):
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        cv2.rectangle(frame, (start_x, start_y + i*30), 
                      (start_x + bar_width, start_y + i*30 + bar_height), (50, 50, 50), -1) #szare tło
        
        color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
        current_width = int(bar_width * prob)
        cv2.rectangle(frame, (start_x, start_y + i*30), 
                      (start_x + current_width, start_y + i*30 + bar_height), color, -1) # >50 - zielony, <50 - czerwony
        
        #etykieta
        text = f"{name}: {int(prob*100)}%"
        cv2.putText(frame, text, (start_x + bar_width + 10, start_y + i*30 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = load_trained_model(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    probs = predict(frame, model)

    visualize(frame, probs)

    cv2.imshow('Climbing Vision AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #print(probs)
             
cap.release()