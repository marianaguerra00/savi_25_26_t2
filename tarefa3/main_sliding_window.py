import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from model import ModelBetterCNN

# -------------------------------------------------
# 1. LEITURA DO BINÁRIO
# -------------------------------------------------
def loadImagesUbyte(filePath, numberOfImages, imageSize):
    try:
        data = np.fromfile(filePath, dtype=np.uint8)
        return data.reshape(numberOfImages, imageSize, imageSize)
    except FileNotFoundError:
        print(f"ERROR: Couldn't find file {filePath}")
        exit()

# -------------------------------------------------
# 2. CONFIGURAÇÕES
# -------------------------------------------------
WINDOW_SIZE = 28        # Tamanho da janela (MNIST nativo)
STRIDE = 4              # Passo da janela (menor = mais lento, mais preciso)
CONFIDENCE_THRESHOLD = 0.90 # O modelo tem de ter 98% de certeza absoluta
NMS_THRESHOLD = 0.3     # Quão agressivo é a eliminar caixas sobrepostas

# Transformação igual à Tarefa 1
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -------------------------------------------------
# 3. CORE LOGIC
# -------------------------------------------------

def get_sliding_windows(image, window_size, stride):
    """
    Gerador que devolve recortes da imagem.
    """
    h, w = image.shape
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            crop = image[y:y + window_size, x:x + window_size]
            yield (x, y, crop)

def predict_crop(model, crop, device):
    """
    Recebe um crop numpy (28x28 uint8), processa e devolve (prob, label).
    """
    # Converter para PIL ou diretamente aplicar transformações
    # O ToTensor espera valores [0, 255] se for uint8, e converte para [0.0, 1.0]
    
    # Prepara o tensor [1, 1, 28, 28]
    crop_tensor = transform_pipeline(crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(crop_tensor)
        # Aplicar Softmax para ter percentagens reais
        probabilities = F.softmax(output, dim=1)
        
        # Obter o valor máximo e o índice
        max_prob, predicted_class = torch.max(probabilities, 1)
        
    return max_prob.item(), predicted_class.item()

def apply_nms(boxes, scores, score_threshold, nms_threshold):
    """
    Aplica Non-Maximum Suppression usando OpenCV.
    """
    if len(boxes) == 0:
        return [], [], []

    # cv2.dnn.NMSBoxes espera listas ou numpy arrays
    # boxes deve ser lista de [x, y, w, h]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
            # Nota: NMS não gere labels, temos de manter a correspondência nós
            # Mas aqui simplificamos assumindo que passamos labels numa lista externa
            # (Vou corrigir isto no main loop para ser mais robusto)
            pass 
            
    return indices # Retornamos os índices para filtrar as listas originais

# -------------------------------------------------
# 4. MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sliding Window Detection")
    parser.add_argument("--version", required=True, help="Ex: versionD")
    parser.add_argument("--model_path", required=True, help="Caminho para o .pth da Tarefa 1")
    parser.add_argument("--image_index", type=int, default=0, help="Qual imagem do binário testar")
    args = parser.parse_args()

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A usar dispositivo: {device}")

    # Carregar Modelo
    model = ModelBetterCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Modelo carregado.")

    # Carregar Dados
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(base_dir, "output", args.version, "test-images-ubyte.bin")
    
    all_images = loadImagesUbyte(bin_path, 10000, 128) 
    
    target_image = all_images[args.image_index]
    
    # --- SLIDING WINDOW ---
    print(f"A processar imagem {args.image_index} com Sliding Window...")
    
    detected_boxes = [] # [x, y, w, h]
    detected_scores = []
    detected_labels = []

    # Loop principal
    # for (x, y, crop) in get_sliding_windows(target_image, WINDOW_SIZE, STRIDE):
    #     if np.mean(crop) < 100:  # 10 em 255 (ajusta se necessário, ex: 20 ou 30)
    #         continue # Ignorar janelas quase vazias
        
    #     prob, label = predict_crop(model, crop, device)
        
    #     # Filtro 1: Threshold de Confiança
    #     if prob >= CONFIDENCE_THRESHOLD:
    #         detected_boxes.append([x, y, WINDOW_SIZE, WINDOW_SIZE])
    #         detected_scores.append(prob)
    #         detected_labels.append(label)

    # print(f"Janelas candidatas (antes do NMS): {len(detected_boxes)}")

    # ... (dentro do main, logo após carregar a imagem) ...
    
    print(f"--- MODO DIAGNÓSTICO (Imagem {args.image_index}) ---")
    
    # Loop de Janela Deslizante
    for (x, y, crop) in get_sliding_windows(target_image, WINDOW_SIZE, STRIDE):
        
        # 1. TESTE DO FILTRO DE INTENSIDADE
        max_val = np.max(crop)
        if max_val < 50: # Filtro muito permissivo (quase tudo passa)
            continue
            
        # 2. INFERÊNCIA
        prob, label = predict_crop(model, crop, device)
        
        # 3. PRINT DE DEBUG (Obrigatório ver isto no terminal)
        # Se a rede tiver mais de 50% de certeza, imprime.
        if prob > 0.5:
            print(f"DEBUG: Pos: ({x},{y}) | MaxPixel: {max_val} | Rede vê: {label} com {prob:.4f} confiança")
        
        # Guardar deteção se passar no limiar oficial
        if prob >= CONFIDENCE_THRESHOLD:
            detected_boxes.append([x, y, WINDOW_SIZE, WINDOW_SIZE])
            detected_scores.append(prob)
            detected_labels.append(label)

    # --- NMS (Limpeza) ---
    if len(detected_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(detected_boxes, detected_scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        final_boxes = []
        final_labels = []
        final_scores = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(detected_boxes[i])
                final_labels.append(detected_labels[i])
                final_scores.append(detected_scores[i])
    else:
        final_boxes = []

    print(f"Deteções finais: {len(final_boxes)}")

    # --- VISUALIZAÇÃO ---
    # Converter para RGB para desenhar retângulos coloridos
    display_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)

    for i, (x, y, w, h) in enumerate(final_boxes):
        label = final_labels[i]
        score = final_scores[i]
        
        # Desenhar retângulo
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Escrever Label
        text = f"{label} ({score:.2f})"
        cv2.putText(display_image, text, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(display_image)
    plt.title(f"Sliding Window Result (Img {args.image_index})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()