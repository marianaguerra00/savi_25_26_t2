import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider, Button
import glob 


# -------------------------------------------------
# LOAD YOLO DATA
# -------------------------------------------------

def load_yolo_data(folder_path, split_prefix):
    """
    Lê os caminhos das imagens filtrando pelo split (train ou test).
    """
    # 1. Encontrar ficheiros com o prefixo correto (ex: "train_*.jpg")
    search_pattern_img = os.path.join(folder_path, f"{split_prefix}*.jpg")
    search_pattern_txt = os.path.join(folder_path, f"{split_prefix}*.txt")
    
    img_files = sorted(glob.glob(search_pattern_img))
    txt_files = sorted(glob.glob(search_pattern_txt))
    
    if len(img_files) == 0:
        print(f"ERRO: Nenhuma imagem encontrada para o padrão: {search_pattern_img}")
        return [], []

    if len(img_files) != len(txt_files):
        print(f"AVISO: {len(img_files)} imagens vs {len(txt_files)} labels. Verificar integridade.")

    all_annotations = []
    
    # 2. Ler Labels para estatísticas
    for txt_file in txt_files:
        image_annots = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    # O visualizador precisa de pixeis absolutos para desenhar o rectangulo
                    # Mas o ficheiro tem normalizados. Vamos converter de volta SÓ para visualizar/stats.
                    # Assumimos 128x128 (podes ler da imagem se quiseres ser rigorosa, mas é lento)
                    IMG_SIZE = 128 
                    
                    x_c, y_c, w_n, h_n = map(float, parts[1:])
                    
                    w = w_n * IMG_SIZE
                    h = h_n * IMG_SIZE
                    x = (x_c * IMG_SIZE) - (w / 2)
                    y = (y_c * IMG_SIZE) - (h / 2)
                    
                    image_annots.append((cls_id, x, y, w, h))
        all_annotations.append(image_annots)
        
    return img_files, all_annotations
# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
def runInterface(images, annotations, digitCounts, classCounts, boxSizes):

    currentImageIndex = [0]
    currentStatIndex = [0]

    fig = plt.figure(figsize=(12, 7))

    # --------------------
    # LEFT SIDE (CONTENT)
    # --------------------

    # IMAGE (TOP LEFT)
    axImage = plt.axes([0.10, 0.55, 0.45, 0.4])
    axImage.axis("off")

    # STATS (BOTTOM LEFT)
    axStats = plt.axes([0.10, 0.10, 0.45, 0.35])

    # ---------------------------
    # RIGHT SIDE (BUTTONS)
    # ---------------------------

    buttonWidth = 0.20
    buttonHeight = 0.08
    buttonX = 0.60

    axNextImg = plt.axes([buttonX, 0.75, buttonWidth, buttonHeight])
    axPrevImg = plt.axes([buttonX, 0.65, buttonWidth, buttonHeight])
    
    axNextStat = plt.axes([buttonX, 0.30, buttonWidth, buttonHeight])
    axPrevStat = plt.axes([buttonX, 0.20, buttonWidth, buttonHeight])
    
    # ------------------
    # DRAW FUNCTIONS
    # ------------------

    def drawImage():
        axImage.clear()
        axImage.axis("off")
        idx = currentImageIndex[0]
        axImage.set_title(f"Image {idx}: {os.path.basename(images[idx])}", fontsize=13) # Mostra nome do ficheiro

        # CARREGA A IMAGEM AGORA (LAZY & UNICODE SAFE)
        # cv2.imread falha com 'º', usamos np.fromfile + cv2.imdecode
        img_path = images[idx]
        try:
            file_bytes = np.fromfile(img_path, dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Erro ao ler imagem {img_path}: {e}")
            return
        
        axImage.imshow(img_array, cmap="gray")

        for digitLabel, x, y, w, h in annotations[currentImageIndex[0]]:
            rect = plt.Rectangle(
                (x, y), w, h,
                edgecolor="red",
                facecolor="none",
                linewidth=2
            )
            axImage.add_patch(rect)

    def drawStats():
        axStats.clear()

        if currentStatIndex[0] == 0:
            uniqueCounts, counts = np.unique(digitCounts, return_counts=True)
            rel = counts / counts.sum()

            axStats.bar(uniqueCounts, rel, edgecolor="black", alpha=0.8)
            axStats.set_title("Number of digits per image")
            axStats.set_xlabel("Digits per image")
            axStats.set_ylabel("Relative frequency")
            axStats.set_xticks(uniqueCounts)

        elif currentStatIndex[0] == 1:
            axStats.bar(range(10), classCounts, edgecolor="black", alpha=0.8)
            axStats.set_title("Digit class distribution")
            axStats.set_xlabel("Digit class")
            axStats.set_ylabel("Number of occurrences")
            axStats.set_xticks(range(10))

        elif currentStatIndex[0] == 2:
            counts, bins = np.histogram(
                boxSizes,
                bins=range(min(boxSizes), max(boxSizes) + 2)
            )
            rel = counts / counts.sum()
            axStats.bar(bins[:-1], rel, edgecolor="black", alpha=0.8)
            axStats.set_title("Bounding box size distribution")
            axStats.set_xlabel("Box size (pixels)")
            axStats.set_ylabel("Relative frequency")

        axStats.grid(True, linestyle="--", alpha=0.5)

    # ------------------------
    # INITIAL DRAW
    # ------------------------
    drawImage()
    drawStats()

    # ------------------------
    # BUTTONS
    # ------------------------

    btnPrevImg = Button(axPrevImg, "Previous image")
    btnNextImg = Button(axNextImg, "Next image")

    btnPrevStat = Button(axPrevStat, "Previous stats")
    btnNextStat = Button(axNextStat, "Next stats")

    def onPrevImage(event):
        currentImageIndex[0] = max(0, currentImageIndex[0] - 1)
        drawImage()
        fig.canvas.draw_idle()

    def onNextImage(event):
        currentImageIndex[0] = min(len(images) - 1, currentImageIndex[0] + 1)
        drawImage()
        fig.canvas.draw_idle()

    def onPrevStat(event):
        currentStatIndex[0] = (currentStatIndex[0] - 1) % 3
        drawStats()
        fig.canvas.draw_idle()

    def onNextStat(event):
        currentStatIndex[0] = (currentStatIndex[0] + 1) % 3
        drawStats()
        fig.canvas.draw_idle()

    btnNextImg.on_clicked(onNextImage)
    btnPrevImg.on_clicked(onPrevImage)
    
    btnNextStat.on_clicked(onNextStat)
    btnPrevStat.on_clicked(onPrevStat)
    

    plt.show()



# -------------------------------------------------
# STATISTICS
# -------------------------------------------------

def computeStatistics(allAnnotations):
    digitCounts = []
    classCounts = np.zeros(10)
    boxSizes = []

    for annotations in allAnnotations:
        digitCounts.append(len(annotations))

        for digitLabel, x, y, w, h in annotations:
            classCounts[digitLabel] += 1
            boxSizes.append(w)

    return digitCounts, classCounts, boxSizes


def setupPlot(title, xlabel, ylabel):
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="MNIST Object Detection Dataset Statistics"
    )

    parser.add_argument(
        "--version",
        required=True,
        choices=["versionA", "versionB", "versionC", "versionD"],
        help="Dataset version to analyze"
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test"],
        help="Dataset split to analyze"
    )

    args = parser.parse_args()

    baseDirectory = os.path.dirname(os.path.abspath(__file__))
    versionDirectory = os.path.join(
        baseDirectory, "output", args.version
    )

    imageFile = os.path.join(
        versionDirectory,
        f"{args.split}-images-ubyte.bin"
    )

    labelFile = os.path.join(
        versionDirectory,
        f"{args.split}-labels-ubyte.bin"
    )

    numberOfImages = 60000 if args.split == "train" else 10000
    imageSize = 128

    print("A carregar metadados YOLO...")
    # images aqui será lista de paths, annotations será a lista de tuplos
    images, annotations = load_yolo_data(versionDirectory, args.split)

    # ---- STATISTICS (dados, não plots) ----
    digitCounts, classCounts, boxSizes = computeStatistics(annotations)

    # ---- INTERFACE ÚNICA ----
    runInterface(
        images=images[:50],
        annotations=annotations[:50],
        digitCounts=digitCounts,
        classCounts=classCounts,
        boxSizes=boxSizes
    )



if __name__ == "__main__":
    main()
