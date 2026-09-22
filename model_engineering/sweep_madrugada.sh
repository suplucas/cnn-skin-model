#!/bin/bash
# SWEEP DA MADRUGADA — letterbox (sem perda de conteudo) + front-end
# de reducao (filtro grande) antes do backbone.
#
# Hipoteses:
#   L:  letterbox cobre 100% da cena -> resolve o colapso do center crop
#   FE: front-end (avg ou conv k=32) reduz 1024->512 com menos aliasing
#       que a bilinear direta
#
# Uso: bash sweep_madrugada.sh
# Monitorar: tail -f sweep_madrugada.log

LOG=sweep_madrugada.log
BASE="training.epochs=50 training.optimizer=adam training.lr=0.0001 training.unfreeze_blocks=2 training.pos_weight=1.96 training.weight_decay=0.0001 dp=true"
RGB="preprocessing.opencv=false preprocessing.grayscale=false"

echo "=== INICIO: $(date) ===" | tee $LOG

run() {
    local name="$1"
    shift
    echo "" | tee -a $LOG
    echo "==========================================" | tee -a $LOG
    echo "[$(date)] $name" | tee -a $LOG
    echo "Comando: python model_engineering/main.py $@" | tee -a $LOG
    echo "==========================================" | tee -a $LOG
    python model_engineering/main.py "$@" >> $LOG 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] OK $name concluído" | tee -a $LOG
    else
        echo "[$(date)] X $name FALHOU (exit=$exit_code)" | tee -a $LOG
    fi
}

# ==========================================
# SANIDADE — pipeline+frontend conseguem aprender? (~10-15 min cada)
# ==========================================

echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "[$(date)] S0a — overfit check: letterbox 512" | tee -a $LOG
echo "==========================================" | tee -a $LOG
python model_engineering/overfit_check.py model=vgg16 training.unfreeze_blocks=2 \
    $RGB preprocessing.input_size=512 preprocessing.crop=letterbox >> $LOG 2>&1
echo "[$(date)] S0a exit=$?" | tee -a $LOG

echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "[$(date)] S0b — overfit check: letterbox 1024 + frontend conv" | tee -a $LOG
echo "==========================================" | tee -a $LOG
python model_engineering/overfit_check.py model=vgg16 training.unfreeze_blocks=2 \
    $RGB preprocessing.input_size=1024 preprocessing.crop=letterbox frontend=conv >> $LOG 2>&1
echo "[$(date)] S0b exit=$?" | tee -a $LOG

# ==========================================
# TREINAMENTOS — dataset RAW (dataset.csv)
# ==========================================

# baseline letterbox: 100% da cena, bilinear no resize (padrao YOLO)
run "01-VGG16-letterbox-512" \
    model=vgg16 $BASE $RGB \
    preprocessing.input_size=512 preprocessing.crop=letterbox

# mesmo letterbox com front-end fixo (AveragePool s=2): 1024 -> 512
run "02-VGG16-letterbox-1024-avg" \
    model=vgg16 $BASE $RGB \
    preprocessing.input_size=1024 preprocessing.crop=letterbox frontend=avg

# MESMO letterbox c/ filtro grande 32x32 aprendido (conv k=32 s=2): 1024 -> 512
run "03-VGG16-letterbox-1024-conv" \
    model=vgg16 $BASE $RGB \
    preprocessing.input_size=1024 preprocessing.crop=letterbox frontend=conv

# controle A: center crop 512 (pipeline antigo sem OpenCV)
run "04-VGG16-center-512" \
    model=vgg16 $BASE $RGB \
    preprocessing.input_size=512

# variante de escala: letterbox 224 (tamanho nativo do ImageNet)
run "05-VGG16-letterbox-224" \
    model=vgg16 $BASE $RGB \
    preprocessing.input_size=224 preprocessing.crop=letterbox

# outros backbones com letterbox 512
run "06-ResNet152-letterbox-512" \
    model=resnet152 $BASE $RGB \
    preprocessing.input_size=512 preprocessing.crop=letterbox

run "07-Swin-letterbox-512" \
    model=swin $BASE $RGB \
    preprocessing.input_size=512 preprocessing.crop=letterbox

# controle B: pipeline ANTERIOR (OpenCV grayscale + center crop),
# mesmos hiperparametros de treino para A/B limpo
run "08-VGG16-opencv-center-512" \
    model=vgg16 $BASE

# ==========================================

echo "" | tee -a $LOG
echo "=== FIM: $(date) ===" | tee -a $LOG
echo "=== Resultados em MLflow: ===" | tee -a $LOG
echo "mlflow ui --port 5000 --backend-store-uri sqlite:///../cnn-skin-model-runs/mlflow/mlflow.db" | tee -a $LOG
