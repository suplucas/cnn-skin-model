#!/bin/bash
# Plano "sane" — testa a hipotese: pipeline OpenCV grayscale mata as features
# dos backbones pre-treinados. Comeca com overfit check (sanidade) e entao
# treina as variantes.
# Uso: bash sane_sweep.sh
# Monitorar: tail -f sane_sweep.log

LOG=sane_sweep.log
BASE="training.epochs=50 training.optimizer=adam training.lr=0.0001 training.unfreeze_blocks=2 training.pos_weight=1.96 training.weight_decay=0.0001 dp=true"

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
# SANIDADE — o pipeline consegue aprender? (rápido, ~10 min cada)
# Se o overfit check FALHAR nos dois, investigar antes de treinar.
# ==========================================

echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "[$(date)] S0a — overfit check SEM OpenCV (hipótese principal)" | tee -a $LOG
echo "==========================================" | tee -a $LOG
python model_engineering/overfit_check.py model=vgg16 training.unfreeze_blocks=2 \
    preprocessing.opencv=false >> $LOG 2>&1
echo "[$(date)] S0a exit=$?" | tee -a $LOG

echo "" | tee -a $LOG
echo "==========================================" | tee -a $LOG
echo "[$(date)] S0b — overfit check COM OpenCV (controle)" | tee -a $LOG
echo "==========================================" | tee -a $LOG
python model_engineering/overfit_check.py model=vgg16 training.unfreeze_blocks=2 >> $LOG 2>&1
echo "[$(date)] S0b exit=$?" | tee -a $LOG

# ==========================================
# TREINAMENTOS — dataset RAW (dataset.csv, sem preprocessed_dir)
# ==========================================

run "01-VGG16-rgb-512" \
    model=vgg16 $BASE preprocessing.opencv=false preprocessing.input_size=512

run "02-VGG16-rgb-224" \
    model=vgg16 $BASE preprocessing.opencv=false preprocessing.input_size=224

run "03-VGG16-rgb-512-aug" \
    model=vgg16 $BASE preprocessing.opencv=false preprocessing.input_size=512 \
    preprocessing.crop=random preprocessing.random_erasing=true

run "04-ResNet152-rgb-512" \
    model=resnet152 $BASE preprocessing.opencv=false preprocessing.input_size=512

run "05-Swin-rgb-512" \
    model=swin $BASE preprocessing.opencv=false preprocessing.input_size=512

run "06-VGG16-opencv-controle" \
    model=vgg16 $BASE

# ==========================================

echo "" | tee -a $LOG
echo "=== FIM: $(date) ===" | tee -a $LOG
echo "=== Resultados em MLflow: ===" | tee -a $LOG
echo "mlflow ui --port 5000 --backend-store-uri sqlite:///../cnn-skin-model-runs/mlflow/mlflow.db" | tee -a $LOG
