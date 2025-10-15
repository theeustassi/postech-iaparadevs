# ========================================
# Setup TensorFlow 2.10 + GPU 
# Seguindo documentacao oficial do TensorFlow
# https://www.tensorflow.org/install/pip?hl=pt-br#windows-native
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " TensorFlow 2.10 GPU - Setup Oficial" -ForegroundColor Cyan
Write-Host " Python 3.9 | CUDA 11.2 | cuDNN 8.1" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Inicializar Conda para PowerShell
Write-Host "[INIT] Inicializando Conda..." -ForegroundColor Yellow
$condaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $condaPath) {
    # Tentar caminhos comuns do Conda
    $possiblePaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\Continuum\miniconda3\Scripts\conda.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $condaPath = $path
            break
        }
    }
}

if (-not $condaPath) {
    Write-Host "  [ERRO] Conda nao encontrado!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor, instale o Miniconda:" -ForegroundColor Yellow
    Write-Host "  1. Baixe: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor White
    Write-Host "  2. Execute o instalador" -ForegroundColor White
    Write-Host "  3. Marque: Add to PATH" -ForegroundColor White
    Write-Host "  4. Reinicie o PowerShell" -ForegroundColor White
    Write-Host "  5. Execute este script novamente" -ForegroundColor White
    exit 1
}

Write-Host "  [OK] Conda encontrado: $condaPath" -ForegroundColor Green

# Definir funcao conda para este script
if ($condaPath) {
    $condaDir = Split-Path -Parent $condaPath
    $condaRoot = Split-Path -Parent $condaDir
    $condaHookPath = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"
    if (Test-Path $condaHookPath) {
        . $condaHookPath
        Write-Host "  [OK] Conda hook carregado" -ForegroundColor Green
    }
}

# Passo 1: Aceitar termos de servico do Conda
Write-Host ""
Write-Host "[1/7] Configurando Conda..." -ForegroundColor Yellow
& $condaPath config --set channel_priority strict 2>$null
& $condaPath config --add channels conda-forge 2>$null
& $condaPath config --set auto_activate_base false 2>$null
Write-Host "  [OK] Conda configurado" -ForegroundColor Green

# Passo 2: Remover ambiente antigo se existir
Write-Host ""
Write-Host "[2/7] Preparando ambiente..." -ForegroundColor Yellow
$envName = "venv"
& $condaPath env remove -n $envName -y 2>$null
Write-Host "  [OK] Pronto para criar novo ambiente" -ForegroundColor Green

# Passo 3: Criar ambiente com Python 3.9 (conforme documentacao oficial)
Write-Host ""
Write-Host "[3/7] Criando ambiente Conda (Python 3.9)..." -ForegroundColor Yellow
& $condaPath create -n $envName python=3.9 -c conda-forge -y
Write-Host "  [OK] Ambiente '$envName' criado" -ForegroundColor Green

# Passo 4: Instalar CUDA Toolkit e cuDNN via conda (recomendado pela documentacao)
Write-Host ""
Write-Host "[4/7] Instalando CUDA 11.2 + cuDNN 8.1 via conda..." -ForegroundColor Yellow
Write-Host "  (Isso vai levar alguns minutos...)" -ForegroundColor Gray
& $condaPath install -n $envName -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
Write-Host "  [OK] CUDA e cuDNN instalados" -ForegroundColor Green

# Passo 5: Instalar TensorFlow 2.10 (conforme documentacao oficial)
Write-Host ""
Write-Host "[5/7] Instalando TensorFlow 2.10..." -ForegroundColor Yellow
Write-Host "  (Aguarde 3-5 minutos...)" -ForegroundColor Gray
& $condaPath run -n $envName pip install "tensorflow<2.11"
Write-Host "  [OK] TensorFlow instalado" -ForegroundColor Green

# Passo 6: Instalar bibliotecas auxiliares
Write-Host ""
Write-Host "[6/7] Instalando bibliotecas..." -ForegroundColor Yellow
& $condaPath run -n $envName pip install numpy pandas matplotlib seaborn scikit-learn opencv-python pillow tqdm jupyterlab ipykernel kaggle
Write-Host "  [OK] Bibliotecas instaladas" -ForegroundColor Green

# Registrar kernel Jupyter
Write-Host "  Registrando kernel Jupyter..." -ForegroundColor Gray
& $condaPath run -n $envName python -m ipykernel install --user --name=venv --display-name="Python 3.9 (TF 2.10 GPU)"
Write-Host "  [OK] Kernel registrado" -ForegroundColor Green

# Passo 7: Testar GPU (conforme documentacao oficial)
Write-Host ""
Write-Host "[7/7] Testando GPU..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
& $condaPath run -n $envName python -c @"
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU Built:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs detectadas:', len(gpus))
for gpu in gpus:
    print(' -', gpu.name)
    
if len(gpus) > 0:
    print('')
    print('✓ SUCCESS! GPU DETECTADA!')
    print('Treinamento sera rapido (30-60 min)')
else:
    print('')
    print('⚠ GPU nao detectada')
    print('Treinamento usara CPU (2-4 horas)')
"@
Write-Host "========================================" -ForegroundColor Cyan

# Resumo final
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " CONFIGURACAO CONCLUIDA!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Configuracao GPU (adicione ao script ou .bashrc):" -ForegroundColor Yellow
Write-Host "  Nao necessario - Conda ja configurou automaticamente!" -ForegroundColor Green
Write-Host ""
Write-Host "Para ativar o ambiente:" -ForegroundColor Cyan
Write-Host "  conda activate $envName" -ForegroundColor White
Write-Host ""
Write-Host "No VS Code:" -ForegroundColor Cyan
Write-Host "  1. Ctrl+Shift+P" -ForegroundColor White
Write-Host "  2. Python: Select Interpreter" -ForegroundColor White
Write-Host "  3. Selecione: Python 3.9.x ('venv')" -ForegroundColor White
Write-Host "  4. Abra: Fase 1\Tech_Challenge_Extra\notebooks\02_treinamento_modelo.ipynb" -ForegroundColor White
Write-Host "  5. Kernel: Python 3.9 (TF 2.10 GPU)" -ForegroundColor White
Write-Host ""
Write-Host "Referencia: https://www.tensorflow.org/install/pip?hl=pt-br#windows-native" -ForegroundColor Gray
Write-Host ""
