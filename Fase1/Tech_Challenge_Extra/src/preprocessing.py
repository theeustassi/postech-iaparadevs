"""
Módulo de pré-processamento de imagens para detecção de pneumonia.
Contém funções para carregar, processar e aumentar dados de imagens médicas.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Configurações padrão
IMG_SIZE = (224, 224)  # Tamanho padrão para modelos pré-treinados
BATCH_SIZE = 32


def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, 
                           augmentation=True, validation_split=0.2):
    """
    Cria geradores de dados para treino, validação e teste.
    
    Args:
        data_dir: Caminho para o diretório com os dados
        img_size: Tupla (altura, largura) para redimensionar imagens
        batch_size: Tamanho do batch
        augmentation: Se True, aplica data augmentation no treino
        validation_split: Proporção dos dados de treino para validação
        
    Returns:
        train_generator, val_generator, test_generator
    """
    data_dir = Path(data_dir)
    
    # Gerador para treino COM data augmentation
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalização
            rotation_range=15,            # Rotação leve
            width_shift_range=0.1,        # Deslocamento horizontal
            height_shift_range=0.1,       # Deslocamento vertical
            shear_range=0.1,              # Cisalhamento
            zoom_range=0.1,               # Zoom
            horizontal_flip=True,         # Espelhamento horizontal
            fill_mode='nearest',          # Preenchimento
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    # Gerador para validação e teste (apenas normalização)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Carregar dados de treino
    train_generator = train_datagen.flow_from_directory(
        data_dir / 'train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Classificação binária
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Carregar dados de validação
    val_generator = train_datagen.flow_from_directory(
        data_dir / 'train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Carregar dados de teste
    test_generator = test_datagen.flow_from_directory(
        data_dir / 'test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def visualize_samples(generator, num_samples=9):
    """
    Visualiza amostras de imagens do gerador.
    
    Args:
        generator: Gerador de dados
        num_samples: Número de amostras para visualizar
    """
    # Pegar um batch
    images, labels = next(generator)
    
    # Calcular grid
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i])
        
        # Label
        label = "PNEUMONIA" if labels[i] == 1 else "NORMAL"
        color = "red" if labels[i] == 1 else "green"
        
        plt.title(f"Classe: {label}", color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_dataset_distribution(data_dir):
    """
    Analisa e imprime a distribuição do dataset.
    
    Args:
        data_dir: Caminho para o diretório com os dados
        
    Returns:
        dict com estatísticas
    """
    data_dir = Path(data_dir)
    stats = {}
    
    print("📊 Distribuição do Dataset\n")
    print("=" * 60)
    
    for split in ['train', 'test', 'val']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            continue
            
        normal_dir = split_dir / 'NORMAL'
        pneumonia_dir = split_dir / 'PNEUMONIA'
        
        normal_count = len(list(normal_dir.glob('*.jpeg'))) + len(list(normal_dir.glob('*.jpg')))
        pneumonia_count = len(list(pneumonia_dir.glob('*.jpeg'))) + len(list(pneumonia_dir.glob('*.jpg')))
        total = normal_count + pneumonia_count
        
        stats[split] = {
            'normal': normal_count,
            'pneumonia': pneumonia_count,
            'total': total
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Normal:    {normal_count:5d} ({normal_count/total*100:.1f}%)")
        print(f"  Pneumonia: {pneumonia_count:5d} ({pneumonia_count/total*100:.1f}%)")
        print(f"  Total:     {total:5d}")
        
    print("\n" + "=" * 60)
    
    return stats


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Carrega e pré-processa uma única imagem.
    
    Args:
        image_path: Caminho para a imagem
        img_size: Tamanho de saída da imagem
        
    Returns:
        Imagem processada como array numpy
    """
    # Ler imagem
    img = cv2.imread(str(image_path))
    
    # Converter BGR para RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    img = cv2.resize(img, img_size)
    
    # Normalizar
    img = img / 255.0
    
    return img


def get_class_weights(train_generator):
    """
    Calcula pesos das classes para lidar com desbalanceamento.
    
    Args:
        train_generator: Gerador de dados de treino
        
    Returns:
        dict com pesos das classes
    """
    from sklearn.utils import class_weight
    
    # Obter labels
    labels = train_generator.classes
    
    # Calcular pesos
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n⚖️ Pesos das Classes (para balanceamento):")
    print(f"  Classe 0 (NORMAL):    {class_weight_dict[0]:.2f}")
    print(f"  Classe 1 (PNEUMONIA): {class_weight_dict[1]:.2f}")
    
    return class_weight_dict


if __name__ == "__main__":
    # Teste do módulo
    print("🔬 Testando módulo de pré-processamento...")
    
    # Definir caminho (ajuste conforme necessário)
    data_path = Path(__file__).parent.parent / "data" / "chest_xray"
    
    if data_path.exists():
        # Analisar distribuição
        stats = analyze_dataset_distribution(data_path)
        
        # Criar geradores
        print("\n🔄 Criando geradores de dados...")
        train_gen, val_gen, test_gen = create_data_generators(data_path)
        
        print(f"✅ Treino: {train_gen.samples} amostras")
        print(f"✅ Validação: {val_gen.samples} amostras")
        print(f"✅ Teste: {test_gen.samples} amostras")
        
        # Calcular pesos
        class_weights = get_class_weights(train_gen)
        
    else:
        print(f"❌ Dataset não encontrado em: {data_path}")
        print("   Execute primeiro: python src/download_dataset.py")
