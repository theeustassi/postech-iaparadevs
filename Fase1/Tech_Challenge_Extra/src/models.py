"""
Módulo com arquiteturas de CNN para classificação de pneumonia.
Contém modelos customizados e com transfer learning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam


def create_simple_cnn(input_shape=(224, 224, 3), dropout_rate=0.5):
    """
    Cria uma CNN simples para classificação binária.
    Boa para entender conceitos básicos e como baseline.
    
    Args:
        input_shape: Shape da imagem de entrada (altura, largura, canais)
        dropout_rate: Taxa de dropout para regularização
        
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Quarta camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Camada de saída (classificação binária)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_vgg16_transfer(input_shape=(224, 224, 3), trainable_layers=4):
    """
    Cria modelo usando VGG16 pré-treinado (Transfer Learning).
    
    Args:
        input_shape: Shape da imagem de entrada
        trainable_layers: Número de camadas do topo para treinar
        
    Returns:
        Modelo Keras compilado
    """
    # Carregar VGG16 sem a camada de classificação
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar camadas base (fine-tuning seletivo)
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Adicionar camadas customizadas
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_resnet50_transfer(input_shape=(224, 224, 3), trainable_layers=10):
    """
    Cria modelo usando ResNet50 pré-treinado (Transfer Learning).
    ResNet geralmente oferece melhor desempenho que VGG.
    
    Args:
        input_shape: Shape da imagem de entrada
        trainable_layers: Número de camadas do topo para treinar
        
    Returns:
        Modelo Keras compilado
    """
    # Carregar ResNet50 sem a camada de classificação
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar camadas base
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Adicionar camadas customizadas
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_mobilenetv2_transfer(input_shape=(224, 224, 3), trainable_layers=20):
    """
    Cria modelo usando MobileNetV2 (modelo mais leve e rápido).
    Bom para quando há limitação de recursos computacionais.
    
    Args:
        input_shape: Shape da imagem de entrada
        trainable_layers: Número de camadas do topo para treinar
        
    Returns:
        Modelo Keras compilado
    """
    # Carregar MobileNetV2 sem a camada de classificação
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar camadas base
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Adicionar camadas customizadas
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def print_model_summary(model, model_name="Modelo"):
    """
    Imprime resumo do modelo de forma formatada.
    
    Args:
        model: Modelo Keras
        model_name: Nome do modelo para exibição
    """
    print(f"\n{'='*60}")
    print(f"📊 Resumo do {model_name}")
    print(f"{'='*60}")
    
    # Contar parâmetros
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\n📊 Estatísticas do Modelo:")
    print(f"   Total de parâmetros: {total_params:,}")
    print(f"   Parâmetros treináveis: {trainable_params:,}")
    print(f"   Parâmetros não-treináveis: {non_trainable_params:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Teste dos modelos
    print("🧪 Testando arquiteturas de modelos...\n")
    
    # 1. CNN Simples
    print("1️⃣ Criando CNN Simples...")
    model_simple = create_simple_cnn()
    print_model_summary(model_simple, "CNN Simples")
    
    # 2. VGG16
    print("\n2️⃣ Criando VGG16 Transfer Learning...")
    model_vgg = create_vgg16_transfer()
    print_model_summary(model_vgg, "VGG16 Transfer Learning")
    
    # 3. ResNet50
    print("\n3️⃣ Criando ResNet50 Transfer Learning...")
    model_resnet = create_resnet50_transfer()
    print_model_summary(model_resnet, "ResNet50 Transfer Learning")
    
    print("\n✅ Todos os modelos foram criados com sucesso!")
