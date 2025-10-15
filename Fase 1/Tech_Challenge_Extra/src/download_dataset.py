"""
Script para baixar o dataset de pneumonia do Kaggle.
Certifique-se de ter suas credenciais do Kaggle configuradas.
"""

import os
import zipfile
from pathlib import Path
import shutil

def download_dataset():
    """
    Baixa e extrai o dataset de pneumonia do Kaggle.
    """
    print("🔍 Verificando se o dataset já existe...")
    
    # Diretório base do projeto
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    chest_xray_dir = data_dir / "chest_xray"
    
    # Verificar se o dataset já existe
    if chest_xray_dir.exists() and (chest_xray_dir / "train").exists():
        print("✅ Dataset já existe! Não é necessário baixar novamente.")
        print(f"📁 Localização: {chest_xray_dir}")
        return
    
    print("📥 Baixando dataset do Kaggle...")
    print("⏳ Isso pode levar alguns minutos dependendo da sua conexão...")
    
    try:
        # Importar kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Autenticar
        api = KaggleApi()
        api.authenticate()
        
        # Criar diretório de dados
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Baixar dataset
        dataset_name = "paultimothymooney/chest-xray-pneumonia"
        print(f"📦 Baixando {dataset_name}...")
        
        api.dataset_download_files(
            dataset_name,
            path=data_dir,
            unzip=True
        )
        
        print("✅ Download concluído!")
        
        # Verificar estrutura
        if chest_xray_dir.exists():
            print("\n📊 Estrutura do dataset:")
            for split in ["train", "test", "val"]:
                split_dir = chest_xray_dir / split
                if split_dir.exists():
                    normal_count = len(list((split_dir / "NORMAL").glob("*.jpeg")))
                    pneumonia_count = len(list((split_dir / "PNEUMONIA").glob("*.jpeg")))
                    print(f"  {split.upper()}:")
                    print(f"    - NORMAL: {normal_count} imagens")
                    print(f"    - PNEUMONIA: {pneumonia_count} imagens")
            
            print(f"\n✨ Dataset pronto para uso em: {chest_xray_dir}")
        else:
            print("⚠️ Aviso: Estrutura do dataset diferente do esperado.")
            
    except ImportError:
        print("❌ Erro: Biblioteca 'kaggle' não encontrada.")
        print("   Instale com: pip install kaggle")
        return
    except Exception as e:
        print(f"❌ Erro ao baixar dataset: {str(e)}")
        print("\n💡 Certifique-se de:")
        print("   1. Ter uma conta no Kaggle")
        print("   2. Ter configurado suas credenciais do Kaggle")
        print("   3. O arquivo kaggle.json está em ~/.kaggle/ (Linux/Mac)")
        print("      ou C:\\Users\\<usuario>\\.kaggle\\ (Windows)")
        print("\n📖 Instruções: https://github.com/Kaggle/kaggle-api#api-credentials")
        return

if __name__ == "__main__":
    print("🫁 Sistema de Detecção de Pneumonia - Download do Dataset")
    print("=" * 60)
    download_dataset()
    print("=" * 60)
    print("✅ Processo finalizado!")
