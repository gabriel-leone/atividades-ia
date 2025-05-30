"""
Análise de Similaridade de Crimes entre Municípios
=================================================

Este script implementa uma análise de similaridade entre municípios
baseada nas descrições de crimes, utilizando embeddings de texto e
análise de grafos para identificar padrões de criminalidade.

Autor: Gabriel
Data: 30/05/2025
"""

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


# Configurações
DATA_FILE = 'data/assaltos.xlsx'
THRESHOLD = 0.7
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
OUTPUT_DIR = 'outputs'

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados do arquivo Excel.
    
    Args:
        file_path (str): Caminho para o arquivo de dados
        
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Dados carregados: {len(df)} registros de {df['municipio'].nunique()} municípios")
        return df
    except FileNotFoundError:
        logger.error(f"Arquivo {file_path} não encontrado")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise


def generate_embeddings(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Gera embeddings das descrições de crimes.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        model_name (str): Nome do modelo de embeddings
        
    Returns:
        pd.DataFrame: DataFrame com embeddings adicionados
    """
    logger.info("Carregando modelo de embeddings...")
    model = SentenceTransformer(model_name)
    
    logger.info("Gerando embeddings das descrições...")
    embeddings = model.encode(df['descricao'].tolist(), show_progress_bar=True)
    
    df_copy = df.copy()
    df_copy['embedding'] = embeddings.tolist()
    
    logger.info("Embeddings gerados com sucesso")
    return df_copy


def calculate_city_embeddings(df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """
    Calcula embeddings médios por cidade.
    
    Args:
        df (pd.DataFrame): DataFrame com embeddings
        
    Returns:
        tuple: Lista de cidades e matriz de similaridade
    """
    logger.info("Calculando embeddings médios por cidade...")
    
    city_embeddings = (
        df.groupby('municipio')['embedding']
        .apply(lambda emb_list: np.mean(emb_list.tolist(), axis=0))
    )
    
    city_embedding_dict = city_embeddings.to_dict()
    cities = list(city_embedding_dict.keys())
    vectors = np.array(list(city_embedding_dict.values()))
    
    logger.info(f"Embeddings calculados para {len(cities)} cidades")
    return cities, vectors


def calculate_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de similaridade entre cidades.
    
    Args:
        vectors (np.ndarray): Vetores de embeddings das cidades
        
    Returns:
        np.ndarray: Matriz de similaridade
    """
    logger.info("Calculando matriz de similaridade...")
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix


def build_graph(cities: list, similarity_matrix: np.ndarray, threshold: float) -> nx.Graph:
    """
    Constrói grafo baseado na similaridade.
    
    Args:
        cities (list): Lista de cidades
        similarity_matrix (np.ndarray): Matriz de similaridade
        threshold (float): Limiar de similaridade
        
    Returns:
        nx.Graph: Grafo construído
    """
    logger.info(f"Construindo grafo com threshold {threshold}...")
    
    G = nx.Graph()
    G.add_nodes_from(cities)
    
    num_cities = len(cities)
    edges_added = 0
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            similarity = similarity_matrix[i][j]
            if similarity >= threshold:
                G.add_edge(cities[i], cities[j], weight=similarity)
                edges_added += 1
    
    logger.info(f"Grafo construído com {edges_added} arestas")
    return G


def calculate_transition_matrix(graph: nx.Graph, cities: list) -> np.ndarray:
    """
    Calcula matriz de transição do grafo.
    
    Args:
        graph (nx.Graph): Grafo de similaridade
        cities (list): Lista de cidades
        
    Returns:
        np.ndarray: Matriz de transição
    """
    logger.info("Calculando matriz de transição...")
    
    adj_matrix = nx.to_numpy_array(graph, nodelist=cities)
    
    # Evitar divisão por zero
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    
    # Normalizar para matriz de transição
    transition_matrix = adj_matrix / row_sums[:, np.newaxis]
    
    logger.info("Matriz de transição calculada")
    return transition_matrix


def calculate_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula distribuição estacionária usando autovalores.
    
    Args:
        transition_matrix (np.ndarray): Matriz de transição
        
    Returns:
        np.ndarray: Distribuição estacionária
    """
    logger.info("Calculando distribuição estacionária...")
    
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    
    # Encontrar vetor próprio associado ao autovalor 1
    stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    
    # Normalizar para distribuição de probabilidade
    stationary = stationary[:, 0]
    stationary = stationary / stationary.sum()
    
    logger.info("Distribuição estacionária calculada")
    return stationary


def save_results(cities: list, similarity_matrix: np.ndarray, 
                transition_matrix: np.ndarray, stationary_distribution: np.ndarray,
                output_dir: str) -> None:
    """
    Salva todos os resultados em arquivos.
    
    Args:
        cities (list): Lista de cidades
        similarity_matrix (np.ndarray): Matriz de similaridade
        transition_matrix (np.ndarray): Matriz de transição
        stationary_distribution (np.ndarray): Distribuição estacionária
        output_dir (str): Diretório de saída
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar matriz de similaridade
    similarity_df = pd.DataFrame(similarity_matrix, index=cities, columns=cities)
    similarity_path = os.path.join(output_dir, f"matriz_similaridade_{timestamp}.csv")
    similarity_df.to_csv(similarity_path)
    logger.info(f"Matriz de similaridade salva em: {similarity_path}")
    
    # Salvar matriz de transição
    transition_df = pd.DataFrame(transition_matrix, index=cities, columns=cities)
    transition_path = os.path.join(output_dir, f"matriz_transicao_{timestamp}.csv")
    transition_df.to_csv(transition_path)
    logger.info(f"Matriz de transição salva em: {transition_path}")
    
    # Salvar distribuição estacionária
    stationary_df = pd.DataFrame({
        'Cidade': cities,
        'Probabilidade_Estacionaria': stationary_distribution
    }).sort_values(by='Probabilidade_Estacionaria', ascending=False)
    
    stationary_path = os.path.join(output_dir, f"distribuicao_estacionaria_{timestamp}.csv")
    stationary_df.to_csv(stationary_path, index=False)
    logger.info(f"Distribuição estacionária salva em: {stationary_path}")
    
    # Salvar relatório resumido
    save_summary_report(stationary_df, output_dir, timestamp)
    
    # Mostrar top 10 no console
    print("\nTOP 10 MUNICÍPIOS POR PROBABILIDADE ESTACIONÁRIA:")
    print("=" * 55)
    for i, row in stationary_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['Cidade']:<30} {row['Probabilidade_Estacionaria']:.4f}")


def save_summary_report(stationary_df: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """
    Salva relatório resumido da análise.
    
    Args:
        stationary_df (pd.DataFrame): DataFrame com distribuição estacionária
        output_dir (str): Diretório de saída
        timestamp (str): Timestamp para o arquivo
    """
    report_path = os.path.join(output_dir, f"relatorio_analise_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE ANÁLISE DE SIMILARIDADE DE CRIMES\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Número de municípios analisados: {len(stationary_df)}\n")
        f.write(f"Threshold de similaridade: {THRESHOLD}\n")
        f.write(f"Modelo de embeddings: {MODEL_NAME}\n\n")
        
        f.write("TOP 10 MUNICÍPIOS POR PROBABILIDADE ESTACIONÁRIA:\n")
        f.write("-" * 45 + "\n")
        
        for i, row in stationary_df.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['Cidade']:<30} {row['Probabilidade_Estacionaria']:.4f}\n")
        
        f.write("\nESTATÍSTICAS DA DISTRIBUIÇÃO ESTACIONÁRIA:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Média: {stationary_df['Probabilidade_Estacionaria'].mean():.4f}\n")
        f.write(f"Desvio padrão: {stationary_df['Probabilidade_Estacionaria'].std():.4f}\n")
        f.write(f"Máximo: {stationary_df['Probabilidade_Estacionaria'].max():.4f}\n")
        f.write(f"Mínimo: {stationary_df['Probabilidade_Estacionaria'].min():.4f}\n")
        f.write(f"Mediana: {stationary_df['Probabilidade_Estacionaria'].median():.4f}\n")
    
    logger.info(f"Relatório resumido salvo em: {report_path}")


def main():
    """Executa a análise completa."""
    logger.info("Iniciando análise de similaridade de crimes...")
    
    try:
        # 1. Carregar dados
        df = load_data(DATA_FILE)
        
        # 2. Gerar embeddings
        df_with_embeddings = generate_embeddings(df, MODEL_NAME)
        
        # 3. Calcular embeddings por cidade
        cities, vectors = calculate_city_embeddings(df_with_embeddings)
        
        # 4. Calcular matriz de similaridade
        similarity_matrix = calculate_similarity_matrix(vectors)
        
        # 5. Construir grafo
        graph = build_graph(cities, similarity_matrix, THRESHOLD)
        
        # 6. Calcular matriz de transição
        transition_matrix = calculate_transition_matrix(graph, cities)
        
        # 7. Calcular distribuição estacionária
        stationary_distribution = calculate_stationary_distribution(transition_matrix)
        
        # 8. Salvar resultados
        save_results(cities, similarity_matrix, transition_matrix, 
                    stationary_distribution, OUTPUT_DIR)
        
        logger.info("Análise completa finalizada com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {e}")
        raise


if __name__ == "__main__":
    main()