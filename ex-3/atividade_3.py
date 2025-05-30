"""
Análise de Clusters de Crimes entre Municípios
==============================================

Este script implementa uma análise de clustering de municípios baseada
nas descrições de crimes, utilizando embeddings de texto, análise de
grafos K-NN e algoritmos de detecção de comunidades.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from networkx.algorithms.community import label_propagation_communities, modularity


# Configurações
DATA_FILE = 'data/assaltos.xlsx'
K_NEIGHBORS = 5
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo dos gráficos
plt.style.use('default')
if HAS_SEABORN:
    sns.set_palette("husl")


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


def calculate_city_embeddings(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Calcula embeddings médios por cidade.
    
    Args:
        df (pd.DataFrame): DataFrame com embeddings
        
    Returns:
        tuple: Lista de cidades, matriz de vetores e dicionário cidade->vetor
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
    return cities, vectors, city_embedding_dict


def build_knn_graph(cities: List[str], similarity_matrix: np.ndarray, k: int) -> nx.Graph:
    """
    Constrói grafo K-NN baseado na similaridade.
    
    Args:
        cities (List[str]): Lista de cidades
        similarity_matrix (np.ndarray): Matriz de similaridade
        k (int): Número de vizinhos mais próximos
        
    Returns:
        nx.Graph: Grafo K-NN construído
    """
    logger.info(f"Construindo grafo K-NN com K={k}...")
    
    G = nx.Graph()
    G.add_nodes_from(cities)
    
    num_cities = len(cities)
    edges_added = 0
    
    for i in range(num_cities):
        # Encontrar os K vizinhos mais similares
        sim_scores = list(enumerate(similarity_matrix[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        neighbors = sim_scores[1:k+1]  # Ignora o próprio (índice 0)
        
        for j, sim in neighbors:
            if not G.has_edge(cities[i], cities[j]):
                G.add_edge(cities[i], cities[j], weight=sim)
                edges_added += 1
    
    logger.info(f"Grafo K-NN construído com {edges_added} arestas")
    return G


def detect_communities(graph: nx.Graph) -> Tuple[List[Set], Dict[int, List[str]]]:
    """
    Detecta comunidades no grafo usando label propagation.
    
    Args:
        graph (nx.Graph): Grafo de entrada
        
    Returns:
        tuple: Lista de comunidades e dicionário cluster_id -> cidades
    """
    logger.info("Detectando comunidades...")
    
    communities = list(label_propagation_communities(graph))
    
    # Organizar clusters em dicionário
    cluster_dict = {}
    for idx, community in enumerate(communities):
        cluster_dict[idx] = list(community)
    
    # Calcular modularidade
    mod = modularity(graph, communities)
    
    logger.info(f"{len(communities)} clusters encontrados com modularidade: {mod:.4f}")
    return communities, cluster_dict


def analyze_clusters(df: pd.DataFrame, cluster_dict: Dict[int, List[str]]) -> pd.DataFrame:
    """
    Analisa as características dos clusters.
    
    Args:
        df (pd.DataFrame): DataFrame original
        cluster_dict (Dict): Dicionário cluster_id -> cidades
        
    Returns:
        pd.DataFrame: DataFrame com estatísticas dos clusters
    """
    logger.info("Analisando características dos clusters...")
    
    cluster_stats = []
    
    for cluster_id, cities in cluster_dict.items():
        cluster_data = df[df['municipio'].isin(cities)]
        
        stats = {
            'cluster_id': cluster_id,
            'num_cities': len(cities),
            'total_crimes': len(cluster_data),
            'avg_crimes_per_city': len(cluster_data) / len(cities),
            'sample_cities': ', '.join(cities[:5])
        }
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats).sort_values('num_cities', ascending=False)


def create_visualizations(cities: List[str], vectors: np.ndarray, graph: nx.Graph, 
                         cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """
    Cria múltiplas visualizações dos clusters.
    
    Args:
        cities (List[str]): Lista de cidades
        vectors (np.ndarray): Vetores de embeddings
        graph (nx.Graph): Grafo K-NN
        cluster_dict (Dict): Dicionário de clusters
        output_dir (str): Diretório de saída
    """
    logger.info("Criando visualizações...")
    
    # Criar mapeamento de cores para clusters
    color_map = {}
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(cluster_dict))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_dict)))
    
    for cluster_id, cities_in_cluster in cluster_dict.items():
        for city in cities_in_cluster:
            color_map[city] = colors[cluster_id]
    
    # 1. Visualizações do grafo com diferentes layouts
    create_graph_visualization(graph, color_map, cluster_dict, output_dir)
    
    # 2. Visualização PCA 2D
    create_pca_visualization(cities, vectors, cluster_dict, output_dir)
    
    # 3. Visualização t-SNE 2D
    create_tsne_visualization(cities, vectors, cluster_dict, output_dir)
    
    # 4. Heatmap de similaridade por cluster
    create_similarity_heatmap(cities, vectors, cluster_dict, output_dir)


def create_graph_visualization(graph: nx.Graph, color_map: Dict, 
                              cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualizações do grafo com diferentes layouts."""
    # 1. Visualização com layout spring
    create_spring_layout_visualization(graph, cluster_dict, output_dir)
    
    # 2. Visualização com layout kamada-kawai
    create_kamada_kawai_visualization(graph, cluster_dict, output_dir)
    
    # 3. Visualização hierárquica
    create_hierarchical_visualization(graph, cluster_dict, output_dir)


def create_spring_layout_visualization(graph: nx.Graph, cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualização do grafo com layout spring."""
    # Criar mapa de cores para clusters
    color_map = {}
    for cluster_id, cities_in_cluster in cluster_dict.items():
        for city in cities_in_cluster:
            color_map[city] = cluster_id
    
    # Define as posições dos nós usando layout spring
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=RANDOM_STATE)
    
    # Cria uma lista de cores para os nós baseada no mapa de cores
    node_colors = [color_map[node] for node in graph.nodes()]
    
    plt.figure(figsize=(16, 12))
    
    # Desenha o grafo
    nx.draw(
        graph,
        pos,
        node_color=node_colors,
        with_labels=True,  # Exibe os nomes das cidades
        node_size=300,     # Tamanho dos nós
        edge_color='gray', # Cor das arestas
        alpha=0.8,         # Transparência
        font_size=8,       # Tamanho da fonte
        font_weight='bold',
        cmap=plt.cm.Set3
    )
    
    plt.title("Visualização do Grafo de Cidades com Clusters (Layout Spring)", 
              fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"grafo_spring_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualização spring layout salva em: {filepath}")


def create_kamada_kawai_visualization(graph: nx.Graph, cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualização do grafo com layout Kamada-Kawai (força dirigida)."""
    # Criar mapa de cores para clusters
    color_map = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_dict)))
    
    for cluster_id, cities_in_cluster in cluster_dict.items():
        for city in cities_in_cluster:
            color_map[city] = colors[cluster_id]
    
    # Layout Kamada-Kawai para melhor separação visual
    try:
        pos = nx.kamada_kawai_layout(graph)
    except:
        # Fallback para spring layout se kamada kawai falhar
        pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=RANDOM_STATE)
    
    plt.figure(figsize=(16, 12))
    
    # Desenhar nós por cluster para melhor controle de legenda
    for cluster_id, cities_in_cluster in cluster_dict.items():
        cluster_nodes = [city for city in cities_in_cluster if city in graph.nodes()]
        if cluster_nodes:
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=cluster_nodes,
                node_color=[colors[cluster_id]] * len(cluster_nodes),
                node_size=350,
                alpha=0.8,
                label=f'Cluster {cluster_id + 1} ({len(cluster_nodes)} cidades)'
            )
    
    # Desenhar arestas
    nx.draw_networkx_edges(graph, pos, alpha=0.4, edge_color='gray', width=1)
    
    # Adicionar labels com melhor posicionamento
    nx.draw_networkx_labels(graph, pos, font_size=7, font_weight='bold')
    
    plt.title("Rede de Similaridade de Crimes - Layout Kamada-Kawai\n(Algoritmo de Força Dirigida)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.axis('off')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"grafo_kamada_kawai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualização Kamada-Kawai salva em: {filepath}")


def create_hierarchical_visualization(graph: nx.Graph, cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualização hierárquica dos clusters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Subplot 1: Grafo com nós dimensionados por grau
    color_map = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_dict)))
    
    for cluster_id, cities_in_cluster in cluster_dict.items():
        for city in cities_in_cluster:
            color_map[city] = colors[cluster_id]
    
    pos = nx.spring_layout(graph, k=0.4, iterations=50, seed=RANDOM_STATE)
    
    # Calcular tamanhos dos nós baseado no grau (conectividade)
    degrees = dict(graph.degree())
    node_sizes = [degrees[node] * 50 + 100 for node in graph.nodes()]
    node_colors = [color_map[node] for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax1,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=6,
            font_weight='bold',
            edge_color='lightgray',
            alpha=0.7)
    
    ax1.set_title("Grafo com Nós Dimensionados por Conectividade\n(Tamanho = Número de Conexões)", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Apenas arestas entre clusters diferentes
    inter_cluster_edges = []
    city_to_cluster = {}
    for cluster_id, cities in cluster_dict.items():
        for city in cities:
            city_to_cluster[city] = cluster_id
    
    for edge in graph.edges():
        city1, city2 = edge
        if city_to_cluster.get(city1) != city_to_cluster.get(city2):
            inter_cluster_edges.append(edge)
    
    # Criar subgrafo apenas com arestas inter-cluster
    inter_graph = graph.edge_subgraph(inter_cluster_edges)
    
    if len(inter_graph.edges()) > 0:
        nx.draw(inter_graph, pos, ax=ax2,
                node_color=node_colors[:len(inter_graph.nodes())],
                node_size=300,
                with_labels=True,
                font_size=7,
                font_weight='bold',
                edge_color='red',
                width=2,
                alpha=0.8)
        
        ax2.set_title("Conexões Entre Clusters Diferentes\n(Arestas Vermelhas = Links Inter-Cluster)", 
                      fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Nenhuma conexão\nentre clusters diferentes', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Conexões Entre Clusters", fontsize=14, fontweight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"grafo_hierarquico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualização hierárquica salva em: {filepath}")


def create_pca_visualization(cities: List[str], vectors: np.ndarray, 
                           cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualização PCA 2D dos clusters."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(vectors)
    
    plt.figure(figsize=(14, 10))
    
    for cluster_id, cities_in_cluster in cluster_dict.items():
        indices = [cities.index(city) for city in cities_in_cluster if city in cities]
        if indices:
            plt.scatter(
                pca_result[indices, 0], 
                pca_result[indices, 1],
                label=f'Cluster {cluster_id + 1} ({len(indices)} cidades)',
                alpha=0.7,
                s=60
            )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} da variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} da variância)')
    plt.title('Análise PCA dos Clusters de Criminalidade\n(Redução de Dimensionalidade para 2D)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"pca_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualização PCA salva em: {filepath}")


def create_tsne_visualization(cities: List[str], vectors: np.ndarray, 
                             cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria visualização t-SNE 2D dos clusters."""
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(14, 10))
    
    for cluster_id, cities_in_cluster in cluster_dict.items():
        indices = [cities.index(city) for city in cities_in_cluster if city in cities]
        if indices:
            plt.scatter(
                tsne_result[indices, 0], 
                tsne_result[indices, 1],
                label=f'Cluster {cluster_id + 1} ({len(indices)} cidades)',
                alpha=0.7,
                s=60
            )
    
    plt.xlabel('t-SNE Componente 1')
    plt.ylabel('t-SNE Componente 2')
    plt.title('Análise t-SNE dos Clusters de Criminalidade\n(Visualização Não-Linear)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"tsne_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualização t-SNE salva em: {filepath}")


def create_similarity_heatmap(cities: List[str], vectors: np.ndarray, 
                             cluster_dict: Dict[int, List[str]], output_dir: str) -> None:
    """Cria heatmap de similaridade entre clusters."""
    similarity_matrix = cosine_similarity(vectors)
    
    # Reorganizar matriz por clusters
    city_to_cluster = {}
    for cluster_id, cities_in_cluster in cluster_dict.items():
        for city in cities_in_cluster:
            city_to_cluster[city] = cluster_id
    
    # Ordenar cidades por cluster
    sorted_cities = sorted(cities, key=lambda city: city_to_cluster.get(city, -1))
    sorted_indices = [cities.index(city) for city in sorted_cities]
    
    reordered_matrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
    
    plt.figure(figsize=(12, 10))
    
    if HAS_SEABORN:
        sns.heatmap(
            reordered_matrix,
            xticklabels=False,
            yticklabels=False,
            cmap='viridis',
            cbar_kws={'label': 'Similaridade de Cosseno'}
        )
    else:
        plt.imshow(reordered_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Similaridade de Cosseno')
        plt.xticks([])
        plt.yticks([])
    
    plt.title('Heatmap de Similaridade entre Municípios\n(Organizados por Clusters)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"heatmap_similaridade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Heatmap de similaridade salvo em: {filepath}")


def save_results(df: pd.DataFrame, cluster_dict: Dict[int, List[str]], 
                cluster_stats: pd.DataFrame, output_dir: str) -> None:
    """
    Salva os resultados da análise em arquivos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        cluster_dict (Dict): Dicionário de clusters
        cluster_stats (pd.DataFrame): Estatísticas dos clusters
        output_dir (str): Diretório de saída
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar clusters
    clusters_data = []
    for cluster_id, cities in cluster_dict.items():
        for city in cities:
            clusters_data.append({
                'cluster_id': cluster_id,
                'cluster_name': f'Cluster {cluster_id + 1}',
                'municipio': city,
                'num_crimes': len(df[df['municipio'] == city])
            })
    
    clusters_df = pd.DataFrame(clusters_data)
    clusters_path = os.path.join(output_dir, f"clusters_municipios_{timestamp}.csv")
    clusters_df.to_csv(clusters_path, index=False)
    logger.info(f"Clusters de municípios salvos em: {clusters_path}")
    
    # Salvar estatísticas dos clusters
    stats_path = os.path.join(output_dir, f"estatisticas_clusters_{timestamp}.csv")
    cluster_stats.to_csv(stats_path, index=False)
    logger.info(f"Estatísticas dos clusters salvas em: {stats_path}")
    
    # Salvar relatório detalhado
    save_detailed_report(df, cluster_dict, cluster_stats, output_dir, timestamp)


def save_detailed_report(df: pd.DataFrame, cluster_dict: Dict[int, List[str]], 
                        cluster_stats: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """Salva relatório detalhado da análise."""
    report_path = os.path.join(output_dir, f"relatorio_clusters_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE ANÁLISE DE CLUSTERS DE CRIMINALIDADE\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Número total de municípios: {len(set(df['municipio']))}\n")
        f.write(f"Número de clusters identificados: {len(cluster_dict)}\n")
        f.write(f"Parâmetro K (vizinhos): {K_NEIGHBORS}\n")
        f.write(f"Modelo de embeddings: {MODEL_NAME}\n\n")
        
        f.write("ESTATÍSTICAS DOS CLUSTERS:\n")
        f.write("-" * 30 + "\n")
        for _, row in cluster_stats.iterrows():
            f.write(f"\nCluster {row['cluster_id'] + 1}:\n")
            f.write(f"  - Número de cidades: {row['num_cities']}\n")
            f.write(f"  - Total de crimes: {row['total_crimes']}\n")
            f.write(f"  - Média de crimes por cidade: {row['avg_crimes_per_city']:.2f}\n")
            f.write(f"  - Cidades exemplo: {row['sample_cities']}\n")
        
        f.write(f"\nDETALHES DOS CLUSTERS:\n")
        f.write("-" * 25 + "\n")
        for cluster_id, cities in cluster_dict.items():
            f.write(f"\n=== CLUSTER {cluster_id + 1} ===\n")
            f.write(f"Cidades ({len(cities)}):\n")
            for i, city in enumerate(sorted(cities), 1):
                f.write(f"  {i:3d}. {city}\n")
                
            # Amostras de crimes do cluster
            f.write(f"\nAmostras de descrições de crimes (primeiras 3 cidades):\n")
            for city in sorted(cities)[:3]:
                city_crimes = df[df['municipio'] == city]['descricao'].tolist()
                f.write(f"\n{city} ({len(city_crimes)} ocorrências):\n")
                for desc in city_crimes[:3]:
                    f.write(f"  - {desc[:100]}...\n")
    
    logger.info(f"Relatório detalhado salvo em: {report_path}")


def main():
    """Executa a análise completa de clusters."""
    logger.info("Iniciando análise de clusters de criminalidade...")
    
    try:
        # 1. Carregar dados
        df = load_data(DATA_FILE)
        
        # 2. Gerar embeddings
        df_with_embeddings = generate_embeddings(df, MODEL_NAME)
        
        # 3. Calcular embeddings por cidade
        cities, vectors, city_embedding_dict = calculate_city_embeddings(df_with_embeddings)
        
        # 4. Calcular matriz de similaridade
        similarity_matrix = cosine_similarity(vectors)
        
        # 5. Construir grafo K-NN
        graph = build_knn_graph(cities, similarity_matrix, K_NEIGHBORS)
        
        # 6. Detectar comunidades
        communities, cluster_dict = detect_communities(graph)
        
        # 7. Analisar clusters
        cluster_stats = analyze_clusters(df, cluster_dict)
        
        # 8. Criar visualizações
        create_visualizations(cities, vectors, graph, cluster_dict, OUTPUT_DIR)
        
        # 9. Salvar resultados
        save_results(df, cluster_dict, cluster_stats, OUTPUT_DIR)
        
        # 10. Mostrar resumo no console
        print(f"\n{'='*60}")
        print("RESUMO DA ANÁLISE DE CLUSTERS")
        print(f"{'='*60}")
        print(f"Municípios analisados: {len(cities)}")
        print(f"Clusters identificados: {len(cluster_dict)}")
        print(f"Parâmetro K (vizinhos): {K_NEIGHBORS}")
        print(f"\nTOP 5 MAIORES CLUSTERS:")
        for _, row in cluster_stats.head().iterrows():
            print(f"  Cluster {row['cluster_id'] + 1}: {row['num_cities']} cidades, {row['total_crimes']} crimes")
        
        logger.info("Análise de clusters finalizada com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {e}")
        raise


if __name__ == "__main__":
    main()