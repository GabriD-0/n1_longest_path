from collections import deque
from dataclasses import dataclass
from math import inf
import sys
from typing import List, Tuple
from functools import lru_cache

"""
Empacotamento de campos para criação de um construtor automatico
assim agilizando e facilitando na geração de logs e debug
"""
@dataclass
class LongestPathResult:
    num_vertices: int
    matriz: List[List[int]] # matriz de adjacência, onde 0 significa “sem aresta”
    origem: int
    destino: int
    ordem_topologica: List[int]
    distancias: List[float] # melhor distância a partir de origem
    predecessor: List[int] # predecessor em UMA melhor rota
    predecessores: List[List[int]] # todos os predecessores que empatam no melhor custo
    caminho_otimo: List[int] # um caminho_otimo origem->destino
    caminhos_otimos: List[List[int]] # todos os caminhos ótimos (empates)
    peso_total: float | None # peso total origem->destino


"""
Função para ler o arquivo
"""
def read_file(filename: str = "entrada.txt") -> Tuple[int, List[List[int]], int, int]:
    arquivos: List[str] = []
    with open(filename, "r", encoding="utf-8") as file:
        for raw in file:
            # remove comentários em linha
            line = raw.split("#", 1)[0].strip()
            if line:
                arquivos.append(line)

    if not arquivos:
        raise ValueError("Arquivo de entrada vazio após remoção de comentários.")

    iterador = iter(arquivos)
    num_vertices = int(next(iterador))

    # lê exatamente num_vertices linhas e formar a matriz
    matriz: List[List[int]] = []
    for _ in range(num_vertices):
        row = list(map(int, next(iterador).split()))
        if len(row) != num_vertices:
            raise ValueError("Linha da matriz não tem num_vertices colunas.")
        matriz.append(row)

    origem, destino = map(int, next(iterador).split())

    if not (0 <= origem < num_vertices and 0 <= destino < num_vertices):
        raise ValueError("Índices de origem/destino fora do intervalo [0, num_vertices).")

    return num_vertices, matriz, origem, destino


"""
Calcula uma ordem topológica dos vértices usando o algoritmo de Kahn, lendo direto da matriz de adjacência (onde 0 significa “não existe aresta”).
"""
def topological_order(num_vertices: int, matriz: List[List[int]]) -> List[int]:
    grau_entrada = [0] * num_vertices

    for orig in range(num_vertices):
        for dest, peso in enumerate(matriz[orig]):
            if peso != 0:
                grau_entrada[dest] += 1

    fila = deque([i for i in range(num_vertices) if grau_entrada[i] == 0])
    ordem_topologica: List[int] = []

    while fila:
        orig = fila.popleft()
        ordem_topologica.append(orig)
        for dest, peso in enumerate(matriz[orig]):
            if peso != 0:
                grau_entrada[dest] -= 1
                if grau_entrada[dest] == 0:
                    fila.append(dest)

    if len(ordem_topologica) != num_vertices:
        raise ValueError("O grafo tem ciclo; longest caminho_otimo em geral vira NP-difícil.")

    return ordem_topologica


"""
Função de calculo do caminho_otimo mais longo de origem até destino em um DAG.
"""
def longest_path_dag(num_vertices: int, matriz: List[List[int]], origem: int, destino: int) -> LongestPathResult:
    ordem_topologica = topological_order(num_vertices, matriz)

    distancias = [-inf] * num_vertices
    predecessor = [-1] * num_vertices
    predecessores: List[List[int]] = [[] for _ in range(num_vertices)]  # (empates)
    distancias[origem] = 0

    for orig in ordem_topologica:
        if distancias[orig] == -inf:
            continue

        for dest, peso in enumerate(matriz[orig]):
            if peso != 0:
                cand = distancias[orig] + peso
                if cand > distancias[dest]:
                    distancias[dest] = cand
                    predecessor[dest] = orig
                    predecessores[dest] = [orig]
                elif cand == distancias[dest]:
                    predecessores[dest].append(orig)
                    if predecessor[dest] == -1:
                        predecessor[dest] = orig

    # Reconstrói UM caminho_otimo pela cadeia 'predecessor'
    caminho_otimo: List[int] = []
    caminhos_otimos: List[List[int]] = []
    peso_total: float | None

    if distancias[destino] == -inf:
        peso_total = None
    else:
        peso_total = distancias[destino]
        cur = destino
        while cur != -1:
            caminho_otimo.append(cur)
            cur = predecessor[cur]
        caminho_otimo.reverse()

        # todos os caminhos ótimos usando o grafo de predecessores (empates)
        sys.setrecursionlimit(max(1000, num_vertices * 10))

        @lru_cache(None)
        def build(u: int) -> List[List[int]]:
            if u == origem:
                return [[origem]]
            paths: List[List[int]] = []
            for p in predecessores[u]:
                for pp in build(p):
                    paths.append(pp + [u])
            return paths

        caminhos_otimos = build(destino)

    return LongestPathResult(
        num_vertices=num_vertices,
        matriz=matriz,
        origem=origem,
        destino=destino,
        ordem_topologica=ordem_topologica,
        distancias=distancias,
        predecessor=predecessor,
        predecessores=predecessores,
        caminho_otimo=caminho_otimo,
        caminhos_otimos=caminhos_otimos,
        peso_total=peso_total,
    )


"""
Função orquestradora escolhe o arquivo de entrada,
chama as funções de leitura e de cálculo do caminho_otimo mais longo,
trata erros, e imprime um relatório completo e legível do resultado.
"""
def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "entrada.txt"

    try:
        num_vertices, matriz, origem, destino = read_file(filename)
        res = longest_path_dag(num_vertices, matriz, origem, destino)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Longest caminho_otimo em DAG (ordem topológica + DP) ===")
    print(f"Arquivo: {filename}")
    print(f"Vértices: {res.num_vertices}")
    print("\nMatriz de adjacência (0 = sem aresta):")

    for row in res.matriz:
        print(" ".join(map(str, row)))

    print(f"\nOrigem: {res.origem}")
    print(f"Destino: {res.destino}")
    print("\nOrdem topológica:")
    print(" ".join(map(str, res.ordem_topologica)))
    print("\nDistâncias máximas a partir de origem (−inf = inalcançável):")

    def fmt(x): return "-inf" if x == -inf else str(x)

    print(" ".join(fmt(x) for x in res.distancias))
    print("\nPredecessores:")
    print(" ".join(map(str, res.predecessor)))

    if res.peso_total is None:
        print(f"\nNão existe caminho_otimo de {res.origem} até {res.destino}.")
    else:
        print(f"\nSolução otima (um dos empates): {' '.join(map(str, res.caminho_otimo))}")
        print(f"Peso total máximo: {res.peso_total}")
        print("\nTodos os caminhos ótimos:")

        for p in res.caminhos_otimos:
            print(" -> ".join(map(str, p)))

if __name__ == "__main__":
    main()