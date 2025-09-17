from collections import deque
from dataclasses import dataclass
from math import inf
import sys
from typing import List, Tuple

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
    predecessor: List[int] # predecessor na melhor rota
    caminho: List[int] # caminho origem->destino (vazio se não há)
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
        raise ValueError("O grafo tem ciclo; longest caminho em geral vira NP-difícil.")

    return ordem_topologica


"""
Função de calculo do caminho mais longo de origem até destino em um DAG.
"""
def longest_path_dag(num_vertices: int, matriz: List[List[int]], origem: int, destino: int) -> LongestPathResult:
    ordem_topologica = topological_order(num_vertices, matriz)

    distancias = [-inf] * num_vertices
    predecessor = [-1] * num_vertices
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

    # reconstrói origem->destino se existir
    caminho: List[int] = []
    peso_total: float | None

    if distancias[destino] == -inf:
        peso_total = None
    else:
        peso_total = distancias[destino]
        cur = destino
        while cur != -1:
            caminho.append(cur)
            cur = predecessor[cur]
        caminho.reverse()

    return LongestPathResult(
        num_vertices=num_vertices,
        matriz=matriz,
        origem=origem,
        destino=destino,
        ordem_topologica=ordem_topologica,
        distancias=distancias,
        predecessor=predecessor,
        caminho=caminho,
        peso_total=peso_total,
    )


"""
Função orquestradora escolhe o arquivo de entrada,
chama as funções de leitura e de cálculo do caminho mais longo,
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

    print("=== Longest caminho em DAG (ordem topológica + DP) ===")
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
    print("\nPredecessores (predecessor):")
    print(" ".join(map(str, res.predecessor)))

    if res.peso_total is None:
        print(f"\nNão existe caminho de {res.origem} até {res.destino}.")
    else:
        print(f"\nCaminho s → t: {' '.join(map(str, res.caminho))}")
        print(f"Peso total: {res.peso_total}")


if __name__ == "__main__":
    main()