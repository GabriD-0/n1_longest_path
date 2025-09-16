from collections import deque
from dataclasses import dataclass
from math import inf
import sys
from typing import List, Tuple

'''
Empacotamento de campos para criação de um construtor automatico
assim agilizando e facilitando na geração de logs e debug
'''
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


def read_input_file(filename: str = "entrada.txt") -> Tuple[int, List[List[int]], int, int]:
    """
    Lê o arquivo, ignorando linhas vazias e comentários (#).
    Aceita comentários em linha (após '#').
    """
    cleaned: List[str] = []
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            # remove comentários em linha
            line = raw.split("#", 1)[0].strip()
            if line:
                cleaned.append(line)

    if not cleaned:
        raise ValueError("Arquivo de entrada vazio após remoção de comentários.")

    it = iter(cleaned)
    num_vertices = int(next(it))

    matriz: List[List[int]] = []
    for _ in range(num_vertices):
        row = list(map(int, next(it).split()))
        if len(row) != num_vertices:
            raise ValueError("Linha da matriz não tem num_vertices colunas.")
        matriz.append(row)

    origem, destino = map(int, next(it).split())
    if not (0 <= origem < num_vertices and 0 <= destino < num_vertices):
        raise ValueError("Índices de origem/destino fora do intervalo [0, num_vertices).")

    return num_vertices, matriz, origem, destino


def topo_order_from_matrix(num_vertices: int, mat: List[List[int]]) -> List[int]:
    """Kahn direto na matriz (0 = sem aresta). Lança ValueError se houver ciclo."""
    indeg = [0] * num_vertices
    for u in range(num_vertices):
        for v, w in enumerate(mat[u]):
            if w != 0:
                indeg[v] += 1

    q = deque([i for i in range(num_vertices) if indeg[i] == 0])
    order: List[int] = []

    while q:
        u = q.popleft()
        order.append(u)
        for v, w in enumerate(mat[u]):
            if w != 0:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    if len(order) != num_vertices:
        raise ValueError("O grafo tem ciclo; longest path em geral vira NP-difícil.")

    return order


def longest_path_dag_matrix(num_vertices: int, mat: List[List[int]], origem: int, destino: int) -> LongestPathResult:
    """DP em ordem topológica direto na matriz (varre linhas inteiras, O(n²))."""
    order = topo_order_from_matrix(num_vertices, mat)

    distancias = [-inf] * num_vertices
    predecessor = [-1] * num_vertices
    distancias[origem] = 0

    for u in order:
        if distancias[u] == -inf:
            continue
        for v, w in enumerate(mat[u]):
            if w != 0:
                cand = distancias[u] + w
                if cand > distancias[v]:
                    distancias[v] = cand
                    predecessor[v] = u

    # reconstrói origem->destino se existir
    path: List[int] = []
    value: float | None
    if distancias[destino] == -inf:
        value = None
    else:
        value = distancias[destino]
        cur = destino
        while cur != -1:
            path.append(cur)
            cur = predecessor[cur]
        path.reverse()

    return LongestPathResult(
        num_vertices=num_vertices,
        matriz=mat,
        origem=origem,
        destino=destino,
        ordem_topologica=order,
        distancias=distancias,
        predecessor=predecessor,
        caminho=path,
        peso_total=value,
    )


def main():
    # permite passar o arquivo como argumento; senão, usa 'entrada.txt'
    filename = sys.argv[1] if len(sys.argv) > 1 else "entrada.txt"
    try:
        num_vertices, mat, origem, destino = read_input_file(filename)
        res = longest_path_dag_matrix(num_vertices, mat, origem, destino)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    # ===== Saída completa e amigável =====
    print("=== Longest Path em DAG (ordem topológica + DP) ===")
    print(f"Arquivo: {filename}")
    print(f"Vértices (num_vertices): {res.num_vertices}")
    print("Matriz de adjacência (0 = sem aresta):")
    for row in res.matriz:
        print(" ".join(map(str, row)))
    print(f"Origem (origem): {res.origem}")
    print(f"Destino (destino): {res.destino}")
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
        print(f"\nCaminho s→t: {' '.join(map(str, res.caminho))}")
        print(f"Peso total (s→t): {res.peso_total}")


if __name__ == "__main__":
    main()