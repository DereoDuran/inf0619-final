# Avaliação de risco em seguros de vida

Projeto final da disciplina INF0619 (Mineração de Dados Complexos, Unicamp, 2026). Grupo 3, Inteligência Natural.

Classificação de propostas de seguro de vida em 8 classes de risco (base Prudential Life Insurance Assessment, versão da disciplina no Kaggle), com **balanced accuracy** como métrica obrigatória.

## Resultado

| Modelo | CV (5 folds) | Kaggle (público) |
|---|---|---|
| Acaso | 0,125 | |
| RF balanced (baseline) | 0,5134 | |
| HGB balanced, hiperparâmetros ajustados + soma de keywords (E08) | 0,5345 | 0,53447 |
| **Stack ordinal (E15-r01, oficial)** | **0,5402** | **0,54377** |

O modelo final é uma regressão logística sobre 7 HGB binários `Response > i` e um HGB balanceado com média de 5 seeds. Do E00 ao E13, nenhuma variação de modelo único passou do platô de ~0,535; o ganho veio de tratar o alvo como ordinal.

## Apresentação

Apresentação do vídeo final em three.js: **https://dereoduran.github.io/inf0619-final/**

Cada uma das 47.487 propostas do treino é uma partícula, e as cenas são montadas com os resultados reais do projeto (OOF do modelo final, `runs.csv`, importâncias, submissões). Teclas: → avança, ← volta, N mostra as notas, F tela cheia. O arquivo publicado é `docs/index.html`, autocontido e offline.

## Estrutura

```
01_eda.ipynb        enquadrar o problema, carregar e explorar os dados, gerar os folds
02_modeling.ipynb   protocolo de experimentos, E00 a E15, submissões
03_analysis.ipynb   análise de erro, métricas, explicabilidade, modelo enxuto
utils.py            caminhos, seed, grupos de colunas, prepara(), funções comuns
dados/              train.csv, test.csv, sample_submission.csv, folds.csv
resultados/         runs.csv (registro dos runs), oof/, teste/, modelos/, buscas e análises
submissoes/         CSVs enviados ao Kaggle e registro.csv
docs/               apresentação publicada no GitHub Pages
```

## Protocolo

- 5 folds estratificados, fixos em `dados/folds.csv`; hashes dos dados e dos folds conferidos em toda carga.
- Cada run grava uma linha em `resultados/runs.csv` e as probabilidades out-of-fold em `resultados/oof/`.
- Um run substitui o controle se ganhar pelo menos 0,005 na média e for melhor em pelo menos 4 dos 5 folds.
- O Kaggle é o teste: o modelo oficial é o escolhido pela validação cruzada.

## Reproduzir

```bash
uv sync
uv run jupyter lab   # Run All no 01, no 02 e no 03, nessa ordem
```

Com `resultados/` preenchido, os runs são lidos do cache em vez de treinados de novo. Python 3.12, scikit-learn 1.9.1, pandas 3.1.1, LightGBM e CatBoost.

## Uso de IA generativa

O Claude (Anthropic) foi usado no planejamento, no código e no texto dos notebooks (escritos no chat, célula por célula, executados e revisados pelo autor), na análise dos resultados e na apresentação do vídeo final. As decisões de protocolo, de modelo e do que entra no relatório foram do grupo. O registro por data está no relatório final.
