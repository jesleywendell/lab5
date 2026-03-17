# Laboratório 5 — Treinamento Fim-a-Fim do Transformer

Laboratório final da Unidade I. O objetivo é conectar a arquitetura Transformer construída no Lab 04 a um dataset real e demonstrar convergência via treinamento com backpropagation.

## Estrutura

```
.
├── lab4/
│   └── transformer.py       # Funções matemáticas puras do Lab 04 (atenção, feed-forward)
├── transformer.py           # Módulos treináveis (nn.Module) que reutilizam o lab4
├── data.py                  # Carregamento do dataset e tokenização
├── train.py                 # Training loop completo (Tarefa 3)
└── overfit_test.py          # Prova de fogo — overfitting em amostra mínima (Tarefa 4)
```

## Dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch datasets transformers
```

## Dataset

[bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — pares de frases inglês → alemão. Utilizados os primeiros 1.000 pares para treinamento.

## Tokenização

Tokenizador pré-treinado `bert-base-multilingual-cased` via `AutoTokenizer` do Hugging Face. Os tokens especiais `[CLS]` (início) e `[SEP]` (fim) são adicionados nas sequências do decoder. Sequências são padadas com zeros para comprimento uniforme no batch.

## Arquitetura

Reutiliza diretamente as funções `scaled_dot_product_attention` e `feed_forward` do `lab4/transformer.py` dentro dos módulos `nn.Module`. Os parâmetros treináveis (W_Q, W_K, W_V, W_O, projeções FFN, embeddings) são gerenciados pelo PyTorch via `nn.Linear` e `nn.Embedding`.

Hiperparâmetros usados no treinamento:

| Parâmetro | Valor |
|-----------|-------|
| d_model | 128 |
| Cabeças de atenção | 4 |
| Camadas encoder/decoder | 2 |
| d_ff | 512 |
| Épocas | 15 |
| Batch size | 32 |
| Learning rate | 1e-3 |

## Treinamento

```bash
python3 train.py
```

Evolução da loss ao longo das épocas:

```
Epoch 01/15 | Loss: 8.8350
Epoch 05/15 | Loss: 3.9493
Epoch 10/15 | Loss: 1.9130
Epoch 15/15 | Loss: 0.4669
```

Otimizador: Adam. Função de perda: CrossEntropyLoss com `ignore_index=0` (padding).

## Prova de Fogo (Overfitting Test)

```bash
python3 overfit_test.py
```

O modelo é treinado por 80 épocas em 8 amostras para verificar que os gradientes fluem corretamente. Resultado esperado:

```
Source    : Two young, White males are outside near many bushes.
Expected  : Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
Generated : Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
```

## Ferramentas de IA

Conforme exigido pelo enunciado: IA generativa foi utilizada como auxílio na implementação das Tarefas 1 e 2 (carregamento do dataset e tokenização). O fluxo de Forward/Backward da Tarefa 3 interage estritamente com as classes construídas nos laboratórios anteriores.
