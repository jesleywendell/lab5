# Laboratório 4 — Transformer Completo From Scratch

Integração dos módulos desenvolvidos nos laboratórios anteriores em uma arquitetura Encoder-Decoder completa, com inferência auto-regressiva.

## Estrutura

```
lab4/
├── transformer.py
├── README.md 
```

## Tarefas implementadas

### Blocos base

Reaproveitando as implementações dos labs anteriores via `sys.path`, e reescrevendo em PyTorch as funções que estavam em NumPy:

- `scaled_dot_product_attention(Q, K, V, mask)` — atenção escalonada com suporte a máscara opcional
- `feed_forward(x, W1, b1, W2, b2)` — FFN com expansão de dimensão e ativação ReLU
- `add_and_norm(x, sublayer_out)` — conexão residual com Layer Normalization: `LayerNorm(x + Sublayer(x))`

### Encoder Block

`encoder_block(x, params)` recebe o tensor de entrada `X` (já com Positional Encoding somado) e executa:

1. Self-Attention: Q, K, V projetados a partir de X
2. Add & Norm
3. FFN
4. Add & Norm

Empilhável: o resultado Z de um bloco alimenta o próximo, produzindo a memória contextualizada do Encoder.

### Decoder Block

`decoder_block(y, Z, params)` recebe o tensor alvo `Y` e a memória `Z` do Encoder e executa:

1. Masked Self-Attention com máscara causal (de `lab3/task1_causal_mask.py`) — impede acesso a tokens futuros
2. Add & Norm
3. Cross-Attention (de `lab3/task2_cross_attention.py`) — Q vem da saída anterior, K e V vêm de Z
4. Add & Norm
5. FFN + Add & Norm
6. Projeção linear para o vocabulário + Softmax

### Inferência auto-regressiva

Loop `while` integrado ao modelo completo:

1. Encoder processa `encoder_input` (tensor fictício simulando "Thinking Machines") e produz `Z`
2. Decoder inicia com `<START>`
3. A cada passo, o Decoder recebe a sequência gerada até o momento e prevê o próximo token pelo `argmax` das probabilidades
4. O token é concatenado à sequência e o ciclo recomeça até a geração de `<EOS>`

## Como executar

```bash
python3 transformer.py
```

PyTorch deve estar instalado.

## Nota de uso de IA

Partes geradas/complementadas com IA, revisadas por Jesley Wendell Leite Soares. O uso de IA generativa se deu exclusivamente como ferramenta auxiliar no desenvolvimento (template), revisão do código e ajudante na escrita do README (template também), com o resto tendo sido feito inteiramente por mim.