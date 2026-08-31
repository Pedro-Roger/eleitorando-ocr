# IA OCR Pool Design

**Date:** 2026-08-31

## Goal

Trocar o fluxo atual centrado em EasyOCR por um pipeline assíncrono centrado em IA via OpenRouter, distribuindo jobs entre vários modelos disponíveis para aumentar throughput de lote e manter revisão humana obrigatória antes do salvamento final.

## Requirements

- O endpoint de entrada continua aceitando upload de imagem e retorna `jobId` imediatamente.
- Cada job representa `1 foto`.
- O processamento ocorre em background, com distribuição entre múltiplos modelos configurados.
- A resposta da IA gera apenas um rascunho editável; nenhum cadastro final é salvo automaticamente.
- O usuário pode editar o rascunho no próprio job antes da confirmação.
- O salvamento definitivo ocorre apenas em ação manual de confirmação.
- O serviço envia alerta no Discord quando houver rate limit, timeout, falha de provider, esgotamento do pool ou exaustão de retries.

## Proposed Architecture

### Job lifecycle

- `queued`: job criado e aguardando slot.
- `processing`: algum slot/modelo está processando a imagem.
- `review_required`: extração concluída, aguardando edição/validação humana.
- `confirmed`: rascunho confirmado e pronto para integração com o cadastro final.
- `error`: falha terminal após retries ou erro fatal.

### Model pool

- Configurar uma lista ordenada de modelos de IA aptos a receber jobs.
- Cada slot do pool contém estado operacional: nome do modelo, disponibilidade, cooldown, contagem de falhas e último erro.
- O dispatcher sempre entrega o próximo job ao primeiro slot elegível.
- Slots que retornarem `429`, timeout ou falhas temporárias entram em cooldown antes de voltar ao pool.
- Se todos os slots estiverem indisponíveis, o sistema preserva o job na fila e dispara alerta operacional.

### Processing contract

- A imagem é convertida para payload base64 uma vez por job.
- O prompt pede apenas o JSON esperado para título/caderneta.
- O parser normaliza a saída e grava:
- `rawResult`: JSON bruto retornado pela IA.
- `reviewDraft`: estrutura normalizada para edição humana.
- `modelUsed`: modelo vencedor do job.
- `latencyMs`: latência do processamento.
- `attemptCount`: tentativas acumuladas.

### Review and confirmation

- `PATCH /jobs/{id}/review` atualiza o rascunho revisado pelo usuário.
- `POST /jobs/{id}/confirm` valida que existe rascunho revisado e só então encaminha o salvamento final.
- O job guarda trilha mínima de auditoria:
- resultado original da IA
- rascunho revisado
- modelo usado
- timestamps

## Model candidates from the provided image

1. `Nemotron 3 Ultra (free)`
2. `MiniMax M3 (free)`
3. `Laguna S 2.1 (free)`
4. `Nemotron 3.5 Lightning (free)`
5. `Nemotron 3 Super (free)`
6. `Ling 3.0 Flash Fin (free)`
7. `MiniMax M2.7 (free)`
8. `Inkling (free)`
9. `Dots3-Note Preview (free)`
10. `North Mini Code (free)`

### Initial recommendation

- Começar com subconjunto menor e observável para evitar caos operacional no primeiro deploy.
- Ordem inicial sugerida:
1. `Nemotron 3 Ultra (free)`
2. `MiniMax M3 (free)`
3. `Nemotron 3.5 Lightning (free)`
4. `Ling 3.0 Flash Fin (free)`
5. `Laguna S 2.1 (free)`

## Error handling

- `429` ou sinal equivalente: marcar slot em cooldown e alertar Discord.
- Timeout: retry com outro slot, com alerta se exceder limiar.
- Resposta sem JSON válido: retry limitado e alerta de parsing.
- Todos os slots em cooldown/erro: manter job pendente, registrar estado agregador e alertar Discord.
- Erro terminal após retries: marcar job como `error` com mensagem curta e contexto operacional.

## Testing strategy

- Testes unitários para seleção de slot, cooldown e fallback entre modelos.
- Testes unitários para parser de resposta IA e transições de status do job.
- Testes de API para `POST /jobs`, `GET /jobs/{id}`, `PATCH /jobs/{id}/review`, `POST /jobs/{id}/confirm`.
- Testes para alerta Discord com payload sanitizado, sem vazar segredo em logs.

## Non-goals

- Não fazer paralelismo de múltiplos modelos para a mesma imagem nesta fase.
- Não integrar salvamento final do cadastro a partir do processamento automático.
- Não manter EasyOCR como caminho principal; qualquer convivência temporária será apenas para transição controlada.
