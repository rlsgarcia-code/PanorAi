# P74 — Estimação de pose relativa versus sobreposição espacial

## Escopo: somente estimação de `R` e direção de `t`

**Data do corte:** 8 de outubro de 2026  
**Fonte executável exata:** commit `0002f36` sobre `origin/main` `a00b74f`  
**Tipo da evidência:** benchmark de desenvolvimento executado a partir do
checkout fonte; não é evidência de wheel instalado nem qualificação de release.

Este documento consolida exclusivamente a estimação de pose relativa entre
dois panoramas RGB. Nenhuma das execuções aqui relatadas usa:

- estimação de range/depth por dense stereo;
- filtragem de matches por range denso;
- refinamento fotométrico denso;
- otimização conjunta ou alternada de `R`, `t` e range;
- evidência multiview.

As nuvens de pontos da P74 são usadas **somente como oráculo externo** para
medir a interseção espacial das cenas e selecionar pares. As transformações
registradas são usadas **somente como ground truth** para medir o erro da pose.
Nem as nuvens nem as poses de referência são entradas do estimador RGB.

## Conclusão executiva

O mínimo de sobreposição recomendado hoje é:

- **70% como limiar operacional provisório**, calculado como o menor overlap
  direcional entre as duas nuvens, e sempre acompanhado de um gate de qualidade
  que falhe de forma fechada;
- **nenhum percentual pode ainda ser declarado como mínimo confiável para
  release**.

O motivo é estatístico e operacional. Na faixa observada de 70,85% a 79,48%,
os dois frontends acertaram 8 de 9 poses no critério estrito: 88,9%, com
intervalo Wilson de 95% entre 56,5% e 98,0%. O ponto estimado é bom, mas o limite
inferior ainda é incompatível com uma promessa de uso confiável. Entre 50% e
70%, apenas 6 de 9 poses foram estritamente corretas: 66,7%.

O frontend DoG diretamente esférico é o candidato preferido **se o gate de
qualidade for obrigatório**. Ele melhorou o erro mediano dos 14 pares que ambos
os métodos resolveram e produziu 13/18 poses aceitas e precisas, contra 12/18
para o coarse-to-fine. Sem o gate, entretanto, ele retornou três poses erradas
graves; portanto “pose retornada” não pode ser tratada como “pose válida”.

## Definições geométricas

A pose relativa leva um ponto expresso na câmera `A` para a câmera `B`:

```text
x_B = R_BA @ x_A + t_BA
```

O estimador monocular observa a rotação `R_BA` e a **direção orientada** de
`t_BA`. A escala métrica de `t` não é observável e não faz parte desta
avaliação.

Os erros são:

- erro de rotação: distância geodésica em `SO(3)`, em graus;
- erro de translação: ângulo entre a direção estimada e a direção orientada de
  referência, em graus.

Os níveis de sucesso congelados são:

| Nível | Erro máximo de `R` | Erro máximo da direção de `t` |
| --- | ---: | ---: |
| amplo | 15° | 30° |
| estrito | 5° | 10° |
| preciso | 2° | 5° |

“Pose retornada” significa apenas que o estimador produziu uma matriz e uma
direção. A recomendação usa o nível estrito como medida de recuperação e trata
qualquer retorno fora do nível amplo como pose errada.

## Como a sobreposição espacial foi medida

Cada nuvem organizada é colocada no sistema comum da sua cena por:

```text
x_scene = R_scan @ x_scan_local + t_scan
```

Para cada direção, mede-se a fração de pontos de uma nuvem que possui uma
superfície vizinha na outra nuvem a até 0,25 m. O número usado no eixo de
dificuldade é:

```text
overlap(A, B) = min(fração A→B, fração B→A)
```

Usar o mínimo impede que uma nuvem pequena totalmente contida em uma nuvem
maior mascare a baixa cobertura no sentido inverso. A validade exige XYZ finito
e range radial maior que `1e-3 m`. O censo considerou 7.058 pares intrafamília
com centros a até 15 m e encontrou 265 pares elegíveis com pelo menos 50% de
overlap em ambas as direções.

O conjunto de resposta contém 45 pares congelados antes das predições: três
pares por família P74 em cada uma das cinco faixas. Cada faixa contém nove
pares. O RGB não foi lido durante a seleção.

Esse overlap depende das nuvens e do registro P74. Ele é um oráculo de
elegibilidade do benchmark, não uma observação que o pipeline RGB puro possua
antes de estimar a pose.

## Algoritmos comparados

### A. Coarse-to-fine esférico

Este perfil foi executado nas cinco faixas de overlap:

- propostas em `512×1024` após convolução Gaussian esférica nativa;
- sigma 1,0 no raster intermediário com altura 1024;
- orçamento de candidatos 2× e até 4.096 keypoints finais;
- verificação/refino tangent-DoG na resolução fonte;
- patches tangentes RootSIFT `r6/n48/d1.5`, orientação fixa e normalização
  local;
- FLANN com Lowe 0,72 e deduplicação angular em 0,15°;
- estimador completo espacial/MSAC, 1.000 trials, refit e diagnóstico de
  estabilidade.

### B. DoG direto com convolução esférica

Este perfil foi executado nas duas faixas com overlap de pelo menos 50%:

- DoG sobre o panorama esférico completo, com backend nativo;
- três oitavas, três níveis por oitava, sigma-base 1,6 px;
- limiar de contraste 0,012 e limiar de borda 10;
- localização subpixel por Taylor de segunda ordem no plano tangente;
- até 4.096 keypoints com seleção equal-area;
- o mesmo RootSIFT, matcher e estimador do perfil A.

O perfil B não instancia `SphericalCoarseDoGDetector` e não usa raster de
propostas reduzido.

## Resposta completa do perfil coarse-to-fine

Os erros medianos abaixo consideram todas as poses retornadas na faixa. A
coluna “erradas” é indispensável, pois a mediana pode permanecer pequena mesmo
na presença de uma falha catastrófica.

| Overlap mínimo | Faixa observada | Poses retornadas | Estritas | Precisas | Erradas | Matches medianos | Erro mediano `R` | Erro mediano `t` | Tempo mediano/par |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <10% | 0,00–8,13% | 0/9 | 0/9 | 0/9 | 0 | 7 | — | — | 56,04 s |
| 10–25% | 12,60–24,90% | 2/9 | 0/9 | 0/9 | 1 | 14 | 4,46° | 47,94° | 57,21 s |
| 25–50% | 29,03–48,32% | 6/9 | 2/9 | 2/9 | 4 | 25 | 13,84° | 48,35° | 60,11 s |
| 50–70% | 52,33–66,45% | 7/9 | 6/9 | 6/9 | 1 | 26 | 0,19° | 0,43° | 60,56 s |
| ≥70% | 70,85–79,48% | 8/9 | 8/9 | 8/9 | 0 | 49 | 0,31° | 0,32° | 128,74 s |

Abaixo de 50%, somente 2/27 pares foram estritos. Com pelo menos 50%, 14/18
foram estritos. O teste exato de Fisher unilateral para essa separação resultou
em `odds ratio = 43,75` e `p = 1,695e-6`. O overlap é, portanto, um forte
indicador de dificuldade, mas não é um certificado de correção.

## Comparação pareada acima de 50%

| Overlap | Perfil | Retornadas | Estritas | Precisas | Erradas | Aceitas pelo diagnóstico | Matches medianos | Tempo mediano/par |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50–70% | coarse-to-fine | 7/9 | 6/9 | 6/9 | 1 | 4/9 | 26 | 60,56 s |
| 50–70% | DoG esférico direto | 8/9 | 6/9 | 5/9 | 2 | 5/9 | 45 | 60,63 s |
| ≥70% | coarse-to-fine | 8/9 | 8/9 | 8/9 | 0 | 8/9 | 49 | 128,74 s |
| ≥70% | DoG esférico direto | 9/9 | 8/9 | 8/9 | 1 | 8/9 | 66 | 59,42 s |
| **≥50% agregado** | **coarse-to-fine** | **15/18** | **14/18** | **14/18** | **1** | **12/18** | **43** | **77,03 s** |
| **≥50% agregado** | **DoG esférico direto** | **17/18** | **14/18** | **13/18** | **3** | **13/18** | **56** | **60,21 s** |

Os dois algoritmos resolveram exatamente os mesmos 14 pares no critério
estrito. Entre esses pares:

- o DoG direto reduziu o erro de `R` em 11/14;
- reduziu o erro da direção de `t` em 9/14;
- teve erro mediano estrito de `R=0,130°` e `t=0,213°`;
- o coarse-to-fine teve `R=0,225°` e `t=0,384°`.

O ganho é de precisão local, não de recuperação geométrica. O DoG direto
retornou duas poses adicionais, mas ambas eram erradas.

### Falhas perigosas do DoG direto

| Par | Resultado coarse | Resultado DoG direto |
| --- | --- | --- |
| M-070→M-071 | sem pose | `R=78,51°`, `t=123,84°` |
| W103→W122 | `R=112,77°`, `t=57,97°` | `R=143,71°`, `t=72,04°` |
| M-071→M-075 | sem pose | `R=14,73°`, `t=138,70°` |

O diagnóstico de qualidade rejeitou os três resultados diretos errados. No
agregado ≥50%, ele selecionou 13/18 poses diretas, todas precisas, contra 12/18
poses coarse, também todas precisas. Porém, o avaliador atual serializou o
perfil de forma aninhada e registrou `quality_gate_enabled=false`. Portanto,
esses números demonstram a capacidade discriminativa do diagnóstico, mas não
provam ainda um caminho de produção que impeça o retorno da pose rejeitada.

## Tempo de processamento

Ambos os perfis processam panoramas canônicos de `2048×4096`. A execução do
DoG direto usou um worker e apresentou a seguinte decomposição mediana em 18
pares:

| Etapa | Tempo mediano |
| --- | ---: |
| carregar e adaptar as duas imagens | 5,63 s/par |
| detectar/descrever panorama A | 25,85 s |
| detectar/descrever panorama B | 26,08 s |
| matching | 0,048 s |
| estimar pose | 2,28 s |
| pipeline completo | 60,21 s/par |

Nesta máquina (`Darwin 25.6 arm64`, Apple M3 Max, Python 3.12.4), o custo atual
de extração direta é aproximadamente 26 s por panorama. O valor histórico de
3–5 s por panorama não foi reproduzido pelo perfil direto atual.

Os tempos coarse da tabela foram coletados em execuções com paralelismo e
apresentaram forte variação por par, especialmente na faixa ≥70%. Eles devem
ser usados para dimensionamento aproximado, não para concluir que o DoG direto
é mais rápido. Uma comparação de release exige warm-up, repetições, mediana,
P95, memória e o mesmo número de workers.

## Precisão estatística da curva

| Faixa | Sucesso estrito | Taxa | Intervalo Wilson 95% |
| --- | ---: | ---: | ---: |
| <10% | 0/9 | 0,0% | 0,0–29,9% |
| 10–25% | 0/9 | 0,0% | 0,0–29,9% |
| 25–50% | 2/9 | 22,2% | 6,3–54,7% |
| 50–70% | 6/9 | 66,7% | 35,4–87,9% |
| ≥70% | 8/9 | 88,9% | 56,5–98,0% |
| ≥50% agregado | 14/18 | 77,8% | 54,8–91,0% |

Os nove pares por faixa não são suficientes para transformar 88,9% em uma
garantia. Além disso, pares da mesma família de captura não são observações
totalmente independentes, o que reduz o tamanho efetivo da amostra.

## Recomendação de uso e de release

### Regra operacional provisória

1. **Rejeitar `<50%`** para estimação de pose destinada à reconstrução.
2. Tratar **50–70% como faixa experimental/de stress**, não como operação
   confiável.
3. Usar **≥70% apenas como pré-condição operacional provisória**.
4. No perfil DoG direto, tornar o gate de qualidade obrigatório e não expor
   poses rejeitadas ao chamador como sucesso.
5. Se o overlap verdadeiro não estiver disponível no uso RGB, substituí-lo por
   uma política conservadora de pares adjacentes/recuperação de imagem e deixar
   claro que ela não oferece a mesma garantia geométrica.

### Decisão de release neste corte

**NO-GO para declarar um percentual mínimo “confiável para uso”.** Nenhuma
faixa testada atingiu simultaneamente recuperação estrita próxima de 95%,
intervalo de confiança estreito e ausência demonstrada de falso positivo num
conjunto held-out independente.

O valor de 70% deve ser publicado, no máximo, como **limiar experimental de
elegibilidade**, nunca como garantia de pose correta.

### Evidência mínima antes de promoção

Antes de promover o algoritmo, recomenda-se congelar o frontend e o gate e
executar um novo conjunto não usado no desenvolvimento, separado por sequência
ou local de captura. O gate de promoção deve exigir:

- zero pose catastrófica aceita;
- taxa estrita pontual de pelo menos 95%;
- limite inferior Wilson de 95% de pelo menos 90%;
- tempo mediano, P95 e memória com execução reproduzível;
- análise separada das faixas 70–80%, 80–90% e ≥90%;
- nenhum ajuste de parâmetros depois de abrir o held-out.

Mesmo com zero falhas observadas, são necessários pelo menos 35 pares
independentes para que o limite inferior Wilson de 95% ultrapasse 90%, e 73
pares para ultrapassar 95%. Como pares de uma mesma sequência são
correlacionados, o estudo deve incluir várias sequências e locais, não apenas
mais arestas do mesmo grafo P74.

## Reprodutibilidade e proveniência

- censo de elegibilidade: 265 pares ≥50% entre 7.058 candidatos;
- amostra congelada SHA-256:
  `299bb6f38a26242eee33082cc2dec3ed05ed7467e3d9dc8014b75f64a29d6b36`;
- resposta coarse JSON SHA-256:
  `3da9eb78873c5f329a415b806c5f4661be9d70e74ddac7153f8410526117497c`;
- células avaliadas do DoG direto ≥70% SHA-256:
  `c47b6ea4cd112806d8406c41a99e96916a3a76b7f471fc04ba14f3a999050e3a`;
- células avaliadas do DoG direto 50–70% SHA-256:
  `80124214da93b0f6ed2133b3749dd955f8a04e9989895b65c19a2f3a5f8bc9b3`.

Os arquivos P74 de RGB, nuvem e pose permanecem externos e não são
distribuídos com PanorAi. Este relatório registra resultados do checkout fonte
e não altera o estado de release da biblioteca.
