# Entropia Cruzada vs. Erro Médio Quadrático

Este material é um estudo dirigido que apresenta a teoria por trás do uso da função _softmax_ e da entropia cruzada como função de custo, quando projetamos redes neurais para lidar com problemas de classificação, com saída desejada no formato _one-hot_.

Esta teoria acompanha o [exercício F06](https://docs.google.com/forms/d/e/1FAIpQLSc1_kUBIi0Y1uBwC_p9YA7pcAHB6q9Z_QlpaYzQ6O4Y4iDLQQ/viewform?usp=sf_link).

## MSE

A função de custo mais conhecida é o Erro Médio Quadrático (MSE - Mean Squared Error). Esta é uma função simples de entender e derivar, e por isso é a que frequentemente aprendemos primeiro. Ela é dada pela fórmula:

$E_{MSE} = \frac{1}{2MN} \sum^M_{k=1} \sum^N_{i=1} (y_{ik} - \hat{y}_{ik})^2$

Que em palavras, significa que subtraímos o valor desejado do valor observado na saída, elevamos ao quadrado, e fazemos isso para cada saída $i$ e amostra $k$, somando e normalizando o resultado.

Essencialmente, se olharmos para uma única amostra $k$ específica temos:
$E_{MSE_k} = \frac{1}{2N} \sum^N_{i=1} (y_{i} - \hat{y}_{i})^2$

## CE

Entretanto para problemas de classificação, a função mais comumente empregada é a Entropia Cruzada (CE - Cross Entropy), que é definida como:

$E_{CE} = - \frac{1}{M} \sum_{k=1}^{M} \sum_{i=1}^{N} \hat{y}_{ik} \log(y_{ik})$

Olhando para uma amostra $k$ específica temos:

$E_{CE_k} = - \sum_{i=1}^{N} \hat{y}_{i} \log(y_{i})$

Este Jupyter Notebook é um estudo dirigido sobre o tema do uso de codificação _one-hot_, função _softmax_ e entropia cruzada, para problemas de classificação em redes neurais artificiais.

Nessas funções da CE acima, note que diferentemente do MSE, na CE não podemos trocar simetricamente $y$ por $\hat{y}$ na equação. Como $\hat{y}$ geralmente é um vetor no formato _one-hot_, isso causaria $\log 0$ que não tem valor definido (tende a infinito, conforme se aproxima de zero).

Neste Jupyter Notebook vamos examinar mais de perto a diferença entre essas duas funções de perda, quando empregadas em um problema de classificação. Abaixo ignoraremos o índice da amostra $k$ para simplificar a notação.

## One-Hot Encoding

Aqui vale lembrarmos que num problema de classificação entre $N$ possíveis classes, temos $N$ saídas, cada uma representando uma classe. As etiquetas (saídas desejadas) $\hat{y}$ são codificadas no formato _one-hot_, ou seja, são vetores de $N$ componentes, onde apenas a componente referente à classe desejada é $1$, sendo todas demais componentes igual a $0$. Essas etiquetas são comparadas com a saída calculada, através da função de custo.

<center><img src="https://drive.google.com/uc?id=1UZ8m6dnU9x7HGPWLZeGF3LgSTFxiBjhW" width="500" /></center>

## Softmax

Nesse tipo de problema a saída $y$ é ativada pela função _softmax_ onde:

$y_i = \frac{\exp(s_i)}{\sum_j\exp(s_j)}$

Isso resulta que todas saídas ficam contidas no intervalo entre $0$ e $1$, e somam exatamente no valor $1$, onde a saída maior se sobressai em relação às demais, já que a diferença aparece na potência da exponencial.

<img src="https://drive.google.com/uc?id=1d3sG1C3ON215tuhHmBR_fPpsKg1C6H0i" width="700" />

Nesse caso, quando formos comparar $y$ (gerado pela função _softmax_) e $\hat{y}$ (etiqueta no formato _one-hot_) que diferença faz usar MSE ou CE como função de custo?
